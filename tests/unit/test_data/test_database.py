"""
Unit tests for database connection management.

Tests DatabaseManager class, connection pooling, health checks,
retry logic, and database utility functions.
"""

import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
    TimeoutError as SQLTimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseConnectionError, DatabaseQueryError
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
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        manager.engine = mock_engine

        await manager._verify_connection()

        # Should execute basic connectivity test
        assert mock_conn.execute.call_count >= 1

        # Should check for TimescaleDB extension
        calls = mock_conn.execute.call_args_list
        timescale_calls = [call for call in calls if "timescaledb" in str(call)]
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
    async def test_get_session_success(self, test_db_manager):
        """Test successful session creation and cleanup."""
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.is_active = False

        test_db_manager.session_factory = Mock(return_value=mock_session)

        async with test_db_manager.get_session() as session:
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
    async def test_execute_query_success(self, test_db_manager):
        """Test successful query execution."""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = ("result",)
        mock_result.fetchall.return_value = [("row1",), ("row2",)]
        mock_session.execute.return_value = mock_result

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        # Test fetch_one
        result = await test_db_manager.execute_query(
            "SELECT 1", parameters={"param": "value"}, fetch_one=True
        )
        assert result == ("result",)

        # Test fetch_all
        result = await test_db_manager.execute_query(
            "SELECT * FROM table", fetch_all=True
        )
        assert result == [("row1",), ("row2",)]

        # Test no fetch (return result object)
        result = await test_db_manager.execute_query("UPDATE table SET col=1")
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_query_error(self, test_db_manager):
        """Test query execution error handling."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Query failed")

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        with pytest.raises(DatabaseQueryError):
            await test_db_manager.execute_query("INVALID SQL")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, test_db_manager):
        """Test health check when database is healthy."""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        test_db_manager.engine = Mock()
        test_db_manager.engine.pool.size.return_value = 5
        test_db_manager.engine.pool.checkedout.return_value = 2

        health = await test_db_manager.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "connection_stats" in health
        assert "performance_metrics" in health
        assert health["performance_metrics"]["response_time_ms"] >= 0
        assert health["performance_metrics"]["pool_size"] == 5
        assert health["performance_metrics"]["checked_out_connections"] == 2

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, test_db_manager):
        """Test health check when database is unhealthy."""
        test_db_manager.get_session = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        health = await test_db_manager.health_check()

        assert health["status"] == "unhealthy"
        assert len(health["errors"]) > 0
        assert health["errors"][0]["type"] == "health_check_failed"

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_available(self, test_db_manager):
        """Test health check with TimescaleDB available."""
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

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        test_db_manager.engine = Mock()
        test_db_manager.engine.pool.size.return_value = 0
        test_db_manager.engine.pool.checkedout.return_value = 0

        health = await test_db_manager.health_check()

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
    async def test_health_check_timescaledb_unavailable(self, test_db_manager):
        """Test health check with TimescaleDB unavailable."""
        mock_session = AsyncMock()

        def mock_execute(query):
            mock_result = Mock()
            if "SELECT 1" in str(query):
                mock_result.scalar.return_value = 1
            elif "get_version_info" in str(query):
                raise Exception("TimescaleDB not available")
            return mock_result

        mock_session.execute.side_effect = mock_execute

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        test_db_manager.engine = Mock()
        test_db_manager.engine.pool.size.return_value = 0
        test_db_manager.engine.pool.checkedout.return_value = 0

        health = await test_db_manager.health_check()

        assert health["timescale_status"] == "unavailable"
        # Test that version info includes error information
        assert "timescale_version_info" in health
        version_info = health["timescale_version_info"]
        assert "error" in version_info
        assert "TimescaleDB not available" in version_info["error"]

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_version_parsing(self, test_db_manager):
        """Test TimescaleDB version information parsing with various formats."""
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
            with self.subTest(version_string=version_string):

                def mock_execute(query):
                    mock_result = Mock()
                    if "SELECT 1" in str(query):
                        mock_result.scalar.return_value = 1
                    elif "get_version_info" in str(query):
                        mock_result.fetchone.return_value = (version_string,)
                    return mock_result

                mock_session.execute.side_effect = mock_execute

                test_db_manager.get_session = AsyncMock()
                test_db_manager.get_session.return_value.__aenter__.return_value = (
                    mock_session
                )
                test_db_manager.engine = Mock()
                test_db_manager.engine.pool.size.return_value = 0
                test_db_manager.engine.pool.checkedout.return_value = 0

                health = await test_db_manager.health_check()

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
    async def test_health_check_timescaledb_version_parsing_error(
        self, test_db_manager
    ):
        """Test TimescaleDB version parsing with fetchone() returning None."""
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

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        test_db_manager.engine = Mock()
        test_db_manager.engine.pool.size.return_value = 0
        test_db_manager.engine.pool.checkedout.return_value = 0

        health = await test_db_manager.health_check()

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

        # Mock components
        mock_task = Mock()
        mock_task.done.return_value = False
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
        mock_manager.get_session.return_value.__aenter__.return_value = mock_session

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
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
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
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
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_with_previous_errors(self, test_db_manager):
        """Test health check reporting previous connection errors."""
        # Simulate previous connection error
        test_db_manager._connection_stats["last_connection_error"] = "Previous error"
        test_db_manager._connection_stats["failed_connections"] = 5

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        test_db_manager.get_session = AsyncMock()
        test_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        test_db_manager.engine = Mock()
        test_db_manager.engine.pool.size.return_value = 0
        test_db_manager.engine.pool.checkedout.return_value = 0

        health = await test_db_manager.health_check()

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

        # Initialize
        await manager.initialize()
        assert manager.is_initialized
        assert manager.engine is not None
        assert manager.session_factory is not None

        # Test session usage
        async with manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        # Test health check
        health = await manager.health_check()
        assert health["status"] == "healthy"

        # Test query execution
        result = await manager.execute_query("SELECT 2", fetch_one=True)
        assert result[0] == 2

        # Clean up
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
        await manager.initialize()

        async def use_session(session_id):
            async with manager.get_session() as session:
                result = await session.execute(text(f"SELECT {session_id}"))
                return result.scalar()

        # Run multiple sessions concurrently
        tasks = [use_session(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)

        assert results == [1, 2, 3, 4, 5]

        await manager.close()

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

        await manager.initialize()

        # Mock session factory to fail then succeed
        original_factory = manager.session_factory
        call_count = 0

        def mock_factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OperationalError("Temporary failure", None, None)
            return original_factory()

        manager.session_factory = mock_factory

        # Should succeed after retry
        async with manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        assert call_count == 2  # Failed once, succeeded on retry

        await manager.close()


# Helper function for mock_open
def mock_open(read_data=""):
    """Create a mock for open() that returns read_data when read."""
    from unittest.mock import mock_open as base_mock_open

    return base_mock_open(read_data=read_data)
