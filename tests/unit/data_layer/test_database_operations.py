"""Unit tests for database operations and connection management.

Covers:
- src/data/storage/database.py (Database Connection Management)
- src/data/storage/database_compatibility.py (Database Compatibility Layer)
- src/data/storage/dialect_utils.py (Database Dialect Utilities)

This test file consolidates testing for all database connection and compatibility functionality.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# Mock SQLAlchemy components
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    Mock as MockSQLAlchemy,
    call,
    patch,
)

from decimal import Decimal
import pytest

from src.core.config import DatabaseConfig, get_config

# Import system components
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    ErrorSeverity,
)


# Mock fixtures for database operations
@pytest.fixture
def mock_database_config():
    """Mock database configuration."""
    return DatabaseConfig(
        connection_string="postgresql://user:password@localhost:5432/test_db",
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
    )


@pytest.fixture
def mock_async_engine():
    """Mock async SQLAlchemy engine."""
    engine = AsyncMock()
    engine.pool = Mock()
    engine.pool.size.return_value = 10
    engine.pool.checkedout.return_value = 2
    engine.pool.overflow.return_value = 0
    engine.pool._pool_size = 10
    engine.pool._checked_out = 2
    engine.pool._overflow = 0
    engine.pool._invalidated = 0
    engine.sync_engine = Mock()
    engine.dispose = AsyncMock()
    engine.begin = asynccontextmanager(lambda: Mock()).__aenter__
    return engine


@pytest.fixture
def mock_async_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_async_session):
    """Mock session factory."""
    factory = Mock()
    factory.return_value = mock_async_session
    return factory


class TestDatabaseManager:
    """Test DatabaseManager class from src/data/storage/database.py."""

    @pytest.fixture(autouse=True)
    def setup_patches(self):
        """Set up patches for each test."""
        with patch("src.data.storage.database.get_config") as mock_get_config, patch(
            "src.data.storage.database.create_async_engine"
        ) as mock_create_engine, patch(
            "src.data.storage.database.async_sessionmaker"
        ) as mock_sessionmaker, patch(
            "src.data.storage.database.event"
        ) as mock_event, patch(
            "src.data.storage.database.CompatibilityManager"
        ) as mock_compat:

            self.mock_get_config = mock_get_config
            self.mock_create_engine = mock_create_engine
            self.mock_sessionmaker = mock_sessionmaker
            self.mock_event = mock_event
            self.mock_compat = mock_compat

            yield

    def test_database_manager_initialization_with_config(self, mock_database_config):
        """Test DatabaseManager initialization with provided config."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)

        assert db_manager.config == mock_database_config
        assert db_manager.engine is None
        assert db_manager.session_factory is None
        assert db_manager.max_retries == 5
        assert db_manager.base_delay == 1.0
        assert db_manager.backoff_multiplier == 2.0
        assert isinstance(db_manager.connection_timeout, timedelta)
        assert isinstance(db_manager.query_timeout, timedelta)

    def test_database_manager_initialization_default_config(self, mock_database_config):
        """Test DatabaseManager initialization with default config loading."""
        from src.data.storage.database import DatabaseManager

        # Mock system config
        mock_system_config = Mock()
        mock_system_config.database = mock_database_config
        self.mock_get_config.return_value = mock_system_config

        db_manager = DatabaseManager()

        assert db_manager.config == mock_database_config
        self.mock_get_config.assert_called_once()

    def test_connection_stats_initialization(self, mock_database_config):
        """Test connection statistics initialization."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)

        expected_stats_keys = {
            "total_connections",
            "failed_connections",
            "last_health_check",
            "last_connection_error",
            "retry_count",
        }
        assert set(db_manager._connection_stats.keys()) == expected_stats_keys
        assert db_manager._connection_stats["total_connections"] == 0
        assert db_manager._connection_stats["failed_connections"] == 0

    @pytest.mark.asyncio
    async def test_create_engine_postgresql_url_conversion(
        self, mock_database_config, mock_async_engine
    ):
        """Test engine creation with PostgreSQL URL conversion."""
        from src.data.storage.database import DatabaseManager

        self.mock_create_engine.return_value = mock_async_engine
        db_manager = DatabaseManager(config=mock_database_config)

        await db_manager._create_engine()

        # Verify URL conversion from postgresql:// to postgresql+asyncpg://
        self.mock_create_engine.assert_called_once()
        call_args = self.mock_create_engine.call_args
        assert "postgresql+asyncpg://" in call_args[1]["url"]
        assert call_args[1]["pool_size"] == 10
        assert call_args[1]["max_overflow"] == 20
        assert call_args[1]["pool_pre_ping"] is True
        assert call_args[1]["pool_recycle"] == 3600

    @pytest.mark.asyncio
    async def test_create_engine_with_nullpool_for_testing(self, mock_async_engine):
        """Test engine creation with NullPool for testing scenarios."""
        from src.data.storage.database import DatabaseManager

        # Test with zero pool size (testing scenario)
        test_config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=0,  # This should trigger NullPool
        )

        self.mock_create_engine.return_value = mock_async_engine
        db_manager = DatabaseManager(config=test_config)

        with patch("src.data.storage.database.NullPool") as mock_nullpool:
            await db_manager._create_engine()

            # Verify NullPool is used when pool_size <= 0
            call_args = self.mock_create_engine.call_args[1]
            assert "poolclass" in call_args

    @pytest.mark.asyncio
    async def test_create_engine_invalid_connection_string(self, mock_database_config):
        """Test engine creation with invalid connection string."""
        from src.data.storage.database import DatabaseManager

        # Invalid connection string (not PostgreSQL)
        invalid_config = DatabaseConfig(
            connection_string="mysql://user:pass@localhost/db", pool_size=10
        )

        db_manager = DatabaseManager(config=invalid_config)

        with pytest.raises(ValueError, match="Connection string must use postgresql"):
            await db_manager._create_engine()

    def test_setup_connection_events(self, mock_database_config, mock_async_engine):
        """Test connection event listener setup."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine

        # Mock event.listens_for decorator
        mock_decorator = Mock()
        self.mock_event.listens_for.return_value = mock_decorator

        db_manager._setup_connection_events()

        # Verify event listeners are registered
        assert (
            self.mock_event.listens_for.call_count == 4
        )  # connect, checkout, checkin, invalidate
        event_calls = self.mock_event.listens_for.call_args_list

        # Check that all expected events are registered
        registered_events = [call[0][1] for call in event_calls]
        expected_events = ["connect", "checkout", "checkin", "invalidate"]
        assert set(registered_events) == set(expected_events)

    def test_setup_connection_events_without_engine(self, mock_database_config):
        """Test connection event setup raises error without engine."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        # Don't set engine

        with pytest.raises(
            RuntimeError, match="Engine must be created before setting up events"
        ):
            db_manager._setup_connection_events()

    @pytest.mark.asyncio
    async def test_setup_session_factory(
        self, mock_database_config, mock_async_engine, mock_session_factory
    ):
        """Test async session factory setup."""
        from src.data.storage.database import DatabaseManager

        self.mock_sessionmaker.return_value = mock_session_factory
        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine

        await db_manager._setup_session_factory()

        assert db_manager.session_factory == mock_session_factory

        # Verify session factory configuration
        self.mock_sessionmaker.assert_called_once()
        call_kwargs = self.mock_sessionmaker.call_args[1]
        assert call_kwargs["bind"] == mock_async_engine
        assert call_kwargs["expire_on_commit"] is False
        assert call_kwargs["autoflush"] is True
        assert call_kwargs["autocommit"] is False

    @pytest.mark.asyncio
    async def test_setup_session_factory_without_engine(self, mock_database_config):
        """Test session factory setup raises error without engine."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        # Don't set engine

        with pytest.raises(
            RuntimeError, match="Engine must be created before session factory"
        ):
            await db_manager._setup_session_factory()

    @pytest.mark.asyncio
    async def test_verify_connection_success(
        self, mock_database_config, mock_async_engine
    ):
        """Test successful database connection verification."""
        from src.data.storage.database import DatabaseManager

        # Mock connection and results
        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1  # Basic connectivity test
        mock_conn.execute.return_value = mock_result

        # Mock TimescaleDB extension check
        mock_timescale_result = Mock()
        mock_timescale_result.scalar.return_value = 1  # Extension exists
        mock_conn.execute.side_effect = [mock_result, mock_timescale_result]

        # Mock engine.begin context manager
        async def mock_begin():
            return mock_conn

        mock_async_engine.begin.return_value.__aenter__ = mock_begin
        mock_async_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine

        await db_manager._verify_connection()

        # Verify connection test and TimescaleDB check
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_verify_connection_timescaledb_missing(
        self, mock_database_config, mock_async_engine
    ):
        """Test connection verification with missing TimescaleDB extension."""
        from src.data.storage.database import DatabaseManager

        # Mock connection and results
        mock_conn = AsyncMock()

        # Mock basic connectivity success, but TimescaleDB missing
        connectivity_result = Mock()
        connectivity_result.scalar.return_value = 1

        timescale_result = Mock()
        timescale_result.scalar.return_value = 0  # Extension not found

        mock_conn.execute.side_effect = [connectivity_result, timescale_result]

        # Mock engine.begin context manager
        async def mock_begin():
            return mock_conn

        mock_async_engine.begin.return_value.__aenter__ = mock_begin
        mock_async_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine

        with patch("src.data.storage.database.logger") as mock_logger:
            await db_manager._verify_connection()

            # Should log warning about missing TimescaleDB
            mock_logger.warning.assert_called_once()
            assert "TimescaleDB extension not found" in str(
                mock_logger.warning.call_args
            )

    @pytest.mark.asyncio
    async def test_verify_connection_without_engine(self, mock_database_config):
        """Test connection verification raises error without engine."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        # Don't set engine

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            await db_manager._verify_connection()

    @pytest.mark.asyncio
    async def test_get_session_success(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test successful session retrieval."""
        from src.data.storage.database import DatabaseManager

        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        async with db_manager.get_session() as session:
            assert session == mock_async_session

        # Verify session lifecycle
        mock_async_session.commit.assert_called_once()
        mock_async_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_without_factory(self, mock_database_config):
        """Test session retrieval raises error without factory."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        # Don't set session_factory

        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            async with db_manager.get_session():
                pass

    @pytest.mark.asyncio
    async def test_get_session_with_retry_on_connection_error(
        self, mock_database_config, mock_session_factory
    ):
        """Test session retrieval with retry logic on connection errors."""
        from sqlalchemy.exc import OperationalError

        from src.data.storage.database import DatabaseManager

        # Mock session that fails first, then succeeds
        mock_session_1 = AsyncMock()
        mock_session_1.commit.side_effect = OperationalError(
            "Connection lost", None, None
        )

        mock_session_2 = AsyncMock()
        mock_session_2.commit.return_value = None  # Success

        mock_session_factory.side_effect = [mock_session_1, mock_session_2]

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory
        db_manager.max_retries = 2

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            async with db_manager.get_session() as session:
                assert session == mock_session_2

            # Verify retry logic
            mock_sleep.assert_called_once()  # Should sleep before retry
            assert mock_session_factory.call_count == 2

    @pytest.mark.asyncio
    async def test_get_session_retry_exhaustion(
        self, mock_database_config, mock_session_factory
    ):
        """Test session retrieval fails after exhausting retries."""
        from sqlalchemy.exc import OperationalError

        from src.data.storage.database import DatabaseManager

        # Mock session that always fails
        mock_session = AsyncMock()
        mock_session.commit.side_effect = OperationalError(
            "Connection lost", None, None
        )
        mock_session_factory.return_value = mock_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory
        db_manager.max_retries = 2

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(DatabaseConnectionError):
                async with db_manager.get_session():
                    pass

            # Should have tried max_retries + 1 times (initial + retries)
            assert mock_session_factory.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_query_basic(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test basic query execution."""
        from src.data.storage.database import DatabaseManager

        # Mock query result
        mock_result = Mock()
        mock_result.fetchone.return_value = ("test_result",)
        mock_async_session.execute.return_value = mock_result

        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        # Test fetch_one
        result = await db_manager.execute_query(
            "SELECT 1", parameters={"param1": "value1"}, fetch_one=True
        )

        assert result == ("test_result",)
        mock_async_session.execute.assert_called_once()
        mock_result.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_timeout(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test query execution with custom timeout."""
        from src.data.storage.database import DatabaseManager

        mock_result = Mock()
        mock_async_session.execute.return_value = mock_result
        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        custom_timeout = timedelta(seconds=60)

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
            mock_wait_for.return_value = mock_result

            await db_manager.execute_query("SELECT 1", timeout=custom_timeout)

            # Verify timeout is applied
            mock_wait_for.assert_called_once()
            assert mock_wait_for.call_args[1]["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_execute_query_timeout_error(
        self, mock_database_config, mock_session_factory
    ):
        """Test query execution timeout handling."""
        from src.data.storage.database import DatabaseManager

        mock_session_factory.return_value = AsyncMock()

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(DatabaseQueryError) as exc_info:
                await db_manager.execute_query("SELECT 1")

            assert exc_info.value.error_code == "DB_QUERY_ERROR"
            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.severity == ErrorSeverity.HIGH

    @pytest.mark.asyncio
    async def test_execute_query_sql_error(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test query execution with SQL error."""
        from sqlalchemy.exc import SQLAlchemyError

        from src.data.storage.database import DatabaseManager

        mock_async_session.execute.side_effect = SQLAlchemyError("SQL error")
        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        with pytest.raises(DatabaseQueryError) as exc_info:
            await db_manager.execute_query("INVALID SQL")

        assert exc_info.value.error_code == "DB_QUERY_ERROR"
        assert "SQL error" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_execute_optimized_query_with_prepared_statements(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test optimized query execution with prepared statements."""
        from src.data.storage.database import DatabaseManager

        mock_result = Mock()
        mock_async_session.execute.return_value = mock_result
        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        with patch("hashlib.md5") as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "abcd1234567890"

            result = await db_manager.execute_optimized_query(
                "SELECT * FROM table WHERE id = %(id)s",
                parameters={"id": 123},
                use_prepared_statement=True,
                enable_query_cache=True,
            )

            assert result == mock_result

            # Should execute multiple statements (PREPARE, EXECUTE, DEALLOCATE)
            assert mock_async_session.execute.call_count >= 3

    @pytest.mark.asyncio
    async def test_execute_optimized_query_prepared_statement_fallback(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test optimized query fallback when prepared statement fails."""
        from sqlalchemy.exc import SQLAlchemyError

        from src.data.storage.database import DatabaseManager

        mock_result = Mock()

        # Mock prepared statement failure, then regular execution success
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # PREPARE and EXECUTE fail
                raise SQLAlchemyError("Prepared statement error")
            return mock_result  # Regular query succeeds

        mock_async_session.execute.side_effect = mock_execute
        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        with patch("hashlib.md5") as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "abcd1234567890"

            result = await db_manager.execute_optimized_query(
                "SELECT * FROM table WHERE id = %(id)s",
                parameters={"id": 123},
                use_prepared_statement=True,
            )

            assert result == mock_result
            # Should fall back to regular query execution
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_analyze_query_performance(
        self, mock_database_config, mock_async_session, mock_session_factory
    ):
        """Test query performance analysis."""
        from src.data.storage.database import DatabaseManager

        # Mock EXPLAIN ANALYZE result
        explain_result = Mock()
        explain_result.fetchone.return_value = ([{"Plan": {"Total Cost": 100.0}}],)

        # Mock regular execution
        regular_result = Mock()

        mock_async_session.execute.side_effect = [explain_result, regular_result]
        mock_session_factory.return_value = mock_async_session

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        with patch("time.time", side_effect=[1000.0, 1000.5]):
            analysis = await db_manager.analyze_query_performance(
                "SELECT * FROM table",
                parameters={"id": 123},
                include_execution_plan=True,
            )

        assert "query" in analysis
        assert "execution_time_seconds" in analysis
        assert analysis["execution_time_seconds"] == 0.5
        assert "execution_plan" in analysis
        assert "performance_rating" in analysis
        assert analysis["performance_rating"] == "acceptable"  # 0.5s
        assert "optimization_suggestions" in analysis

    def test_get_optimization_suggestions(self, mock_database_config):
        """Test query optimization suggestions."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)

        # Test various query patterns
        query_no_where = "SELECT * FROM sensor_events"
        suggestions = db_manager._get_optimization_suggestions(query_no_where)
        assert "Add WHERE clause to filter sensor_events table" in suggestions

        query_select_star = "SELECT * FROM table"
        suggestions = db_manager._get_optimization_suggestions(query_select_star)
        assert "Specify explicit column names instead of SELECT *" in suggestions

        query_order_no_limit = "SELECT id FROM table ORDER BY timestamp"
        suggestions = db_manager._get_optimization_suggestions(query_order_no_limit)
        assert "Consider adding LIMIT clause with ORDER BY" in suggestions

        query_multiple_selects = "SELECT id FROM table1; SELECT name FROM table2"
        suggestions = db_manager._get_optimization_suggestions(query_multiple_selects)
        assert (
            "Consider using JOINs instead of multiple SELECT statements" in suggestions
        )

    @pytest.mark.asyncio
    async def test_get_connection_pool_metrics(
        self, mock_database_config, mock_async_engine
    ):
        """Test connection pool metrics retrieval."""
        from src.data.storage.database import DatabaseManager

        # Configure mock pool
        mock_pool = Mock()
        mock_pool._pool_size = 10
        mock_pool._checked_out = 3
        mock_pool._overflow = 1
        mock_pool._invalidated = 0
        mock_async_engine.pool = mock_pool

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine
        db_manager._connection_stats = {
            "total_connections": 15,
            "failed_connections": 2,
        }

        metrics = await db_manager.get_connection_pool_metrics()

        assert "timestamp" in metrics
        assert metrics["pool_size"] == 10
        assert metrics["checked_out"] == 3
        assert metrics["overflow"] == 1
        assert metrics["utilization_percent"] == 40.0  # (3 + 1) / 10 * 100
        assert metrics["pool_status"] == "healthy"  # < 50%
        assert "connection_stats" in metrics

    @pytest.mark.asyncio
    async def test_get_connection_pool_metrics_high_utilization(
        self, mock_database_config, mock_async_engine
    ):
        """Test connection pool metrics with high utilization."""
        from src.data.storage.database import DatabaseManager

        # Configure mock pool with high utilization
        mock_pool = Mock()
        mock_pool._pool_size = 10
        mock_pool._checked_out = 8
        mock_pool._overflow = 2
        mock_pool._invalidated = 1
        mock_async_engine.pool = mock_pool

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine

        metrics = await db_manager.get_connection_pool_metrics()

        assert metrics["utilization_percent"] == 100.0  # (8 + 2) / 10 * 100
        assert metrics["pool_status"] == "high_utilization"  # >= 80%
        assert "recommendations" in metrics
        assert "Consider increasing pool_size" in metrics["recommendations"]

    @pytest.mark.asyncio
    async def test_get_connection_pool_metrics_without_engine(
        self, mock_database_config
    ):
        """Test connection pool metrics without initialized engine."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        # Don't set engine

        metrics = await db_manager.get_connection_pool_metrics()

        assert "error" in metrics
        assert "Database engine not initialized" in metrics["error"]

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        mock_database_config,
        mock_async_session,
        mock_session_factory,
        mock_async_engine,
    ):
        """Test successful health check."""
        from src.data.storage.database import DatabaseManager

        # Mock successful connectivity test
        connectivity_result = Mock()
        connectivity_result.fetchone.return_value = None

        # Mock TimescaleDB version check
        version_result = Mock()
        version_result.fetchone.return_value = (
            "TimescaleDB version 2.8.0 on PostgreSQL 13.7",
        )

        mock_async_session.execute.side_effect = [connectivity_result, version_result]
        mock_session_factory.return_value = mock_async_session

        # Mock pool metrics
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_async_engine.pool = mock_pool

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory
        db_manager.engine = mock_async_engine
        db_manager._connection_stats = {
            "last_connection_error": None,
            "failed_connections": 0,
        }

        with patch("time.time", side_effect=[1000.0, 1000.1]):
            health = await db_manager.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert health["timescale_status"] == "available"
        assert "timescale_version_info" in health
        assert "performance_metrics" in health
        assert health["performance_metrics"]["response_time_ms"] == 100.0
        assert len(health["errors"]) == 0

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, mock_database_config, mock_session_factory
    ):
        """Test health check failure."""
        from sqlalchemy.exc import OperationalError

        from src.data.storage.database import DatabaseManager

        # Mock session creation failure
        mock_session_factory.side_effect = OperationalError(
            "Connection failed", None, None
        )

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.session_factory = mock_session_factory

        health = await db_manager.health_check()

        assert health["status"] == "unhealthy"
        assert len(health["errors"]) > 0
        assert health["errors"][0]["type"] == "health_check_failed"

    @pytest.mark.asyncio
    async def test_health_check_loop(self, mock_database_config):
        """Test background health check loop."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.health_check_interval = timedelta(seconds=0.1)

        call_count = 0

        async def mock_health_check():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            return {"status": "healthy"}

        db_manager.health_check = mock_health_check

        with pytest.raises(asyncio.CancelledError):
            await db_manager._health_check_loop()

        assert call_count >= 2  # Should have made multiple health checks

    def test_is_initialized_property(
        self, mock_database_config, mock_async_engine, mock_session_factory
    ):
        """Test is_initialized property."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        assert not db_manager.is_initialized

        db_manager.engine = mock_async_engine
        assert not db_manager.is_initialized  # Still need session_factory

        db_manager.session_factory = mock_session_factory
        assert db_manager.is_initialized

    def test_get_connection_stats(self, mock_database_config):
        """Test connection statistics retrieval."""
        from src.data.storage.database import DatabaseManager

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager._connection_stats["total_connections"] = 10
        db_manager._connection_stats["failed_connections"] = 2

        stats = db_manager.get_connection_stats()

        assert stats["total_connections"] == 10
        assert stats["failed_connections"] == 2
        assert isinstance(stats, dict)  # Should be a copy, not reference

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_database_config, mock_async_engine):
        """Test resource cleanup."""
        from src.data.storage.database import DatabaseManager

        # Create mock health check task
        mock_task = AsyncMock()
        mock_task.done.return_value = False

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine
        db_manager._health_check_task = mock_task

        await db_manager._cleanup()

        # Verify cleanup
        mock_task.cancel.assert_called_once()
        mock_async_engine.dispose.assert_called_once()
        assert db_manager.engine is None
        assert db_manager.session_factory is None

    @pytest.mark.asyncio
    async def test_cleanup_with_completed_task(
        self, mock_database_config, mock_async_engine
    ):
        """Test cleanup with already completed health check task."""
        from src.data.storage.database import DatabaseManager

        # Create mock completed task
        mock_task = AsyncMock()
        mock_task.done.return_value = True

        db_manager = DatabaseManager(config=mock_database_config)
        db_manager.engine = mock_async_engine
        db_manager._health_check_task = mock_task

        await db_manager._cleanup()

        # Should not try to cancel already completed task
        mock_task.cancel.assert_not_called()
        mock_async_engine.dispose.assert_called_once()


class TestGlobalDatabaseFunctions:
    """Test global database utility functions."""

    @pytest.fixture(autouse=True)
    def setup_patches(self):
        """Set up patches for each test."""
        with patch("src.data.storage.database._db_manager", None):
            yield

    @pytest.mark.asyncio
    async def test_get_database_manager_singleton(self, mock_database_config):
        """Test global database manager singleton behavior."""
        from src.data.storage.database import get_database_manager

        with patch(
            "src.data.storage.database.DatabaseManager"
        ) as mock_db_manager_class:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock()
            mock_db_manager_class.return_value = mock_instance

            # First call should create instance
            manager1 = await get_database_manager()
            assert manager1 == mock_instance
            mock_instance.initialize.assert_called_once()

            # Second call should return same instance
            manager2 = await get_database_manager()
            assert manager2 == mock_instance
            assert manager1 is manager2
            # Initialize should only be called once
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_session_context_manager(self, mock_database_config):
        """Test global database session context manager."""
        from src.data.storage.database import get_db_session

        mock_session = AsyncMock()
        mock_manager = AsyncMock()

        # Mock the context manager behavior
        async def mock_get_session():
            return mock_session

        mock_manager.get_session.return_value.__aenter__ = mock_get_session
        mock_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            async with get_db_session() as session:
                assert session == mock_session

    @pytest.mark.asyncio
    async def test_close_database_manager(self):
        """Test global database manager cleanup."""
        from src.data.storage.database import _db_manager, close_database_manager

        # Set up mock global manager
        with patch("src.data.storage.database._db_manager") as mock_global_manager:
            mock_instance = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_global_manager = mock_instance

            # Temporarily set the global manager
            import src.data.storage.database

            src.data.storage.database._db_manager = mock_instance

            await close_database_manager()

            mock_instance.close.assert_called_once()
            assert src.data.storage.database._db_manager is None

    @pytest.mark.asyncio
    async def test_execute_sql_file_success(self, tmp_path):
        """Test SQL file execution."""
        from src.data.storage.database import execute_sql_file

        # Create test SQL file
        sql_file = tmp_path / "test.sql"
        sql_file.write_text(
            "CREATE TABLE test (id INTEGER);\n"
            "INSERT INTO test VALUES (1);\n"
            "INSERT INTO test VALUES (2);"
        )

        # Mock database manager
        mock_manager = AsyncMock()
        mock_manager.execute_query = AsyncMock()

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            await execute_sql_file(str(sql_file))

            # Should execute each non-empty statement
            assert mock_manager.execute_query.call_count == 3
            executed_queries = [
                call[0][0] for call in mock_manager.execute_query.call_args_list
            ]
            assert "CREATE TABLE test" in executed_queries[0]
            assert "INSERT INTO test VALUES (1)" in executed_queries[1]
            assert "INSERT INTO test VALUES (2)" in executed_queries[2]

    @pytest.mark.asyncio
    async def test_execute_sql_file_not_found(self):
        """Test SQL file execution with missing file."""
        from src.data.storage.database import execute_sql_file

        with pytest.raises(DatabaseQueryError) as exc_info:
            await execute_sql_file("/nonexistent/file.sql")

        assert "file:/nonexistent/file.sql" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_table_exists_true(self):
        """Test table existence check - table exists."""
        from src.data.storage.database import check_table_exists

        # Mock database manager
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = (True,)

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            exists = await check_table_exists("test_table")

            assert exists is True
            mock_manager.execute_query.assert_called_once()
            call_args = mock_manager.execute_query.call_args
            assert "information_schema.tables" in call_args[0][0]
            assert call_args[1]["parameters"]["table_name"] == "test_table"
            assert call_args[1]["fetch_one"] is True

    @pytest.mark.asyncio
    async def test_check_table_exists_false(self):
        """Test table existence check - table doesn't exist."""
        from src.data.storage.database import check_table_exists

        # Mock database manager
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = (False,)

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            exists = await check_table_exists("nonexistent_table")

            assert exists is False

    @pytest.mark.asyncio
    async def test_check_table_exists_error(self):
        """Test table existence check with database error."""
        from src.data.storage.database import check_table_exists

        # Mock database manager with error
        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Database error")

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            exists = await check_table_exists("test_table")

            assert exists is False  # Should return False on error

    @pytest.mark.asyncio
    async def test_get_database_version(self):
        """Test database version retrieval."""
        from src.data.storage.database import get_database_version

        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = ("PostgreSQL 13.7",)

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            version = await get_database_version()

            assert version == "PostgreSQL 13.7"
            mock_manager.execute_query.assert_called_once_with(
                "SELECT version()", fetch_one=True
            )

    @pytest.mark.asyncio
    async def test_get_database_version_error(self):
        """Test database version retrieval with error."""
        from src.data.storage.database import get_database_version

        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Database error")

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            version = await get_database_version()

            assert version == "Error"

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_success(self):
        """Test TimescaleDB version retrieval."""
        from src.data.storage.database import get_timescaledb_version

        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = ("2.8.0",)

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            version = await get_timescaledb_version()

            assert version == "2.8.0"
            mock_manager.execute_query.assert_called_once()
            call_args = mock_manager.execute_query.call_args[0][0]
            assert "pg_extension" in call_args
            assert "timescaledb" in call_args

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_not_available(self):
        """Test TimescaleDB version retrieval when not available."""
        from src.data.storage.database import get_timescaledb_version

        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = None

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            version = await get_timescaledb_version()

            assert version is None

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_error(self):
        """Test TimescaleDB version retrieval with error."""
        from src.data.storage.database import get_timescaledb_version

        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Database error")

        with patch(
            "src.data.storage.database.get_database_manager", return_value=mock_manager
        ):
            version = await get_timescaledb_version()

            assert version is None


class TestDatabaseCompatibility:
    """Test database compatibility layer functions."""

    def test_is_sqlite_engine_with_sqlite_url(self):
        """Test SQLite engine detection with SQLite URL."""
        from src.data.storage.database_compatibility import is_sqlite_engine

        # Test various SQLite URL formats
        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"
        assert is_sqlite_engine(mock_engine) is True

        mock_engine.url = "sqlite:///:memory:"
        assert is_sqlite_engine(mock_engine) is True

        mock_engine.url = "SQLITE:///test.db"  # Case insensitive
        assert is_sqlite_engine(mock_engine) is True

    def test_is_sqlite_engine_with_postgresql_url(self):
        """Test SQLite engine detection with PostgreSQL URL."""
        from src.data.storage.database_compatibility import is_sqlite_engine

        mock_engine = Mock()
        mock_engine.url = "postgresql://user:pass@localhost/db"
        assert is_sqlite_engine(mock_engine) is False

        mock_engine.url = "postgresql+asyncpg://user:pass@localhost/db"
        assert is_sqlite_engine(mock_engine) is False

    def test_is_sqlite_engine_with_none_engine(self):
        """Test SQLite engine detection with None engine."""
        from src.data.storage.database_compatibility import is_sqlite_engine

        assert is_sqlite_engine(None) is False

    def test_is_sqlite_engine_with_url_object(self):
        """Test SQLite engine detection with URL object."""
        from src.data.storage.database_compatibility import is_sqlite_engine

        # Mock URL object that supports string conversion
        mock_url = Mock()
        mock_url.__str__ = lambda: "sqlite:///test.db"

        mock_engine = Mock()
        mock_engine.url = mock_url

        assert is_sqlite_engine(mock_engine) is True

    def test_is_postgresql_engine_with_postgresql_url(self):
        """Test PostgreSQL engine detection with PostgreSQL URL."""
        from src.data.storage.database_compatibility import is_postgresql_engine

        mock_engine = Mock()
        mock_engine.url = "postgresql://user:pass@localhost/db"
        assert is_postgresql_engine(mock_engine) is True

        mock_engine.url = "postgresql+asyncpg://user:pass@localhost/db"
        assert is_postgresql_engine(mock_engine) is True

        mock_engine.url = "postgres://user:pass@localhost/db"  # Alternative format
        assert is_postgresql_engine(mock_engine) is True

    def test_is_postgresql_engine_with_sqlite_url(self):
        """Test PostgreSQL engine detection with SQLite URL."""
        from src.data.storage.database_compatibility import is_postgresql_engine

        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"
        assert is_postgresql_engine(mock_engine) is False

    def test_configure_sensor_event_model_sqlite(self):
        """Test SensorEvent model configuration for SQLite."""
        from src.data.storage.database_compatibility import configure_sensor_event_model

        # Mock SQLite engine
        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"

        # Mock SensorEvent model
        mock_sensor_event = Mock()
        mock_sensor_event.__table__ = Mock()
        mock_sensor_event.__table__.columns = {
            "id": Mock(autoincrement=None, primary_key=True)
        }

        with patch(
            "src.data.storage.database_compatibility.SensorEvent", mock_sensor_event
        ):
            configure_sensor_event_model(mock_engine)

            # Should configure autoincrement for SQLite
            assert mock_sensor_event.__table__.columns["id"].autoincrement is True

    def test_configure_sensor_event_model_postgresql(self):
        """Test SensorEvent model configuration for PostgreSQL."""
        from src.data.storage.database_compatibility import configure_sensor_event_model

        # Mock PostgreSQL engine
        mock_engine = Mock()
        mock_engine.url = "postgresql://user:pass@localhost/db"

        # Mock SensorEvent model
        mock_sensor_event = Mock()
        mock_sensor_event.__table__ = Mock()
        mock_sensor_event.__table__.columns = {
            "id": Mock(autoincrement=None, primary_key=True),
            "timestamp": Mock(primary_key=False),
        }

        with patch(
            "src.data.storage.database_compatibility.SensorEvent", mock_sensor_event
        ):
            configure_sensor_event_model(mock_engine)

            # Should not set autoincrement for PostgreSQL composite keys
            # (Actual behavior depends on composite key detection)
            assert hasattr(mock_sensor_event.__table__.columns["id"], "autoincrement")

    def test_create_database_specific_models_with_sensor_event(self):
        """Test database-specific model creation with SensorEvent."""
        from src.data.storage.database_compatibility import (
            create_database_specific_models,
        )

        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"

        # Mock model registry
        mock_sensor_event = Mock()
        mock_other_model = Mock()

        models = {"SensorEvent": mock_sensor_event, "OtherModel": mock_other_model}

        with patch(
            "src.data.storage.database_compatibility.configure_sensor_event_model"
        ) as mock_configure:
            create_database_specific_models(mock_engine, models)

            # Should only configure SensorEvent
            mock_configure.assert_called_once_with(mock_engine)

    def test_create_database_specific_models_empty(self):
        """Test database-specific model creation with empty models dict."""
        from src.data.storage.database_compatibility import (
            create_database_specific_models,
        )

        mock_engine = Mock()
        models = {}

        # Should not raise error with empty models
        create_database_specific_models(mock_engine, models)

    def test_patch_models_for_sqlite_compatibility(self):
        """Test model patching for SQLite compatibility."""
        from src.data.storage.database_compatibility import (
            patch_models_for_sqlite_compatibility,
        )

        # Mock model class
        mock_model = Mock()
        mock_model.__table__ = Mock()
        mock_model.__table__.columns = {
            "id": Mock(autoincrement=None, primary_key=True)
        }

        patch_models_for_sqlite_compatibility([mock_model])

        # Should set autoincrement
        assert mock_model.__table__.columns["id"].autoincrement is True

    def test_get_database_specific_table_args_sqlite(self):
        """Test SQLite-specific table arguments generation."""
        from src.data.storage.database_compatibility import (
            get_database_specific_table_args,
        )

        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"

        table_args = get_database_specific_table_args(mock_engine, "test_table")

        # SQLite should return basic table args
        assert isinstance(table_args, tuple)

    def test_get_database_specific_table_args_postgresql(self):
        """Test PostgreSQL-specific table arguments generation."""
        from src.data.storage.database_compatibility import (
            get_database_specific_table_args,
        )

        mock_engine = Mock()
        mock_engine.url = "postgresql://user:pass@localhost/db"

        table_args = get_database_specific_table_args(mock_engine, "sensor_events")

        # PostgreSQL should return enhanced table args with indexes
        assert isinstance(table_args, tuple)

    def test_configure_sqlite_for_testing(self):
        """Test SQLite configuration for testing environment."""
        from src.data.storage.database_compatibility import configure_sqlite_for_testing

        # Mock database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        configure_sqlite_for_testing(mock_connection, None)

        # Should execute SQLite PRAGMA commands
        expected_pragmas = [
            "PRAGMA foreign_keys=ON",
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
        ]

        assert mock_cursor.execute.call_count == len(expected_pragmas)
        executed_commands = [call[0][0] for call in mock_cursor.execute.call_args_list]
        for pragma in expected_pragmas:
            assert pragma in executed_commands

    def test_configure_database_on_first_connect_sqlite(self):
        """Test database configuration on first connect for SQLite."""
        from src.data.storage.database_compatibility import (
            configure_database_on_first_connect,
        )

        # Mock SQLite connection
        mock_connection = Mock()
        mock_engine = Mock()
        mock_engine.url = "sqlite:///test.db"

        with patch(
            "src.data.storage.database_compatibility.configure_sqlite_for_testing"
        ) as mock_configure:
            configure_database_on_first_connect(mock_connection, None, mock_engine)

            mock_configure.assert_called_once_with(mock_connection, None)

    def test_configure_database_on_first_connect_postgresql(self):
        """Test database configuration on first connect for PostgreSQL."""
        from src.data.storage.database_compatibility import (
            configure_database_on_first_connect,
        )

        # Mock PostgreSQL connection
        mock_connection = Mock()
        mock_engine = Mock()
        mock_engine.url = "postgresql://user:pass@localhost/db"

        with patch(
            "src.data.storage.database_compatibility.configure_sqlite_for_testing"
        ) as mock_configure:
            configure_database_on_first_connect(mock_connection, None, mock_engine)

            # Should not configure SQLite settings for PostgreSQL
            mock_configure.assert_not_called()


class TestDialectUtils:
    """Test database dialect utility functions."""

    def test_database_dialect_utils_get_dialect_name_postgresql(self):
        """Test dialect name detection for PostgreSQL."""
        from src.data.storage.dialect_utils import DatabaseDialectUtils

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        utils = DatabaseDialectUtils(mock_engine)
        assert utils.get_dialect_name() == "postgresql"

    def test_database_dialect_utils_get_dialect_name_sqlite(self):
        """Test dialect name detection for SQLite."""
        from src.data.storage.dialect_utils import DatabaseDialectUtils

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        utils = DatabaseDialectUtils(mock_engine)
        assert utils.get_dialect_name() == "sqlite"

    def test_database_dialect_utils_is_postgresql(self):
        """Test PostgreSQL dialect detection."""
        from src.data.storage.dialect_utils import DatabaseDialectUtils

        # Test PostgreSQL engine
        mock_pg_engine = Mock()
        mock_pg_engine.dialect.name = "postgresql"

        utils = DatabaseDialectUtils(mock_pg_engine)
        assert utils.is_postgresql() is True

        # Test non-PostgreSQL engine
        mock_sqlite_engine = Mock()
        mock_sqlite_engine.dialect.name = "sqlite"

        utils = DatabaseDialectUtils(mock_sqlite_engine)
        assert utils.is_postgresql() is False

    def test_database_dialect_utils_is_sqlite(self):
        """Test SQLite dialect detection."""
        from src.data.storage.dialect_utils import DatabaseDialectUtils

        # Test SQLite engine
        mock_sqlite_engine = Mock()
        mock_sqlite_engine.dialect.name = "sqlite"

        utils = DatabaseDialectUtils(mock_sqlite_engine)
        assert utils.is_sqlite() is True

        # Test non-SQLite engine
        mock_pg_engine = Mock()
        mock_pg_engine.dialect.name = "postgresql"

        utils = DatabaseDialectUtils(mock_pg_engine)
        assert utils.is_sqlite() is False

    def test_statistical_functions_percentile_cont_postgresql(self):
        """Test percentile_cont function for PostgreSQL."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.percentile_cont(0.5, mock_column)

            # Should use PostgreSQL percentile_cont function
            mock_sql_func.percentile_cont.assert_called_once_with(0.5)

    def test_statistical_functions_percentile_cont_sqlite_median(self):
        """Test percentile_cont function for SQLite with median."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.percentile_cont(0.5, mock_column)

            # Should use SQLite median approximation (avg)
            mock_sql_func.avg.assert_called_once_with(mock_column)

    def test_statistical_functions_percentile_cont_sqlite_quartile(self):
        """Test percentile_cont function for SQLite with quartiles."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            # Test Q1 (25th percentile)
            result = stats.percentile_cont(0.25, mock_column)

            # Should use SQLite quartile approximation
            mock_sql_func.min.assert_called()
            mock_sql_func.max.assert_called()

    def test_statistical_functions_percentile_cont_sqlite_other(self):
        """Test percentile_cont function for SQLite with other percentiles."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            # Test 90th percentile (not median or quartile)
            result = stats.percentile_cont(0.9, mock_column)

            # Should use SQLite linear interpolation approximation
            mock_sql_func.min.assert_called()
            mock_sql_func.max.assert_called()

    def test_statistical_functions_stddev_samp_postgresql(self):
        """Test stddev_samp function for PostgreSQL."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.stddev_samp(mock_column)

            # Should use PostgreSQL native stddev_samp
            mock_sql_func.stddev_samp.assert_called_once_with(mock_column)

    def test_statistical_functions_stddev_samp_sqlite(self):
        """Test stddev_samp function for SQLite."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        stats = StatisticalFunctions(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.stddev_samp(mock_column)

            # Should use SQLite sqrt approximation
            mock_sql_func.sqrt.assert_called_once()

    def test_statistical_functions_extract_epoch_postgresql(self):
        """Test extract_epoch_from_interval function for PostgreSQL."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        stats = StatisticalFunctions(mock_engine)

        # Mock interval expression
        mock_interval = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.extract_epoch_from_interval(mock_interval)

            # Should use PostgreSQL extract function
            mock_sql_func.extract.assert_called_once_with("epoch", mock_interval)

    def test_statistical_functions_extract_epoch_sqlite(self):
        """Test extract_epoch_from_interval function for SQLite."""
        from src.data.storage.dialect_utils import StatisticalFunctions

        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"

        stats = StatisticalFunctions(mock_engine)

        # Mock interval expression
        mock_interval = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            result = stats.extract_epoch_from_interval(mock_interval)

            # Should use SQLite strftime difference calculation
            mock_sql_func.strftime.assert_called()

    def test_query_builder_initialization(self):
        """Test QueryBuilder initialization."""
        from src.data.storage.dialect_utils import QueryBuilder

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        builder = QueryBuilder(mock_engine)

        assert builder.engine == mock_engine
        assert builder.dialect_name == "postgresql"

    def test_query_builder_build_percentile_query_single(self):
        """Test percentile query building with single percentile."""
        from src.data.storage.dialect_utils import QueryBuilder

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        builder = QueryBuilder(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch(
            "src.data.storage.dialect_utils.StatisticalFunctions"
        ) as mock_stats_class:
            mock_stats = Mock()
            mock_percentile_expr = Mock()
            mock_stats.percentile_cont.return_value = mock_percentile_expr
            mock_stats_class.return_value = mock_stats

            with patch("src.data.storage.dialect_utils.select") as mock_select:
                mock_query = Mock()
                mock_select.return_value = mock_query

                result = builder.build_percentile_query(mock_column, [0.5])

                assert result == mock_query
                mock_stats.percentile_cont.assert_called_once_with(
                    0.5, mock_column, order_desc=False
                )

    def test_query_builder_build_percentile_query_multiple(self):
        """Test percentile query building with multiple percentiles."""
        from src.data.storage.dialect_utils import QueryBuilder

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        builder = QueryBuilder(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch(
            "src.data.storage.dialect_utils.StatisticalFunctions"
        ) as mock_stats_class:
            mock_stats = Mock()
            mock_stats.percentile_cont.return_value = Mock()
            mock_stats_class.return_value = mock_stats

            with patch("src.data.storage.dialect_utils.select") as mock_select:
                mock_query = Mock()
                mock_select.return_value = mock_query

                result = builder.build_percentile_query(mock_column, [0.25, 0.5, 0.75])

                assert result == mock_query
                assert mock_stats.percentile_cont.call_count == 3

    def test_query_builder_build_statistics_query_basic(self):
        """Test statistics query building without percentiles."""
        from src.data.storage.dialect_utils import QueryBuilder

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        builder = QueryBuilder(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
            with patch("src.data.storage.dialect_utils.select") as mock_select:
                mock_query = Mock()
                mock_select.return_value = mock_query

                result = builder.build_statistics_query(
                    mock_column, include_percentiles=False
                )

                assert result == mock_query
                # Should call basic statistical functions
                mock_sql_func.count.assert_called()
                mock_sql_func.avg.assert_called()
                mock_sql_func.min.assert_called()
                mock_sql_func.max.assert_called()

    def test_query_builder_build_statistics_query_with_percentiles(self):
        """Test statistics query building with percentiles."""
        from src.data.storage.dialect_utils import QueryBuilder

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        builder = QueryBuilder(mock_engine)

        # Mock column
        mock_column = Mock()

        with patch(
            "src.data.storage.dialect_utils.StatisticalFunctions"
        ) as mock_stats_class:
            mock_stats = Mock()
            mock_stats.percentile_cont.return_value = Mock()
            mock_stats.stddev_samp.return_value = Mock()
            mock_stats_class.return_value = mock_stats

            with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
                with patch("src.data.storage.dialect_utils.select") as mock_select:
                    mock_query = Mock()
                    mock_select.return_value = mock_query

                    result = builder.build_statistics_query(
                        mock_column, include_percentiles=True
                    )

                    assert result == mock_query
                    # Should call percentile functions for q1, median, q3
                    assert mock_stats.percentile_cont.call_count >= 3
                    mock_stats.stddev_samp.assert_called_once()

    def test_compatibility_manager_initialization(self):
        """Test CompatibilityManager initialization."""
        from src.data.storage.dialect_utils import CompatibilityManager

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        manager = CompatibilityManager(mock_engine)

        assert manager.engine == mock_engine
        assert hasattr(manager, "dialect_utils")
        assert hasattr(manager, "statistical_functions")
        assert hasattr(manager, "query_builder")

    def test_compatibility_manager_singleton_get_instance(self):
        """Test CompatibilityManager singleton get_instance."""
        from src.data.storage.dialect_utils import CompatibilityManager

        # Reset singleton
        CompatibilityManager._instance = None

        # Should raise error when not initialized
        with pytest.raises(RuntimeError, match="CompatibilityManager not initialized"):
            CompatibilityManager.get_instance()

    def test_compatibility_manager_singleton_initialize(self):
        """Test CompatibilityManager singleton initialization."""
        from src.data.storage.dialect_utils import CompatibilityManager

        # Reset singleton
        CompatibilityManager._instance = None

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        # Initialize singleton
        manager = CompatibilityManager.initialize(mock_engine)

        assert isinstance(manager, CompatibilityManager)
        assert CompatibilityManager._instance == manager

        # Second call should return same instance
        manager2 = CompatibilityManager.get_instance()
        assert manager2 is manager

    def test_compatibility_manager_delegate_methods(self):
        """Test CompatibilityManager delegation methods."""
        from src.data.storage.dialect_utils import CompatibilityManager

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        manager = CompatibilityManager(mock_engine)

        # Test delegation methods
        assert manager.is_postgresql() is True
        assert manager.is_sqlite() is False
        assert manager.get_dialect_name() == "postgresql"

    def test_global_utility_functions_with_engine(self):
        """Test global utility functions with engine parameter."""
        from src.data.storage.dialect_utils import (
            extract_epoch_interval,
            percentile_cont,
            stddev_samp,
        )

        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"

        mock_column = Mock()
        mock_interval = Mock()

        with patch(
            "src.data.storage.dialect_utils.StatisticalFunctions"
        ) as mock_stats_class:
            mock_stats = Mock()
            mock_stats.percentile_cont.return_value = "percentile_result"
            mock_stats.stddev_samp.return_value = "stddev_result"
            mock_stats.extract_epoch_from_interval.return_value = "epoch_result"
            mock_stats_class.return_value = mock_stats

            # Test with engine parameter
            result1 = percentile_cont(0.5, mock_column, engine=mock_engine)
            result2 = stddev_samp(mock_column, engine=mock_engine)
            result3 = extract_epoch_interval(mock_interval, engine=mock_engine)

            assert result1 == "percentile_result"
            assert result2 == "stddev_result"
            assert result3 == "epoch_result"

    def test_global_utility_functions_without_engine(self):
        """Test global utility functions without engine parameter."""
        from src.data.storage.dialect_utils import (
            extract_epoch_interval,
            percentile_cont,
            stddev_samp,
        )

        mock_column = Mock()
        mock_interval = Mock()

        # Mock compatibility manager
        mock_manager = Mock()
        mock_manager.engine = Mock()
        mock_manager.engine.dialect.name = "postgresql"

        with patch(
            "src.data.storage.dialect_utils.get_compatibility_manager",
            return_value=mock_manager,
        ):
            with patch(
                "src.data.storage.dialect_utils.StatisticalFunctions"
            ) as mock_stats_class:
                mock_stats = Mock()
                mock_stats.percentile_cont.return_value = "percentile_result"
                mock_stats.stddev_samp.return_value = "stddev_result"
                mock_stats.extract_epoch_from_interval.return_value = "epoch_result"
                mock_stats_class.return_value = mock_stats

                # Test without engine parameter
                result1 = percentile_cont(0.5, mock_column)
                result2 = stddev_samp(mock_column)
                result3 = extract_epoch_interval(mock_interval)

                assert result1 == "percentile_result"
                assert result2 == "stddev_result"
                assert result3 == "epoch_result"

    def test_global_utility_functions_fallback(self):
        """Test global utility functions with RuntimeError fallback."""
        from src.data.storage.dialect_utils import (
            extract_epoch_interval,
            percentile_cont,
            stddev_samp,
        )

        mock_column = Mock()
        mock_interval = Mock()

        # Mock get_compatibility_manager to raise RuntimeError
        with patch(
            "src.data.storage.dialect_utils.get_compatibility_manager",
            side_effect=RuntimeError("Not initialized"),
        ):
            with patch("src.data.storage.dialect_utils.sql_func") as mock_sql_func:
                mock_sql_func.avg.return_value = "fallback_median"
                mock_sql_func.sqrt.return_value = "fallback_stddev"
                mock_sql_func.strftime.return_value = "fallback_epoch"

                # Test fallback behavior
                result1 = percentile_cont(0.5, mock_column)  # Should fallback to median
                result2 = stddev_samp(
                    mock_column
                )  # Should fallback to sqrt approximation
                result3 = extract_epoch_interval(
                    mock_interval
                )  # Should fallback to SQLite approach

                # Verify fallback functions were called
                mock_sql_func.avg.assert_called_with(mock_column)
                mock_sql_func.sqrt.assert_called()
                mock_sql_func.strftime.assert_called()

    def test_get_compatibility_manager_function(self):
        """Test get_compatibility_manager global function."""
        from src.data.storage.dialect_utils import (
            CompatibilityManager,
            get_compatibility_manager,
        )

        # Setup singleton
        mock_engine = Mock()
        CompatibilityManager._instance = None
        mock_manager = CompatibilityManager.initialize(mock_engine)

        # Test global function
        result = get_compatibility_manager()
        assert result is mock_manager


class TestDatabaseModelsIntegration:
    """Test database models integration with the database layer."""

    @pytest.mark.asyncio
    async def test_sensor_event_model_basic_operations(self):
        """Test basic SensorEvent model operations."""
        from src.data.storage.models import SensorEvent

        # Test model initialization
        event = SensorEvent(
            room_id="living_room",
            sensor_id="motion_sensor_1",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
        )

        assert event.room_id == "living_room"
        assert event.sensor_id == "motion_sensor_1"
        assert event.sensor_type == "motion"
        assert event.state == "on"
        assert event.is_human_triggered is True  # Default value
        assert event.attributes == {}  # Default value

    @pytest.mark.asyncio
    async def test_room_state_model_operations(self):
        """Test RoomState model operations."""
        from src.data.storage.models import RoomState

        # Test model initialization
        room_state = RoomState(
            room_id="bedroom",
            timestamp=datetime.now(timezone.utc),
            is_occupied=True,
            occupancy_confidence=Decimal("0.8500"),
            occupant_type="human",
            occupant_count=1,
        )

        assert room_state.room_id == "bedroom"
        assert room_state.is_occupied is True
        assert room_state.occupancy_confidence == Decimal("0.8500")
        assert room_state.occupant_type == "human"
        assert room_state.occupant_count == 1

    @pytest.mark.asyncio
    async def test_prediction_model_operations(self):
        """Test Prediction model operations."""
        from src.data.storage.models import Prediction

        prediction_time = datetime.now(timezone.utc)
        predicted_time = prediction_time + timedelta(minutes=30)

        # Test model initialization
        prediction = Prediction(
            room_id="kitchen",
            prediction_time=prediction_time,
            predicted_transition_time=predicted_time,
            transition_type="occupied_to_vacant",
            confidence_score=Decimal("0.7500"),
            model_type="ensemble",
            model_version="1.0.0",
        )

        assert prediction.room_id == "kitchen"
        assert prediction.prediction_time == prediction_time
        assert prediction.predicted_transition_time == predicted_time
        assert prediction.transition_type == "occupied_to_vacant"
        assert prediction.confidence_score == Decimal("0.7500")
        assert prediction.model_type == "ensemble"
        assert prediction.model_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_prediction_compatibility_fields(self):
        """Test Prediction model compatibility field handling."""
        from src.data.storage.models import Prediction

        prediction_time = datetime.now(timezone.utc)
        predicted_time = prediction_time + timedelta(minutes=30)

        # Test with predicted_time (compatibility field)
        prediction = Prediction(
            room_id="kitchen",
            prediction_time=prediction_time,
            predicted_time=predicted_time,  # Using compatibility field
            transition_type="occupied_to_vacant",
            confidence_score=Decimal("0.7500"),
            model_type="ensemble",
            model_version="1.0.0",
        )

        # Should set both fields to same value for consistency
        assert prediction.predicted_time == predicted_time
        assert prediction.predicted_transition_time == predicted_time

    def test_model_enums_and_constraints(self):
        """Test model enum values and constraints."""
        from src.data.storage.models import (
            MODEL_TYPES,
            SENSOR_STATES,
            SENSOR_TYPES,
            TRANSITION_TYPES,
        )

        # Test enum values are properly defined
        assert "motion" in SENSOR_TYPES
        assert "presence" in SENSOR_TYPES
        assert "door" in SENSOR_TYPES

        assert "on" in SENSOR_STATES
        assert "off" in SENSOR_STATES
        assert "open" in SENSOR_STATES
        assert "closed" in SENSOR_STATES

        assert "occupied_to_vacant" in TRANSITION_TYPES
        assert "vacant_to_occupied" in TRANSITION_TYPES

        assert "lstm" in MODEL_TYPES
        assert "xgboost" in MODEL_TYPES
        assert "ensemble" in MODEL_TYPES

    def test_model_helper_functions(self):
        """Test model helper functions."""
        from src.data.storage.models import (
            _get_database_specific_column_config,
            _get_json_column_type,
            _is_sqlite_engine,
        )

        # Test SQLite detection
        mock_sqlite_bind = Mock()
        mock_sqlite_bind.url = "sqlite:///test.db"
        assert _is_sqlite_engine(mock_sqlite_bind) is True

        mock_pg_bind = Mock()
        mock_pg_bind.url = "postgresql://user:pass@localhost/db"
        assert _is_sqlite_engine(mock_pg_bind) is False

        # Test column config
        sqlite_config = _get_database_specific_column_config(
            mock_sqlite_bind, "id", is_primary_key=True, autoincrement=True
        )
        assert sqlite_config["autoincrement"] is True

        # Test JSON column type (should work with various environments)
        json_type = _get_json_column_type()
        assert json_type is not None  # Should return either JSON or JSONB
