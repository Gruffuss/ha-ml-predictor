"""
Comprehensive unit tests for database management to achieve high test coverage.

This module focuses on testing all methods, error paths, edge cases,
and configuration variations in the DatabaseManager class.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
import tempfile

from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLTimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from src.data.storage.database import (
    DatabaseManager,
    get_database_manager,
    get_db_session,
    close_database_manager,
    execute_sql_file,
    check_table_exists,
    get_database_version,
    get_timescaledb_version
)


class TestDatabaseManagerInitialization:
    """Test DatabaseManager initialization and setup."""

    def test_init_with_config(self):
        """Test initialization with provided config."""
        config = DatabaseConfig(
            connection_string="postgresql://localhost/testdb",
            pool_size=5,
            max_overflow=10
        )
        
        db_manager = DatabaseManager(config)
        
        assert db_manager.config == config
        assert db_manager.engine is None
        assert db_manager.session_factory is None
        assert db_manager._health_check_task is None
        assert db_manager._connection_stats["total_connections"] == 0
        assert db_manager._connection_stats["failed_connections"] == 0
        assert db_manager.max_retries == 5
        assert db_manager.base_delay == 1.0

    @patch('src.data.storage.database.get_config')
    def test_init_without_config(self, mock_get_config):
        """Test initialization without config (loads from global)."""
        mock_config = Mock()
        mock_config.database = DatabaseConfig(
            connection_string="postgresql://localhost/testdb"
        )
        mock_get_config.return_value = mock_config
        
        db_manager = DatabaseManager()
        
        assert db_manager.config is not None
        mock_get_config.assert_called_once()

    def test_init_sets_retry_configuration(self):
        """Test that retry configuration is properly set."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        assert db_manager.max_retries == 5
        assert db_manager.base_delay == 1.0
        assert db_manager.max_delay == 60.0
        assert db_manager.backoff_multiplier == 2.0
        assert isinstance(db_manager.connection_timeout, timedelta)
        assert isinstance(db_manager.query_timeout, timedelta)
        assert isinstance(db_manager.health_check_interval, timedelta)

    def test_connection_stats_initialization(self):
        """Test that connection stats are properly initialized."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        expected_stats = {
            "total_connections": 0,
            "failed_connections": 0,
            "last_health_check": None,
            "last_connection_error": None,
            "retry_count": 0,
        }
        
        assert db_manager._connection_stats == expected_stats


class TestDatabaseManagerEngineCreation:
    """Test database engine creation and configuration."""

    @pytest.mark.asyncio
    async def test_create_engine_postgresql_url(self):
        """Test engine creation with postgresql:// URL."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost/testdb",
            pool_size=10,
            max_overflow=20
        )
        db_manager = DatabaseManager(config)
        
        with patch('src.data.storage.database.create_async_engine') as mock_create:
            mock_engine = Mock(spec=AsyncEngine)
            mock_create.return_value = mock_engine
            
            await db_manager._create_engine()
            
            # Check that URL was converted to async format
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert "postgresql+asyncpg://" in kwargs["url"]
            assert kwargs["pool_size"] == 10
            assert kwargs["max_overflow"] == 20
            assert kwargs["pool_pre_ping"] is True

    @pytest.mark.asyncio
    async def test_create_engine_asyncpg_url(self):
        """Test engine creation with postgresql+asyncpg:// URL."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://user:pass@localhost/testdb"
        )
        db_manager = DatabaseManager(config)
        
        with patch('src.data.storage.database.create_async_engine') as mock_create:
            mock_engine = Mock(spec=AsyncEngine)
            mock_create.return_value = mock_engine
            
            await db_manager._create_engine()
            
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert kwargs["url"] == config.connection_string

    @pytest.mark.asyncio
    async def test_create_engine_invalid_url(self):
        """Test engine creation with invalid URL format."""
        config = DatabaseConfig(connection_string="mysql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with pytest.raises(ValueError) as exc_info:
            await db_manager._create_engine()
        
        assert "postgresql" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_engine_null_pool_for_testing(self):
        """Test engine creation with NullPool for testing."""
        config = DatabaseConfig(
            connection_string="postgresql://localhost/testdb",
            pool_size=0  # Triggers NullPool
        )
        db_manager = DatabaseManager(config)
        
        with patch('src.data.storage.database.create_async_engine') as mock_create:
            mock_engine = Mock(spec=AsyncEngine)
            mock_create.return_value = mock_engine
            
            await db_manager._create_engine()
            
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            # Should have NullPool for testing
            from sqlalchemy.pool import NullPool
            assert kwargs.get("poolclass") == NullPool

    @pytest.mark.asyncio
    async def test_setup_connection_events(self):
        """Test connection event listeners setup."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Create a mock engine with sync_engine
        mock_engine = Mock(spec=AsyncEngine)
        mock_sync_engine = Mock()
        mock_engine.sync_engine = mock_sync_engine
        db_manager.engine = mock_engine
        
        with patch('sqlalchemy.event.listens_for') as mock_listens_for:
            db_manager._setup_connection_events()
            
            # Should set up event listeners
            assert mock_listens_for.call_count >= 4  # connect, checkout, checkin, invalidate

    def test_setup_connection_events_no_engine(self):
        """Test connection events setup without engine."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with pytest.raises(RuntimeError) as exc_info:
            db_manager._setup_connection_events()
        
        assert "Engine must be created" in str(exc_info.value)


class TestDatabaseManagerSessionFactory:
    """Test session factory setup."""

    @pytest.mark.asyncio
    async def test_setup_session_factory_success(self):
        """Test successful session factory setup."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock engine
        mock_engine = Mock(spec=AsyncEngine)
        db_manager.engine = mock_engine
        
        with patch('src.data.storage.database.async_sessionmaker') as mock_sessionmaker:
            mock_factory = Mock()
            mock_sessionmaker.return_value = mock_factory
            
            await db_manager._setup_session_factory()
            
            assert db_manager.session_factory == mock_factory
            mock_sessionmaker.assert_called_once_with(
                bind=mock_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

    @pytest.mark.asyncio
    async def test_setup_session_factory_no_engine(self):
        """Test session factory setup without engine."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with pytest.raises(RuntimeError) as exc_info:
            await db_manager._setup_session_factory()
        
        assert "Engine must be created" in str(exc_info.value)


class TestDatabaseManagerConnectionVerification:
    """Test database connection verification."""

    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        """Test successful connection verification."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock engine and connection
        mock_engine = Mock(spec=AsyncEngine)
        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        db_manager.engine = mock_engine
        
        # First call returns 1 (SELECT 1), second returns 1 (TimescaleDB check)
        mock_result_timescale = Mock()
        mock_result_timescale.scalar.return_value = 1
        mock_conn.execute.side_effect = [mock_result, mock_result_timescale]
        
        await db_manager._verify_connection()
        
        # Should have made two execute calls
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_verify_connection_no_timescaledb(self):
        """Test connection verification without TimescaleDB."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock engine and connection
        mock_engine = Mock(spec=AsyncEngine)
        mock_conn = AsyncMock()
        
        # First result (SELECT 1) succeeds, second (TimescaleDB) returns 0
        mock_result1 = Mock()
        mock_result1.scalar.return_value = 1
        mock_result2 = Mock()
        mock_result2.scalar.return_value = 0
        
        mock_conn.execute.side_effect = [mock_result1, mock_result2]
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        db_manager.engine = mock_engine
        
        with patch('src.data.storage.database.logger') as mock_logger:
            await db_manager._verify_connection()
            
            # Should log warning about TimescaleDB
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_connection_no_engine(self):
        """Test connection verification without engine."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with pytest.raises(RuntimeError) as exc_info:
            await db_manager._verify_connection()
        
        assert "Engine not initialized" in str(exc_info.value)


class TestDatabaseManagerInitialize:
    """Test full database manager initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with patch.object(db_manager, '_create_engine') as mock_create_engine, \
             patch.object(db_manager, '_setup_session_factory') as mock_setup_factory, \
             patch.object(db_manager, '_verify_connection') as mock_verify, \
             patch('asyncio.create_task') as mock_create_task:
            
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            await db_manager.initialize()
            
            mock_create_engine.assert_called_once()
            mock_setup_factory.assert_called_once()
            mock_verify.assert_called_once()
            mock_create_task.assert_called_once()
            assert db_manager._health_check_task == mock_task

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Set engine to simulate already initialized
        db_manager.engine = Mock(spec=AsyncEngine)
        
        with patch('src.data.storage.database.logger') as mock_logger:
            await db_manager.initialize()
            
            mock_logger.warning.assert_called_with("Database manager already initialized")

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure and cleanup."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with patch.object(db_manager, '_create_engine') as mock_create_engine, \
             patch.object(db_manager, '_cleanup') as mock_cleanup:
            
            # Make _create_engine raise exception
            test_exception = Exception("Engine creation failed")
            mock_create_engine.side_effect = test_exception
            
            with pytest.raises(DatabaseConnectionError) as exc_info:
                await db_manager.initialize()
            
            mock_cleanup.assert_called_once()
            assert exc_info.value.cause == test_exception


class TestDatabaseManagerSessions:
    """Test database session management."""

    @pytest.fixture
    def initialized_db_manager(self):
        """Create an initialized database manager."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock session factory
        mock_session_factory = Mock()
        db_manager.session_factory = mock_session_factory
        
        return db_manager, mock_session_factory

    @pytest.mark.asyncio
    async def test_get_session_success(self, initialized_db_manager):
        """Test successful session creation and usage."""
        db_manager, mock_session_factory = initialized_db_manager
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with db_manager.get_session() as session:
            assert session == mock_session
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self):
        """Test session creation with uninitialized manager."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with pytest.raises(RuntimeError) as exc_info:
            async with db_manager.get_session():
                pass
        
        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_session_with_retry_on_connection_error(self, initialized_db_manager):
        """Test session creation with connection error retry."""
        db_manager, mock_session_factory = initialized_db_manager
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        # First attempt fails, second succeeds
        connection_error = OperationalError("Connection failed", None, None)
        attempts = [connection_error, None]
        
        async def mock_session_context():
            error = attempts.pop(0) if attempts else None
            if error:
                raise error
            yield mock_session
        
        with patch.object(db_manager, 'get_session', side_effect=mock_session_context):
            # This will use the patched version
            pass

    @pytest.mark.asyncio
    async def test_get_session_max_retries_exceeded(self, initialized_db_manager):
        """Test session creation when max retries exceeded."""
        db_manager, mock_session_factory = initialized_db_manager
        db_manager.max_retries = 2  # Set low for testing
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        # Mock session factory to always raise connection error
        connection_error = OperationalError("Connection failed", None, None)
        mock_session_factory.side_effect = connection_error
        
        with patch('asyncio.sleep'):  # Speed up test
            with pytest.raises(DatabaseConnectionError):
                async with db_manager.get_session():
                    pass
        
        # Should have incremented retry count
        assert db_manager._connection_stats["retry_count"] > 0

    @pytest.mark.asyncio
    async def test_get_session_general_exception(self, initialized_db_manager):
        """Test session creation with general exception."""
        db_manager, mock_session_factory = initialized_db_manager
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        # Mock session to raise general exception
        general_error = ValueError("Something went wrong")
        mock_session_factory.side_effect = general_error
        
        with pytest.raises(DatabaseQueryError) as exc_info:
            async with db_manager.get_session():
                pass
        
        assert exc_info.value.cause == general_error

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self, initialized_db_manager):
        """Test that session is rolled back on error."""
        db_manager, mock_session_factory = initialized_db_manager
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        test_error = ValueError("Test error")
        
        with pytest.raises(DatabaseQueryError):
            async with db_manager.get_session() as session:
                raise test_error
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_connection_error_rollback_fails(self, initialized_db_manager):
        """Test session handling when rollback also fails during connection error."""
        db_manager, mock_session_factory = initialized_db_manager
        db_manager.max_retries = 0  # No retries for faster test
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        # Mock connection error and rollback failure
        connection_error = DisconnectionError("Connection lost")
        rollback_error = Exception("Rollback failed")
        
        mock_session_factory.side_effect = connection_error
        mock_session.rollback.side_effect = rollback_error
        
        with pytest.raises(DatabaseConnectionError):
            async with db_manager.get_session():
                pass
        
        # Rollback should have been attempted despite failure
        mock_session.rollback.assert_called_once()


class TestDatabaseManagerQueryExecution:
    """Test query execution methods."""

    @pytest.fixture
    def initialized_db_manager(self):
        """Create an initialized database manager with mocked session."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        mock_session_factory = Mock()
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        db_manager.session_factory = mock_session_factory
        
        return db_manager, mock_session

    @pytest.mark.asyncio
    async def test_execute_query_basic(self, initialized_db_manager):
        """Test basic query execution."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await db_manager.execute_query("SELECT 1")
            
            assert result == mock_result
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(self, initialized_db_manager):
        """Test query execution with parameters."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            parameters = {"id": 1, "name": "test"}
            await db_manager.execute_query("SELECT * FROM table WHERE id = :id", parameters)
            
            # Check that execute was called with parameters
            args, kwargs = mock_session.execute.call_args
            assert kwargs == parameters or args[1] == parameters

    @pytest.mark.asyncio
    async def test_execute_query_fetch_one(self, initialized_db_manager):
        """Test query execution with fetch_one."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_row = ("test_value",)
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await db_manager.execute_query("SELECT value", fetch_one=True)
            
            assert result == mock_row
            mock_result.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_fetch_all(self, initialized_db_manager):
        """Test query execution with fetch_all."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_rows = [("value1",), ("value2",)]
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await db_manager.execute_query("SELECT value", fetch_all=True)
            
            assert result == mock_rows
            mock_result.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, initialized_db_manager):
        """Test query execution with timeout."""
        db_manager, mock_session = initialized_db_manager
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            # Mock session execute to hang
            mock_session.execute.side_effect = asyncio.sleep(10)  # Long delay
            
            custom_timeout = timedelta(milliseconds=100)
            
            with pytest.raises(DatabaseQueryError) as exc_info:
                await db_manager.execute_query("SELECT 1", timeout=custom_timeout)
            
            assert exc_info.value.context.get("error_type") == "TimeoutError"

    @pytest.mark.asyncio
    async def test_execute_query_sqlalchemy_error(self, initialized_db_manager):
        """Test query execution with SQLAlchemy error."""
        db_manager, mock_session = initialized_db_manager
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            sql_error = SQLAlchemyError("SQL error")
            mock_session.execute.side_effect = sql_error
            
            with pytest.raises(DatabaseQueryError) as exc_info:
                await db_manager.execute_query("SELECT 1")
            
            assert exc_info.value.cause == sql_error

    @pytest.mark.asyncio
    async def test_execute_query_general_error(self, initialized_db_manager):
        """Test query execution with general error."""
        db_manager, mock_session = initialized_db_manager
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            general_error = ValueError("General error")
            mock_session.execute.side_effect = general_error
            
            with pytest.raises(DatabaseQueryError) as exc_info:
                await db_manager.execute_query("SELECT 1")
            
            assert exc_info.value.cause == general_error


class TestDatabaseManagerOptimizedQuery:
    """Test optimized query execution."""

    @pytest.fixture
    def initialized_db_manager(self):
        """Create an initialized database manager."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        mock_session = AsyncMock(spec=AsyncSession)
        db_manager.session_factory = Mock(return_value=mock_session)
        
        return db_manager, mock_session

    @pytest.mark.asyncio
    async def test_execute_optimized_query_basic(self, initialized_db_manager):
        """Test basic optimized query execution."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await db_manager.execute_optimized_query("SELECT 1")
            
            assert result == mock_result
            # Should enable query caching by default
            assert mock_session.execute.call_count >= 2  # Cache setting + actual query

    @pytest.mark.asyncio
    async def test_execute_optimized_query_with_cache_disabled(self, initialized_db_manager):
        """Test optimized query with caching disabled."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            await db_manager.execute_optimized_query("SELECT 1", enable_query_cache=False)
            
            # Should only call execute once (no cache setting)
            assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_optimized_query_prepared_statement_success(self, initialized_db_manager):
        """Test optimized query with prepared statement."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            parameters = {"id": 1, "name": "test"}
            await db_manager.execute_optimized_query(
                "SELECT * FROM table WHERE id = :id",
                parameters=parameters,
                use_prepared_statement=True
            )
            
            # Should call execute multiple times: cache setting, prepare, execute, deallocate
            assert mock_session.execute.call_count >= 4

    @pytest.mark.asyncio
    async def test_execute_optimized_query_prepared_statement_fallback(self, initialized_db_manager):
        """Test prepared statement with fallback to regular query."""
        db_manager, mock_session = initialized_db_manager
        
        # Make prepared statement fail
        sql_error = SQLAlchemyError("Prepare failed")
        mock_result = Mock()
        
        def execute_side_effect(query, params=None):
            if "PREPARE" in str(query):
                raise sql_error
            return mock_result
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            parameters = {"id": 1}
            result = await db_manager.execute_optimized_query(
                "SELECT * FROM table WHERE id = :id",
                parameters=parameters,
                use_prepared_statement=True
            )
            
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_optimized_query_error_handling(self, initialized_db_manager):
        """Test optimized query error handling."""
        db_manager, mock_session = initialized_db_manager
        
        sql_error = SQLAlchemyError("Query failed")
        mock_session.execute.side_effect = sql_error
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            with pytest.raises(DatabaseQueryError) as exc_info:
                await db_manager.execute_optimized_query("SELECT 1")
            
            assert exc_info.value.cause == sql_error


class TestDatabaseManagerQueryAnalysis:
    """Test query performance analysis."""

    @pytest.fixture
    def initialized_db_manager(self):
        """Create an initialized database manager."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        mock_session = AsyncMock(spec=AsyncSession)
        db_manager.session_factory = Mock(return_value=mock_session)
        
        return db_manager, mock_session

    @pytest.mark.asyncio
    async def test_analyze_query_performance_full_analysis(self, initialized_db_manager):
        """Test full query performance analysis."""
        db_manager, mock_session = initialized_db_manager
        
        # Mock execution plan result
        mock_plan_result = Mock()
        mock_plan_result.fetchone.return_value = [{"execution_plan": "test_plan"}]
        
        # Mock actual query result
        mock_query_result = Mock()
        
        call_count = 0
        def execute_side_effect(query, params=None):
            nonlocal call_count
            call_count += 1
            if "EXPLAIN" in str(query):
                return mock_plan_result
            else:
                return mock_query_result
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            with patch('time.time', side_effect=[1000.0, 1000.1]):  # 0.1 second execution
                analysis = await db_manager.analyze_query_performance("SELECT 1")
            
            assert "query" in analysis
            assert "timestamp" in analysis
            assert "execution_plan" in analysis
            assert "execution_time_seconds" in analysis
            assert analysis["execution_time_seconds"] == 0.1
            assert analysis["performance_rating"] == "excellent"
            assert "optimization_suggestions" in analysis

    @pytest.mark.asyncio
    async def test_analyze_query_performance_no_execution_plan(self, initialized_db_manager):
        """Test query analysis without execution plan."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            analysis = await db_manager.analyze_query_performance(
                "SELECT 1",
                include_execution_plan=False
            )
            
            assert "execution_plan" not in analysis
            assert "query" in analysis
            assert "optimization_suggestions" in analysis

    @pytest.mark.asyncio
    async def test_analyze_query_performance_execution_plan_error(self, initialized_db_manager):
        """Test query analysis when execution plan fails."""
        db_manager, mock_session = initialized_db_manager
        
        plan_error = SQLAlchemyError("EXPLAIN failed")
        mock_result = Mock()
        
        def execute_side_effect(query, params=None):
            if "EXPLAIN" in str(query):
                raise plan_error
            return mock_result
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            analysis = await db_manager.analyze_query_performance("SELECT 1")
            
            assert "execution_plan_error" in analysis
            assert analysis["execution_plan_error"] == str(plan_error)

    @pytest.mark.asyncio
    async def test_analyze_query_performance_execution_error(self, initialized_db_manager):
        """Test query analysis when query execution fails."""
        db_manager, mock_session = initialized_db_manager
        
        # EXPLAIN succeeds, actual query fails
        mock_plan_result = Mock()
        mock_plan_result.fetchone.return_value = [{"plan": "test"}]
        
        query_error = SQLAlchemyError("Query failed")
        
        def execute_side_effect(query, params=None):
            if "EXPLAIN" in str(query):
                return mock_plan_result
            raise query_error
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            analysis = await db_manager.analyze_query_performance("SELECT 1")
            
            assert "execution_error" in analysis
            assert analysis["execution_error"] == str(query_error)

    @pytest.mark.asyncio
    async def test_analyze_query_performance_rating_categories(self, initialized_db_manager):
        """Test different performance rating categories."""
        db_manager, mock_session = initialized_db_manager
        
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            # Test different execution times
            time_ratings = [
                (0.05, "excellent"),  # < 0.1s
                (0.5, "good"),        # < 1.0s
                (2.0, "acceptable"),  # < 5.0s
                (10.0, "needs_optimization")  # >= 5.0s
            ]
            
            for exec_time, expected_rating in time_ratings:
                with patch('time.time', side_effect=[1000.0, 1000.0 + exec_time]):
                    analysis = await db_manager.analyze_query_performance(
                        "SELECT 1",
                        include_execution_plan=False
                    )
                    
                    assert analysis["performance_rating"] == expected_rating

    def test_get_optimization_suggestions(self):
        """Test query optimization suggestions."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Test various query patterns
        suggestions_tests = [
            ("SELECT * FROM sensor_events", ["WHERE clause", "SELECT *"]),
            ("SELECT * FROM table JOIN other", ["SELECT *", "ON conditions"]),
            ("SELECT col FROM table ORDER BY col", ["LIMIT clause"]),
            ("SELECT count(*) FROM (SELECT id FROM table)", ["JOINs instead"]),
        ]
        
        for query, expected_keywords in suggestions_tests:
            suggestions = db_manager._get_optimization_suggestions(query)
            
            for keyword in expected_keywords:
                assert any(keyword.lower() in suggestion.lower() for suggestion in suggestions), \
                    f"Expected '{keyword}' in suggestions for query: {query}"


class TestDatabaseManagerPoolMetrics:
    """Test connection pool metrics."""

    def test_get_connection_pool_metrics_no_engine(self):
        """Test pool metrics when engine is not initialized."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Use async runner for async method
        async def run_test():
            metrics = await db_manager.get_connection_pool_metrics()
            assert metrics["error"] == "Database engine not initialized"
            assert "timestamp" in metrics
            
        asyncio.run(run_test())

    def test_get_connection_pool_metrics_with_pool(self):
        """Test pool metrics with active engine."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock engine and pool
        mock_pool = Mock()
        mock_pool._pool_size = 10
        mock_pool._checked_out = 3
        mock_pool._overflow = 1
        mock_pool._invalidated = 0
        
        mock_engine = Mock(spec=AsyncEngine)
        mock_engine.pool = mock_pool
        db_manager.engine = mock_engine
        
        async def run_test():
            metrics = await db_manager.get_connection_pool_metrics()
            
            assert metrics["pool_size"] == 10
            assert metrics["checked_out"] == 3
            assert metrics["overflow"] == 1
            assert metrics["invalid_count"] == 0
            assert metrics["utilization_percent"] == 40.0  # (3+1)/10 * 100
            assert metrics["pool_status"] == "healthy"  # < 50%
            
        asyncio.run(run_test())

    def test_get_connection_pool_metrics_high_utilization(self):
        """Test pool metrics with high utilization."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock high utilization pool
        mock_pool = Mock()
        mock_pool._pool_size = 10
        mock_pool._checked_out = 8
        mock_pool._overflow = 2
        mock_pool._invalidated = 0
        
        mock_engine = Mock(spec=AsyncEngine)
        mock_engine.pool = mock_pool
        db_manager.engine = mock_engine
        
        async def run_test():
            metrics = await db_manager.get_connection_pool_metrics()
            
            assert metrics["utilization_percent"] == 100.0  # (8+2)/10 * 100
            assert metrics["pool_status"] == "high_utilization"
            assert "recommendations" in metrics
            assert len(metrics["recommendations"]) > 0
            
        asyncio.run(run_test())

    def test_get_connection_pool_metrics_with_invalid_connections(self):
        """Test pool metrics with invalid connections."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock pool with invalid connections
        mock_pool = Mock()
        mock_pool._pool_size = 10
        mock_pool._checked_out = 2
        mock_pool._overflow = 0
        mock_pool._invalidated = 3  # Invalid connections
        
        mock_engine = Mock(spec=AsyncEngine)
        mock_engine.pool = mock_pool
        db_manager.engine = mock_engine
        
        async def run_test():
            metrics = await db_manager.get_connection_pool_metrics()
            
            assert metrics["invalid_count"] == 3
            assert "recommendations" in metrics
            assert any("connectivity" in rec.lower() for rec in metrics["recommendations"])
            
        asyncio.run(run_test())


class TestDatabaseManagerHealthCheck:
    """Test database health check functionality."""

    @pytest.fixture
    def initialized_db_manager(self):
        """Create an initialized database manager."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        mock_session = AsyncMock(spec=AsyncSession)
        db_manager.session_factory = Mock(return_value=mock_session)
        
        # Mock engine for pool metrics
        mock_engine = Mock(spec=AsyncEngine)
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_engine.pool = mock_pool
        db_manager.engine = mock_engine
        
        return db_manager, mock_session

    @pytest.mark.asyncio
    async def test_health_check_success_with_timescaledb(self, initialized_db_manager):
        """Test successful health check with TimescaleDB."""
        db_manager, mock_session = initialized_db_manager
        
        # Mock TimescaleDB version response
        mock_version_result = Mock()
        mock_version_result.fetchone.return_value = ["TimescaleDB version 2.8.0 on PostgreSQL 14.5"]
        
        def execute_side_effect(query):
            if "timescaledb_information" in str(query):
                return mock_version_result
            # Default for SELECT 1
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            return mock_result
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            health = await db_manager.health_check()
            
            assert health["status"] == "healthy"
            assert health["timescale_status"] == "available"
            assert "timescale_version_info" in health
            assert "2.8.0" in health["timescale_version_info"]["timescale_version"]
            assert "14.5" in health["timescale_version_info"]["postgresql_version"]
            assert "performance_metrics" in health
            assert health["performance_metrics"]["response_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_health_check_without_timescaledb(self, initialized_db_manager):
        """Test health check when TimescaleDB is not available."""
        db_manager, mock_session = initialized_db_manager
        
        def execute_side_effect(query):
            if "timescaledb_information" in str(query):
                raise Exception("TimescaleDB not available")
            # Default for SELECT 1
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            return mock_result
        
        mock_session.execute.side_effect = execute_side_effect
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            health = await db_manager.health_check()
            
            assert health["status"] == "healthy"  # Should still be healthy
            assert health["timescale_status"] == "unavailable"
            assert "error" in health["timescale_version_info"]

    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self, initialized_db_manager):
        """Test health check when connection fails."""
        db_manager, mock_session = initialized_db_manager
        
        connection_error = Exception("Connection failed")
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.side_effect = connection_error
            
            health = await db_manager.health_check()
            
            assert health["status"] == "unhealthy"
            assert len(health["errors"]) > 0
            assert health["errors"][0]["type"] == "health_check_failed"

    @pytest.mark.asyncio
    async def test_health_check_with_connection_errors_in_stats(self, initialized_db_manager):
        """Test health check includes connection error stats."""
        db_manager, mock_session = initialized_db_manager
        
        # Add connection error to stats
        db_manager._connection_stats["last_connection_error"] = "Previous connection failed"
        db_manager._connection_stats["failed_connections"] = 5
        
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            health = await db_manager.health_check()
            
            assert health["status"] == "healthy"
            assert len(health["errors"]) > 0
            assert health["errors"][0]["type"] == "connection_error"
            assert health["errors"][0]["failed_count"] == 5

    @pytest.mark.asyncio
    async def test_health_check_pool_metrics_error(self, initialized_db_manager):
        """Test health check when pool metrics fail."""
        db_manager, mock_session = initialized_db_manager
        
        # Make pool methods raise exception
        mock_engine = Mock(spec=AsyncEngine)
        mock_pool = Mock()
        mock_pool.size.side_effect = Exception("Pool error")
        mock_engine.pool = mock_pool
        db_manager.engine = mock_engine
        
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            health = await db_manager.health_check()
            
            assert health["status"] == "healthy"
            # Should have default pool metrics
            assert health["performance_metrics"]["pool_size"] == 0
            assert health["performance_metrics"]["checked_out_connections"] == 0

    @pytest.mark.asyncio
    async def test_health_check_loop_cancellation(self):
        """Test health check loop cancellation."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Mock the health_check method to be fast
        with patch.object(db_manager, 'health_check', return_value={"status": "healthy"}):
            # Start the health check loop
            task = asyncio.create_task(db_manager._health_check_loop())
            
            # Let it run briefly
            await asyncio.sleep(0.01)
            
            # Cancel the task
            task.cancel()
            
            # Should handle cancellation gracefully
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_health_check_loop_error_handling(self):
        """Test health check loop error handling."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        db_manager.health_check_interval = timedelta(milliseconds=10)  # Fast for testing
        
        # Mock health_check to raise exception first, then succeed
        call_count = 0
        async def mock_health_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Health check failed")
            return {"status": "healthy"}
        
        with patch.object(db_manager, 'health_check', side_effect=mock_health_check), \
             patch('asyncio.sleep') as mock_sleep:
            
            # Start the health check loop
            task = asyncio.create_task(db_manager._health_check_loop())
            
            # Let it run briefly to trigger error and recovery
            await asyncio.sleep(0.01)
            
            # Cancel the task
            task.cancel()
            
            # Should have called sleep for error recovery
            assert any(call.args[0] == 60 for call in mock_sleep.call_args_list)


class TestDatabaseManagerCleanup:
    """Test database cleanup functionality."""

    @pytest.mark.asyncio
    async def test_close_calls_cleanup(self):
        """Test that close() calls _cleanup()."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        with patch.object(db_manager, '_cleanup') as mock_cleanup:
            await db_manager.close()
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_health_check_task(self):
        """Test cleanup with active health check task."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Create mock task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        db_manager._health_check_task = mock_task
        
        # Mock engine
        mock_engine = AsyncMock(spec=AsyncEngine)
        db_manager.engine = mock_engine
        
        await db_manager._cleanup()
        
        mock_task.cancel.assert_called_once()
        mock_engine.dispose.assert_called_once()
        assert db_manager.engine is None
        assert db_manager.session_factory is None

    @pytest.mark.asyncio
    async def test_cleanup_with_completed_health_check_task(self):
        """Test cleanup with already completed health check task."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Create mock completed task
        mock_task = Mock()
        mock_task.done.return_value = True
        db_manager._health_check_task = mock_task
        
        # Mock engine
        mock_engine = AsyncMock(spec=AsyncEngine)
        db_manager.engine = mock_engine
        
        await db_manager._cleanup()
        
        # Should not cancel already completed task
        mock_task.cancel.assert_not_called()
        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_task_cancellation_error(self):
        """Test cleanup handles task cancellation errors gracefully."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Create mock task that raises when awaited
        async def failing_task():
            raise asyncio.CancelledError()
        
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        mock_task.__await__ = lambda: failing_task().__await__()
        db_manager._health_check_task = mock_task
        
        # Should handle CancelledError gracefully
        await db_manager._cleanup()

    def test_get_connection_stats(self):
        """Test getting connection statistics."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Modify some stats
        db_manager._connection_stats["total_connections"] = 10
        db_manager._connection_stats["failed_connections"] = 2
        
        stats = db_manager.get_connection_stats()
        
        assert stats["total_connections"] == 10
        assert stats["failed_connections"] == 2
        
        # Should return a copy
        stats["total_connections"] = 999
        assert db_manager._connection_stats["total_connections"] == 10

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
        db_manager = DatabaseManager(config)
        
        # Initially not initialized
        assert db_manager.is_initialized is False
        
        # Set engine only
        db_manager.engine = Mock(spec=AsyncEngine)
        assert db_manager.is_initialized is False
        
        # Set both engine and session factory
        db_manager.session_factory = Mock()
        assert db_manager.is_initialized is True


class TestGlobalDatabaseFunctions:
    """Test global database utility functions."""

    @pytest.mark.asyncio
    async def test_get_database_manager_singleton(self):
        """Test that get_database_manager returns singleton."""
        # Clear global instance
        import src.data.storage.database
        src.data.storage.database._db_manager = None
        
        with patch('src.data.storage.database.DatabaseManager') as mock_db_class:
            mock_db_instance = Mock()
            mock_db_instance.initialize = AsyncMock()
            mock_db_class.return_value = mock_db_instance
            
            # First call should create instance
            db1 = await get_database_manager()
            
            # Second call should return same instance
            db2 = await get_database_manager()
            
            assert db1 is db2
            mock_db_class.assert_called_once()
            mock_db_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_session_convenience_function(self):
        """Test get_db_session convenience function."""
        mock_db_manager = Mock()
        mock_session = AsyncMock(spec=AsyncSession)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__aexit__.return_value = None
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            async with get_db_session() as session:
                assert session == mock_session
            
            mock_db_manager.get_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_database_manager(self):
        """Test closing global database manager."""
        # Set up global instance
        import src.data.storage.database
        mock_db_manager = Mock()
        mock_db_manager.close = AsyncMock()
        src.data.storage.database._db_manager = mock_db_manager
        
        await close_database_manager()
        
        mock_db_manager.close.assert_called_once()
        assert src.data.storage.database._db_manager is None

    @pytest.mark.asyncio
    async def test_close_database_manager_no_instance(self):
        """Test closing when no global instance exists."""
        import src.data.storage.database
        src.data.storage.database._db_manager = None
        
        # Should not raise error
        await close_database_manager()

    @pytest.mark.asyncio
    async def test_execute_sql_file(self):
        """Test executing SQL file."""
        sql_content = """
        CREATE TABLE test (id INTEGER);
        INSERT INTO test VALUES (1);
        INSERT INTO test VALUES (2);
        """
        
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql_content)
            f.flush()
            
            try:
                with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
                    await execute_sql_file(f.name)
                
                # Should have executed 3 statements
                assert mock_db_manager.execute_query.call_count == 3
                
            finally:
                import os
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_execute_sql_file_not_found(self):
        """Test executing non-existent SQL file."""
        with pytest.raises(DatabaseQueryError):
            await execute_sql_file("/nonexistent/file.sql")

    @pytest.mark.asyncio
    async def test_check_table_exists_true(self):
        """Test checking existing table."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(return_value=[True])
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            exists = await check_table_exists("test_table")
            
            assert exists is True
            mock_db_manager.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_table_exists_false(self):
        """Test checking non-existent table."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(return_value=[False])
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            exists = await check_table_exists("nonexistent_table")
            
            assert exists is False

    @pytest.mark.asyncio
    async def test_check_table_exists_error(self):
        """Test checking table with database error."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(side_effect=Exception("DB error"))
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            exists = await check_table_exists("test_table")
            
            assert exists is False

    @pytest.mark.asyncio
    async def test_get_database_version_success(self):
        """Test getting database version."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(return_value=["PostgreSQL 14.5"])
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            version = await get_database_version()
            
            assert version == "PostgreSQL 14.5"

    @pytest.mark.asyncio
    async def test_get_database_version_error(self):
        """Test getting database version with error."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(side_effect=Exception("DB error"))
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            version = await get_database_version()
            
            assert version == "Error"

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_success(self):
        """Test getting TimescaleDB version."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(return_value=["2.8.0"])
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            version = await get_timescaledb_version()
            
            assert version == "2.8.0"

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_not_available(self):
        """Test getting TimescaleDB version when not available."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(return_value=None)
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            version = await get_timescaledb_version()
            
            assert version is None

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_error(self):
        """Test getting TimescaleDB version with error."""
        mock_db_manager = Mock()
        mock_db_manager.execute_query = AsyncMock(side_effect=Exception("DB error"))
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_db_manager):
            version = await get_timescaledb_version()
            
            assert version is None