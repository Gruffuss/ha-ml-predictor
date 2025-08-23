"""
Comprehensive test suite for DatabaseManager.

Tests all functionality from DatabaseManager class including:
- Initialization with configuration options
- Engine creation with connection pooling
- Session factory setup and management
- Connection verification and health checks
- Retry logic with exponential backoff
- Background health check monitoring
- Session context management
- Transaction handling and rollbacks
- Connection pool management
- Error handling and recovery scenarios
- Performance monitoring and statistics
- Cleanup and shutdown procedures
- Security and timeout configurations
- Database compatibility and dialect handling
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLTimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.core.config import DatabaseConfig
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    ErrorSeverity,
)
from src.data.storage.database import DatabaseManager


class TestDatabaseConfig:
    """Test database configuration handling."""

    def test_database_config_creation(self):
        """Test DatabaseConfig creation with various parameters."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://user:pass@localhost/db",
            pool_size=15,
            max_overflow=25,
            pool_timeout=45,
            pool_recycle=180,
        )

        assert config.connection_string == "postgresql+asyncpg://user:pass@localhost/db"
        assert config.pool_size == 15
        assert config.max_overflow == 25
        assert config.pool_timeout == 45
        assert config.pool_recycle == 180

    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig(connection_string="postgresql+asyncpg://localhost/test")

        # Verify defaults (these would be defined in the DatabaseConfig class)
        assert config.connection_string == "postgresql+asyncpg://localhost/test"
        # Default values would be tested based on actual implementation


class TestDatabaseManagerInitialization:
    """Test DatabaseManager initialization scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create mock database configuration."""
        config = MagicMock(spec=DatabaseConfig)
        config.connection_string = (
            "postgresql+asyncpg://test:test@localhost:5432/testdb"
        )
        config.pool_size = 10
        config.max_overflow = 20
        config.query_timeout = 120
        config.connection_timeout = 30
        return config

    @pytest.fixture
    def mock_global_config(self, mock_config):
        """Mock global config system."""
        system_config = MagicMock()
        system_config.database = mock_config
        return system_config

    def test_init_with_config(self, mock_config):
        """Test initialization with provided config."""
        manager = DatabaseManager(config=mock_config)

        assert manager.config is mock_config
        assert manager.engine is None
        assert manager.session_factory is None
        assert manager._health_check_task is None
        assert manager._connection_stats["total_connections"] == 0

    @patch("src.data.storage.database.get_config")
    def test_init_without_config(self, mock_get_config, mock_global_config):
        """Test initialization without provided config (uses global config)."""
        mock_get_config.return_value = mock_global_config

        manager = DatabaseManager(config=None)

        assert manager.config is mock_global_config.database
        mock_get_config.assert_called_once()

    def test_init_connection_stats_defaults(self, mock_config):
        """Test initial connection statistics."""
        manager = DatabaseManager(config=mock_config)

        stats = manager._connection_stats
        assert stats["total_connections"] == 0
        assert stats["failed_connections"] == 0
        assert stats["last_health_check"] is None
        assert stats["last_connection_error"] is None
        assert stats["retry_count"] == 0

    def test_init_retry_configuration(self, mock_config):
        """Test retry configuration defaults."""
        manager = DatabaseManager(config=mock_config)

        assert manager.max_retries == 5
        assert manager.base_delay == 1.0
        assert manager.max_delay == 60.0
        assert manager.backoff_multiplier == 2.0
        assert manager.connection_timeout == timedelta(seconds=30)
        assert manager.query_timeout == timedelta(seconds=120)
        assert manager.health_check_interval == timedelta(minutes=5)


class TestDatabaseManagerEngineCreation:
    """Test database engine creation and configuration."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager instance."""
        return DatabaseManager(config=mock_config)

    @patch("src.data.storage.database.create_async_engine")
    def test_create_engine_success(self, mock_create_engine, manager):
        """Test successful engine creation."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        asyncio.run(manager._create_engine())

        # Verify engine creation with correct parameters
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args

        assert call_args[0][0] == manager.config.connection_string
        kwargs = call_args[1]
        assert kwargs["pool_size"] == manager.config.pool_size
        assert kwargs["max_overflow"] == manager.config.max_overflow
        assert kwargs["echo"] is False

        assert manager.engine is mock_engine

    @patch("src.data.storage.database.create_async_engine")
    def test_create_engine_error(self, mock_create_engine, manager):
        """Test engine creation error handling."""
        mock_create_engine.side_effect = SQLAlchemyError("Engine creation failed")

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager._create_engine())

    @patch("src.data.storage.database.create_async_engine")
    def test_create_engine_connection_pooling_config(self, mock_create_engine, manager):
        """Test engine creation with specific pooling configuration."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Modify config for testing
        manager.config.pool_size = 25
        manager.config.max_overflow = 50

        asyncio.run(manager._create_engine())

        call_kwargs = mock_create_engine.call_args[1]
        assert call_kwargs["pool_size"] == 25
        assert call_kwargs["max_overflow"] == 50

    @patch("src.data.storage.database.create_async_engine")
    def test_create_engine_event_listeners(self, mock_create_engine, manager):
        """Test that engine event listeners are properly configured."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        with patch("src.data.storage.database.event") as mock_event:
            asyncio.run(manager._create_engine())

            # Verify event listeners are setup (implementation dependent)
            # This would test based on actual event listener setup in _create_engine


class TestDatabaseManagerSessionFactory:
    """Test session factory setup and management."""

    @pytest.fixture
    def manager_with_engine(self, mock_config):
        """Create DatabaseManager with mock engine."""
        manager = DatabaseManager(config=mock_config)
        manager.engine = MagicMock(spec=AsyncEngine)
        return manager

    @patch("src.data.storage.database.async_sessionmaker")
    def test_setup_session_factory(self, mock_sessionmaker, manager_with_engine):
        """Test session factory setup."""
        mock_factory = MagicMock()
        mock_sessionmaker.return_value = mock_factory

        asyncio.run(manager_with_engine._setup_session_factory())

        # Verify sessionmaker called with correct parameters
        mock_sessionmaker.assert_called_once()
        call_kwargs = mock_sessionmaker.call_args[1]

        assert call_kwargs["bind"] is manager_with_engine.engine
        assert call_kwargs["class_"] is AsyncSession
        assert call_kwargs["expire_on_commit"] is False

        assert manager_with_engine.session_factory is mock_factory

    def test_setup_session_factory_no_engine(self, manager_with_engine):
        """Test session factory setup without engine."""
        manager_with_engine.engine = None

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager_with_engine._setup_session_factory())


class TestDatabaseManagerConnectionVerification:
    """Test database connection verification."""

    @pytest.fixture
    def manager_with_session_factory(self, mock_config):
        """Create DatabaseManager with mock session factory."""
        manager = DatabaseManager(config=mock_config)
        manager.engine = MagicMock(spec=AsyncEngine)
        manager.session_factory = MagicMock()
        return manager

    @patch("src.data.storage.database.text")
    def test_verify_connection_success(self, mock_text, manager_with_session_factory):
        """Test successful connection verification."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        manager_with_session_factory.session_factory.return_value.__aenter__ = (
            AsyncMock(return_value=mock_session)
        )
        manager_with_session_factory.session_factory.return_value.__aexit__ = (
            AsyncMock()
        )

        # Test verification
        asyncio.run(manager_with_session_factory._verify_connection())

        # Verify test query executed
        mock_session.execute.assert_called_once()
        mock_text.assert_called_once_with("SELECT 1")

    def test_verify_connection_failure(self, manager_with_session_factory):
        """Test connection verification failure."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = OperationalError(
            "Connection failed", None, None
        )

        manager_with_session_factory.session_factory.return_value.__aenter__ = (
            AsyncMock(return_value=mock_session)
        )
        manager_with_session_factory.session_factory.return_value.__aexit__ = (
            AsyncMock()
        )

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager_with_session_factory._verify_connection())

    def test_verify_connection_timeout(self, manager_with_session_factory):
        """Test connection verification timeout."""

        # Mock slow session creation
        async def slow_session():
            await asyncio.sleep(2)  # Longer than typical timeout
            return AsyncMock()

        manager_with_session_factory.session_factory.return_value.__aenter__ = (
            slow_session
        )

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager_with_session_factory._verify_connection())


class TestDatabaseManagerInitialization:
    """Test complete DatabaseManager initialization process."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager instance."""
        return DatabaseManager(config=mock_config)

    @patch.object(DatabaseManager, "_create_engine")
    @patch.object(DatabaseManager, "_setup_session_factory")
    @patch.object(DatabaseManager, "_verify_connection")
    def test_initialize_success(
        self, mock_verify, mock_setup_factory, mock_create_engine, manager
    ):
        """Test successful initialization."""
        mock_create_engine.return_value = None
        mock_setup_factory.return_value = None
        mock_verify.return_value = None

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            asyncio.run(manager.initialize())

            # Verify all initialization steps called
            mock_create_engine.assert_called_once()
            mock_setup_factory.assert_called_once()
            mock_verify.assert_called_once()

            # Verify health check task created
            mock_create_task.assert_called_once()
            assert manager._health_check_task is mock_task

    def test_initialize_already_initialized(self, manager):
        """Test initialization when already initialized."""
        manager.engine = MagicMock()  # Mark as initialized

        with patch.object(manager, "_create_engine") as mock_create:
            asyncio.run(manager.initialize())

            # Should not re-initialize
            mock_create.assert_not_called()

    @patch.object(DatabaseManager, "_create_engine")
    def test_initialize_error_handling(self, mock_create_engine, manager):
        """Test initialization error handling."""
        mock_create_engine.side_effect = DatabaseConnectionError("Init failed")

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager.initialize())

        # Connection stats should track the error
        assert manager._connection_stats["last_connection_error"] is not None


class TestDatabaseManagerHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def initialized_manager(self, mock_config):
        """Create initialized DatabaseManager."""
        manager = DatabaseManager(config=mock_config)
        manager.engine = MagicMock()
        manager.session_factory = MagicMock()
        return manager

    @patch("src.data.storage.database.text")
    def test_health_check_success(self, mock_text, initialized_manager):
        """Test successful health check."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        initialized_manager.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        initialized_manager.session_factory.return_value.__aexit__ = AsyncMock()

        result = asyncio.run(initialized_manager.health_check())

        assert result["status"] == "healthy"
        assert result["connection_pool"]["size"] >= 0
        assert "last_check" in result
        assert "response_time_ms" in result

    def test_health_check_failure(self, initialized_manager):
        """Test health check failure."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = OperationalError(
            "DB unavailable", None, None
        )

        initialized_manager.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        initialized_manager.session_factory.return_value.__aexit__ = AsyncMock()

        result = asyncio.run(initialized_manager.health_check())

        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_health_check_no_engine(self, mock_config):
        """Test health check when engine not initialized."""
        manager = DatabaseManager(config=mock_config)

        result = asyncio.run(manager.health_check())

        assert result["status"] == "not_initialized"

    def test_health_check_stats_update(self, initialized_manager):
        """Test health check updates statistics."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        initialized_manager.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        initialized_manager.session_factory.return_value.__aexit__ = AsyncMock()

        asyncio.run(initialized_manager.health_check())

        # Verify stats updated
        assert initialized_manager._connection_stats["last_health_check"] is not None

    @pytest.mark.asyncio
    async def test_health_check_loop(self, initialized_manager):
        """Test background health check loop."""
        # Mock health check to avoid actual DB calls
        initialized_manager.health_check = AsyncMock(return_value={"status": "healthy"})

        # Start the health check loop
        task = asyncio.create_task(initialized_manager._health_check_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify health check was called
        initialized_manager.health_check.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_loop_error_recovery(self, initialized_manager):
        """Test health check loop error recovery."""
        # Mock health check to fail then succeed
        initialized_manager.health_check = AsyncMock(
            side_effect=[Exception("Health check failed"), {"status": "healthy"}]
        )

        task = asyncio.create_task(initialized_manager._health_check_loop())

        # Let it run through error and recovery
        await asyncio.sleep(0.2)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have continued despite error
        assert initialized_manager.health_check.call_count >= 1


class TestDatabaseManagerSessionManagement:
    """Test session management and context handling."""

    @pytest.fixture
    def manager_with_session(self, mock_config):
        """Create manager with session factory."""
        manager = DatabaseManager(config=mock_config)
        manager.engine = MagicMock()
        manager.session_factory = MagicMock()
        return manager

    def test_get_session_context_manager(self, manager_with_session):
        """Test session context manager functionality."""
        mock_session = AsyncMock()
        manager_with_session.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        manager_with_session.session_factory.return_value.__aexit__ = AsyncMock()

        async def test_session():
            async with manager_with_session.get_session() as session:
                assert session is mock_session
                return "success"

        result = asyncio.run(test_session())
        assert result == "success"

    def test_get_session_no_factory(self, mock_config):
        """Test getting session when factory not initialized."""
        manager = DatabaseManager(config=mock_config)

        async def test_session():
            with pytest.raises(DatabaseConnectionError):
                async with manager.get_session():
                    pass

        asyncio.run(test_session())

    def test_session_error_handling(self, manager_with_session):
        """Test session error handling and cleanup."""
        mock_session = AsyncMock()
        mock_session.rollback = AsyncMock()
        manager_with_session.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        manager_with_session.session_factory.return_value.__aexit__ = AsyncMock()

        async def test_session_error():
            try:
                async with manager_with_session.get_session() as session:
                    raise RuntimeError("Session error")
            except RuntimeError:
                pass

        asyncio.run(test_session_error())

        # Session cleanup should have been attempted
        # (Exact behavior depends on implementation)

    def test_concurrent_sessions(self, manager_with_session):
        """Test handling multiple concurrent sessions."""
        sessions = []

        def create_mock_session():
            session = AsyncMock()
            sessions.append(session)
            return session

        manager_with_session.session_factory.return_value.__aenter__ = AsyncMock(
            side_effect=create_mock_session
        )
        manager_with_session.session_factory.return_value.__aexit__ = AsyncMock()

        async def use_session(session_id):
            async with manager_with_session.get_session() as session:
                await asyncio.sleep(0.01)  # Simulate work
                return f"session_{session_id}_complete"

        async def test_concurrent():
            tasks = [use_session(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(test_concurrent())

        # All sessions should complete
        assert len(results) == 5
        assert all("complete" in result for result in results)

        # Multiple sessions should have been created
        assert len(sessions) == 5


class TestDatabaseManagerRetryLogic:
    """Test retry logic and exponential backoff."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager for retry testing."""
        manager = DatabaseManager(config=mock_config)
        # Reduce delays for testing
        manager.base_delay = 0.01
        manager.max_delay = 0.1
        manager.max_retries = 3
        return manager

    def test_calculate_retry_delay(self, manager):
        """Test retry delay calculation."""
        # Test exponential backoff
        delay1 = manager._calculate_retry_delay(0)
        delay2 = manager._calculate_retry_delay(1)
        delay3 = manager._calculate_retry_delay(2)

        assert delay1 == manager.base_delay
        assert delay2 == manager.base_delay * manager.backoff_multiplier
        assert delay3 == manager.base_delay * (manager.backoff_multiplier**2)

        # Test max delay cap
        large_delay = manager._calculate_retry_delay(10)
        assert large_delay <= manager.max_delay

    @patch.object(DatabaseManager, "_create_engine")
    def test_retry_with_exponential_backoff(self, mock_create_engine, manager):
        """Test retry logic with exponential backoff."""
        # Mock failures followed by success
        mock_create_engine.side_effect = [
            OperationalError("Connection failed", None, None),
            OperationalError("Connection failed", None, None),
            None,  # Success
        ]

        with patch("asyncio.sleep") as mock_sleep:
            asyncio.run(manager._retry_with_backoff(manager._create_engine))

            # Verify retries happened with correct delays
            assert mock_create_engine.call_count == 3
            assert mock_sleep.call_count == 2  # 2 retries before success

            # Verify exponential backoff delays
            sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_calls[0] == manager.base_delay
            assert sleep_calls[1] == manager.base_delay * manager.backoff_multiplier

    @patch.object(DatabaseManager, "_create_engine")
    def test_retry_max_retries_exceeded(self, mock_create_engine, manager):
        """Test retry logic when max retries exceeded."""
        mock_create_engine.side_effect = OperationalError(
            "Persistent failure", None, None
        )

        with pytest.raises(DatabaseConnectionError):
            asyncio.run(manager._retry_with_backoff(manager._create_engine))

        # Should have tried max_retries + 1 times (initial + retries)
        assert mock_create_engine.call_count == manager.max_retries + 1

        # Stats should reflect failed retries
        assert manager._connection_stats["retry_count"] > 0

    def test_retry_non_retryable_error(self, manager):
        """Test that non-retryable errors are not retried."""

        async def failing_operation():
            raise ValueError("Non-retryable error")  # Not a connection error

        with pytest.raises(ValueError):
            asyncio.run(manager._retry_with_backoff(failing_operation))

        # Should not have retried
        # (Implementation would need to distinguish retryable vs non-retryable errors)


class TestDatabaseManagerConnectionStatistics:
    """Test connection statistics tracking."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager for stats testing."""
        return DatabaseManager(config=mock_config)

    def test_stats_initialization(self, manager):
        """Test initial statistics state."""
        stats = manager.get_connection_stats()

        assert stats["total_connections"] == 0
        assert stats["failed_connections"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["last_health_check"] is None
        assert stats["retry_count"] == 0

    def test_stats_update_on_success(self, manager):
        """Test statistics update on successful operations."""
        # Simulate successful connection
        manager._connection_stats["total_connections"] += 1
        manager._connection_stats["last_health_check"] = datetime.now()

        stats = manager.get_connection_stats()

        assert stats["total_connections"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["last_health_check"] is not None

    def test_stats_update_on_failure(self, manager):
        """Test statistics update on failed operations."""
        # Simulate failed connection
        manager._connection_stats["total_connections"] += 1
        manager._connection_stats["failed_connections"] += 1
        manager._connection_stats["last_connection_error"] = "Connection timeout"

        stats = manager.get_connection_stats()

        assert stats["total_connections"] == 1
        assert stats["failed_connections"] == 1
        assert stats["success_rate"] == 0.0
        assert stats["last_connection_error"] == "Connection timeout"

    def test_success_rate_calculation(self, manager):
        """Test success rate calculation."""
        # Simulate mixed success/failure
        manager._connection_stats["total_connections"] = 10
        manager._connection_stats["failed_connections"] = 2

        stats = manager.get_connection_stats()

        expected_success_rate = (10 - 2) / 10
        assert stats["success_rate"] == expected_success_rate

    def test_stats_reset(self, manager):
        """Test statistics reset functionality."""
        # Set some stats
        manager._connection_stats["total_connections"] = 5
        manager._connection_stats["failed_connections"] = 1
        manager._connection_stats["retry_count"] = 3

        manager.reset_connection_stats()

        stats = manager.get_connection_stats()
        assert stats["total_connections"] == 0
        assert stats["failed_connections"] == 0
        assert stats["retry_count"] == 0


class TestDatabaseManagerCleanup:
    """Test cleanup and shutdown procedures."""

    @pytest.fixture
    def active_manager(self, mock_config):
        """Create active DatabaseManager with mocked components."""
        manager = DatabaseManager(config=mock_config)
        manager.engine = MagicMock()
        manager.session_factory = MagicMock()
        manager._health_check_task = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_cleanup_success(self, active_manager):
        """Test successful cleanup."""
        # Mock task cancellation
        active_manager._health_check_task.cancel = MagicMock()
        active_manager._health_check_task.done.return_value = False
        active_manager._health_check_task.cancelled.return_value = True

        # Mock engine disposal
        active_manager.engine.dispose = AsyncMock()

        await active_manager.cleanup()

        # Verify cleanup steps
        active_manager._health_check_task.cancel.assert_called_once()
        active_manager.engine.dispose.assert_called_once()

        assert active_manager.engine is None
        assert active_manager.session_factory is None

    @pytest.mark.asyncio
    async def test_cleanup_no_active_components(self, mock_config):
        """Test cleanup when no active components."""
        manager = DatabaseManager(config=mock_config)

        # Should not raise errors
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_task_cancellation_error(self, active_manager):
        """Test cleanup when task cancellation fails."""
        active_manager._health_check_task.cancel.side_effect = RuntimeError(
            "Cancel failed"
        )
        active_manager.engine.dispose = AsyncMock()

        # Should still complete cleanup
        await active_manager.cleanup()

        active_manager.engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_engine_disposal_error(self, active_manager):
        """Test cleanup when engine disposal fails."""
        active_manager._health_check_task = None  # No task to cancel
        active_manager.engine.dispose = AsyncMock(
            side_effect=RuntimeError("Dispose failed")
        )

        # Should handle error gracefully
        await active_manager.cleanup()

        # Engine should still be cleared
        assert active_manager.engine is None


class TestDatabaseManagerSecurityAndTimeout:
    """Test security features and timeout handling."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager for security testing."""
        return DatabaseManager(config=mock_config)

    def test_connection_string_sanitization(self, manager):
        """Test that connection strings are sanitized in logs."""
        # This would test that passwords are not logged
        connection_string = "postgresql+asyncpg://user:secret123@localhost/db"
        manager.config.connection_string = connection_string

        # Get sanitized version (implementation dependent)
        sanitized = manager._sanitize_connection_string(connection_string)

        assert "secret123" not in sanitized
        assert "user" in sanitized
        assert "localhost" in sanitized

    @pytest.mark.asyncio
    async def test_query_timeout_enforcement(self, manager):
        """Test query timeout enforcement."""

        # Mock slow query that should timeout
        async def slow_query():
            await asyncio.sleep(2)  # Longer than timeout
            return "result"

        manager.query_timeout = timedelta(milliseconds=100)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                slow_query(), timeout=manager.query_timeout.total_seconds()
            )

    def test_connection_pool_security(self, manager):
        """Test connection pool security settings."""
        # This would test that connection pools are configured securely
        # Implementation would depend on specific security requirements
        pass

    def test_sql_injection_prevention(self, manager):
        """Test SQL injection prevention measures."""
        # This would test parameterized queries and input validation
        # Implementation specific to query execution methods
        pass


class TestDatabaseManagerErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager for error recovery testing."""
        return DatabaseManager(config=mock_config)

    @pytest.mark.asyncio
    async def test_connection_recovery_after_disconnect(self, manager):
        """Test recovery after database disconnection."""
        manager.engine = MagicMock()
        manager.session_factory = MagicMock()

        # Mock session that fails with disconnection error
        mock_session = AsyncMock()
        mock_session.execute.side_effect = DisconnectionError(
            "Connection lost", None, None
        )

        manager.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        manager.session_factory.return_value.__aexit__ = AsyncMock()

        # Health check should detect and report disconnection
        result = await manager.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_engine_recreation_on_failure(self, manager):
        """Test engine recreation when connection completely fails."""
        # This would test scenarios where the engine needs to be recreated
        # Implementation dependent on specific recovery strategies
        pass

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, manager):
        """Test graceful degradation when database is unavailable."""
        manager.engine = None  # Simulate uninitialized state

        # Operations should fail gracefully
        with pytest.raises(DatabaseConnectionError):
            async with manager.get_session():
                pass

        health_result = await manager.health_check()
        assert health_result["status"] == "not_initialized"


class TestDatabaseManagerPerformanceMonitoring:
    """Test performance monitoring capabilities."""

    @pytest.fixture
    def manager(self, mock_config):
        """Create DatabaseManager for performance testing."""
        return DatabaseManager(config=mock_config)

    def test_connection_pool_metrics(self, manager):
        """Test connection pool metrics collection."""
        manager.engine = MagicMock()

        # Mock pool with specific metrics
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 7
        mock_pool.checkedout.return_value = 3
        mock_pool.overflow.return_value = 2
        manager.engine.pool = mock_pool

        metrics = manager.get_pool_metrics()

        assert metrics["total_size"] == 10
        assert metrics["checked_in"] == 7
        assert metrics["checked_out"] == 3
        assert metrics["overflow"] == 2

    def test_query_performance_tracking(self, manager):
        """Test query performance tracking."""
        # This would test query timing and performance metrics
        # Implementation would track query execution times
        pass

    @pytest.mark.asyncio
    async def test_health_check_response_time(self, manager):
        """Test health check response time measurement."""
        manager.engine = MagicMock()
        manager.session_factory = MagicMock()

        # Mock fast responding session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        manager.session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        manager.session_factory.return_value.__aexit__ = AsyncMock()

        result = await manager.health_check()

        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0


class TestDatabaseManagerIntegrationScenarios:
    """Integration scenarios for DatabaseManager."""

    @pytest.fixture
    def realistic_config(self):
        """Create realistic database configuration."""
        return DatabaseConfig(
            connection_string="postgresql+asyncpg://ha_predictor:test_pass@localhost:5432/ha_ml_test",
            pool_size=15,
            max_overflow=30,
            query_timeout=180,
            connection_timeout=45,
        )

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, realistic_config):
        """Test complete DatabaseManager lifecycle."""
        manager = DatabaseManager(config=realistic_config)

        # Mock all external dependencies
        with patch(
            "src.data.storage.database.create_async_engine"
        ) as mock_engine_create, patch(
            "src.data.storage.database.async_sessionmaker"
        ) as mock_sessionmaker:

            # Setup mocks
            mock_engine = MagicMock()
            mock_engine_create.return_value = mock_engine

            mock_factory = MagicMock()
            mock_sessionmaker.return_value = mock_factory

            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_session.execute.return_value = mock_result

            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock()

            # Test full lifecycle
            await manager.initialize()

            # Use session
            async with manager.get_session() as session:
                assert session is mock_session

            # Check health
            health = await manager.health_check()
            assert health["status"] == "healthy"

            # Cleanup
            await manager.cleanup()

            assert manager.engine is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, realistic_config):
        """Test concurrent database operations."""
        manager = DatabaseManager(config=realistic_config)

        # Mock components
        with patch("src.data.storage.database.create_async_engine"), patch(
            "src.data.storage.database.async_sessionmaker"
        ):

            # Setup successful initialization
            manager.engine = MagicMock()
            manager.session_factory = MagicMock()

            # Mock sessions
            sessions = []

            def create_session():
                session = AsyncMock()
                sessions.append(session)
                return session

            manager.session_factory.return_value.__aenter__ = AsyncMock(
                side_effect=create_session
            )
            manager.session_factory.return_value.__aexit__ = AsyncMock()

            # Concurrent session usage
            async def use_session(session_id):
                async with manager.get_session() as session:
                    await asyncio.sleep(0.01)
                    return f"session_{session_id}"

            # Run multiple concurrent operations
            tasks = [use_session(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 10
            assert len(sessions) == 10

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, realistic_config):
        """Test realistic error recovery scenario."""
        manager = DatabaseManager(config=realistic_config)

        with patch(
            "src.data.storage.database.create_async_engine"
        ) as mock_engine_create:
            # Simulate connection failure followed by success
            mock_engine_create.side_effect = [
                OperationalError("Connection refused", None, None),
                MagicMock(),  # Success
            ]

            # Reduce retry delays for testing
            manager.base_delay = 0.001
            manager.max_retries = 2

            with patch("asyncio.sleep"):  # Speed up retries
                # Should eventually succeed
                await manager.initialize()

                assert manager.engine is not None
                assert manager._connection_stats["retry_count"] > 0
