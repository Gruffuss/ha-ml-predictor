"""
Database connection management for the occupancy prediction system.

This module provides async database connection management with connection pooling,
health checks, retry logic, and proper cleanup for PostgreSQL with TimescaleDB.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import event, text
from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLTimeoutError,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from src.core.config import DatabaseConfig, get_config
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    ErrorSeverity,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages async database connections with connection pooling, health checks,
    and automatic retry logic with exponential backoff.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.

        Args:
            config: Database configuration. If None, loads from global config.
        """
        if config is None:
            system_config = get_config()
            config = system_config.database

        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_stats = {
            "total_connections": 0,
            "failed_connections": 0,
            "last_health_check": None,
            "last_connection_error": None,
            "retry_count": 0,
        }

        # Retry configuration
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.backoff_multiplier = 2.0

    async def initialize(self) -> None:
        """Initialize database engine and connection pool."""
        if self.engine is not None:
            logger.warning("Database manager already initialized")
            return

        try:
            await self._create_engine()
            await self._setup_session_factory()
            await self._verify_connection()

            # Start background health check
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

            logger.info(
                "Database manager initialized successfully",
                extra={
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                },
            )

        except Exception as e:
            await self._cleanup()
            raise DatabaseConnectionError(
                connection_string=self.config.connection_string, cause=e
            )

    async def _create_engine(self) -> None:
        """Create async SQLAlchemy engine with optimized settings."""
        # Parse connection string to add async driver
        conn_string = self.config.connection_string
        if conn_string.startswith("postgresql://"):
            conn_string = conn_string.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        elif not conn_string.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "Connection string must use postgresql:// or postgresql+asyncpg://"
            )

        # Engine configuration optimized for TimescaleDB and SQLAlchemy 2.0
        engine_kwargs = {
            "url": conn_string,
            "echo": False,  # Set to True for SQL debugging
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": 30,
            "pool_recycle": 3600,  # Recycle connections every hour
            "pool_pre_ping": True,  # Validate connections before use
        }

        # Use QueuePool for production, NullPool for testing
        if self.config.pool_size > 0:
            engine_kwargs["poolclass"] = QueuePool
        else:
            engine_kwargs["poolclass"] = NullPool

        self.engine = create_async_engine(**engine_kwargs)

        # Add connection event listeners
        self._setup_connection_events()

    def _setup_connection_events(self) -> None:
        """Setup database connection event listeners for monitoring with SQLAlchemy 2.0."""
        if self.engine is None:
            raise RuntimeError(
                "Engine must be created before setting up events"
            )

        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self._connection_stats["total_connections"] += 1
            logger.debug(
                "New database connection established",
                extra={
                    "total_connections": self._connection_stats[
                        "total_connections"
                    ]
                },
            )

            # Set connection parameters for TimescaleDB optimization
            try:
                with dbapi_connection.cursor() as cursor:
                    # Set timezone to UTC
                    cursor.execute("SET timezone = 'UTC'")
                    # Optimize for analytical workloads
                    cursor.execute("SET default_statistics_target = 100")
                    cursor.execute("SET random_page_cost = 1.1")
                    cursor.execute("SET effective_cache_size = '256MB'")
            except Exception as e:
                logger.warning(f"Failed to set connection parameters: {e}")

        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug("Database connection checked out from pool")

        @event.listens_for(self.engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            logger.debug("Database connection returned to pool")

        @event.listens_for(self.engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            self._connection_stats["failed_connections"] += 1
            self._connection_stats["last_connection_error"] = (
                str(exception) if exception else "Unknown error"
            )
            logger.warning(
                "Database connection invalidated",
                extra={
                    "exception": (
                        str(exception) if exception else "Unknown error"
                    ),
                    "failed_connections": self._connection_stats[
                        "failed_connections"
                    ],
                },
            )

    async def _setup_session_factory(self) -> None:
        """Setup async session factory."""
        if self.engine is None:
            raise RuntimeError("Engine must be created before session factory")

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )

    async def _verify_connection(self) -> None:
        """Verify database connection and TimescaleDB extension."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        async with self.engine.begin() as conn:
            # Test basic connectivity
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

            # Verify TimescaleDB extension
            result = await conn.execute(
                text(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
                )
            )
            if result.scalar() == 0:
                logger.warning(
                    "TimescaleDB extension not found - some features may not work"
                )
            else:
                logger.info("TimescaleDB extension verified")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.

        Returns:
            AsyncSession: Database session

        Raises:
            DatabaseConnectionError: If connection cannot be established
        """
        if self.session_factory is None:
            raise RuntimeError("Database manager not initialized")

        session = None
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                session = self.session_factory()
                yield session
                await session.commit()
                return

            except (
                OperationalError,
                DisconnectionError,
                SQLTimeoutError,
            ) as e:
                if session:
                    await session.rollback()
                    await session.close()

                retry_count += 1
                if retry_count > self.max_retries:
                    self._connection_stats["retry_count"] += retry_count
                    raise DatabaseConnectionError(
                        connection_string=self.config.connection_string,
                        cause=e,
                    )

                # Exponential backoff
                delay = min(
                    self.base_delay
                    * (self.backoff_multiplier ** (retry_count - 1)),
                    self.max_delay,
                )

                logger.warning(
                    f"Database connection failed, retrying in {delay:.1f}s",
                    extra={
                        "retry_count": retry_count,
                        "max_retries": self.max_retries,
                        "error": str(e),
                    },
                )

                await asyncio.sleep(delay)

            except Exception as e:
                if session:
                    await session.rollback()
                    await session.close()

                raise DatabaseQueryError(query="session_management", cause=e)

            finally:
                if session:
                    await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Any:
        """
        Execute a raw SQL query with error handling.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            fetch_one: Return single result
            fetch_all: Return all results

        Returns:
            Query result or None

        Raises:
            DatabaseQueryError: If query execution fails
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})

                if fetch_one:
                    return result.fetchone()
                elif fetch_all:
                    return result.fetchall()
                else:
                    return result

        except Exception as e:
            raise DatabaseQueryError(
                query=query, parameters=parameters, cause=e
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check.

        Returns:
            Health check results including connection stats and performance metrics
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_stats": self._connection_stats.copy(),
            "timescale_status": None,
            "performance_metrics": {},
            "errors": [],
        }

        try:
            start_time = time.time()

            # Basic connectivity test
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))

                # Check TimescaleDB status
                try:
                    result = await session.execute(
                        text(
                            "SELECT timescaledb_information.get_version_info()"
                        )
                    )
                    health_status["timescale_status"] = "available"
                except Exception:
                    health_status["timescale_status"] = "unavailable"

                # Performance metrics
                response_time = time.time() - start_time
                health_status["performance_metrics"] = {
                    "response_time_ms": round(response_time * 1000, 2),
                }

                # Add pool metrics if available (SQLAlchemy 2.0 compatible)
                if self.engine and hasattr(self.engine, "pool"):
                    try:
                        pool = self.engine.pool
                        health_status["performance_metrics"].update(
                            {
                                "pool_size": (
                                    pool.size() if hasattr(pool, "size") else 0
                                ),
                                "checked_out_connections": (
                                    pool.checkedout()
                                    if hasattr(pool, "checkedout")
                                    else 0
                                ),
                                "overflow_connections": (
                                    pool.overflow()
                                    if hasattr(pool, "overflow")
                                    else 0
                                ),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Could not retrieve pool metrics: {e}")
                        health_status["performance_metrics"].update(
                            {
                                "pool_size": 0,
                                "checked_out_connections": 0,
                                "overflow_connections": 0,
                            }
                        )

                # Check for recent errors
                if self._connection_stats["last_connection_error"]:
                    health_status["errors"].append(
                        {
                            "type": "connection_error",
                            "message": self._connection_stats[
                                "last_connection_error"
                            ],
                            "failed_count": self._connection_stats[
                                "failed_connections"
                            ],
                        }
                    )

                self._connection_stats["last_health_check"] = datetime.utcnow()

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(
                {"type": "health_check_failed", "message": str(e)}
            )
            logger.error(f"Database health check failed: {e}")

        return health_status

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup method."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close database engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None

        self.session_factory = None

        logger.info("Database manager cleaned up successfully")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        return self._connection_stats.copy()

    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self.engine is not None and self.session_factory is not None


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """
    Get or create global database manager instance.

    Returns:
        DatabaseManager: Initialized database manager
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()

    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get database session.

    Yields:
        AsyncSession: Database session
    """
    db_manager = await get_database_manager()
    async with db_manager.get_session() as session:
        yield session


async def close_database_manager() -> None:
    """Close global database manager."""
    global _db_manager

    if _db_manager:
        await _db_manager.close()
        _db_manager = None


# Database utility functions for common operations
async def execute_sql_file(file_path: str) -> None:
    """
    Execute SQL commands from a file.

    Args:
        file_path: Path to SQL file

    Raises:
        DatabaseQueryError: If SQL execution fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            sql_content = file.read()

        # Split by semicolon and execute each statement
        statements = [
            stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
        ]

        db_manager = await get_database_manager()

        for statement in statements:
            if statement:
                await db_manager.execute_query(statement)

        logger.info(f"Successfully executed SQL file: {file_path}")

    except Exception as e:
        raise DatabaseQueryError(query=f"file:{file_path}", cause=e)


async def check_table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        table_name: Name of table to check

    Returns:
        True if table exists, False otherwise
    """
    try:
        db_manager = await get_database_manager()
        result = await db_manager.execute_query(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name =:table_name
            )
            """,
            parameters={"table_name": table_name},
            fetch_one=True,
        )
        return result[0] if result else False

    except Exception as e:
        logger.error(f"Failed to check table existence: {e}")
        return False


async def get_database_version() -> str:
    """
    Get database version information.

    Returns:
        Database version string
    """
    try:
        db_manager = await get_database_manager()
        result = await db_manager.execute_query(
            "SELECT version()", fetch_one=True
        )
        return result[0] if result else "Unknown"

    except Exception as e:
        logger.error(f"Failed to get database version: {e}")
        return "Error"


async def get_timescaledb_version() -> Optional[str]:
    """
    Get TimescaleDB version information.

    Returns:
        TimescaleDB version string or None if not available
    """
    try:
        db_manager = await get_database_manager()
        result = await db_manager.execute_query(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'",
            fetch_one=True,
        )
        return result[0] if result else None

    except Exception as e:
        logger.debug(f"TimescaleDB not available: {e}")
        return None
