"""
Database connection management for the occupancy prediction system.

This module provides async database connection management with connection pooling,
health checks, retry logic, and proper cleanup for PostgreSQL with TimescaleDB.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import hashlib
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

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

        # Retry configuration with timedelta support
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.backoff_multiplier = 2.0
        self.connection_timeout = timedelta(seconds=30)  # Connection timeout
        self.query_timeout = timedelta(seconds=120)  # Query timeout
        self.health_check_interval = timedelta(minutes=5)  # Health check interval

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
            self._health_check_task = asyncio.create_task(self._health_check_loop())

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
                connection_string=self.config.connection_string,
                cause=e,
            )

    async def _create_engine(self) -> None:
        """Create async SQLAlchemy engine with optimized settings."""
        # Parse connection string to add async driver
        conn_string = self.config.connection_string
        if conn_string.startswith("postgresql://"):
            conn_string = conn_string.replace("postgresql://", "postgresql+asyncpg://")
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

        # For async engines, don't specify poolclass - SQLAlchemy will use the correct async pool
        # QueuePool is not compatible with async engines - SQLAlchemy will use AsyncAdaptedQueuePool automatically
        if self.config.pool_size <= 0:
            # Only set NullPool for testing scenarios
            from sqlalchemy.pool import NullPool

            engine_kwargs["poolclass"] = NullPool

        self.engine = create_async_engine(**engine_kwargs)

        # Add connection event listeners
        self._setup_connection_events()

    def _setup_connection_events(self) -> None:
        """Setup database connection event listeners for monitoring with SQLAlchemy 2.0."""
        if self.engine is None:
            raise RuntimeError("Engine must be created before setting up events")

        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self._connection_stats["total_connections"] += 1
            logger.debug(
                "New database connection established",
                extra={
                    "total_connections": self._connection_stats["total_connections"]
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
                    "exception": (str(exception) if exception else "Unknown error"),
                    "failed_connections": self._connection_stats["failed_connections"],
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
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
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
                    self.base_delay * (self.backoff_multiplier ** (retry_count - 1)),
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
        timeout: Optional[timedelta] = None,
    ) -> Any:
        """
        Execute a raw SQL query with advanced error handling and timeout.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            fetch_one: Return single result
            fetch_all: Return all results
            timeout: Query timeout (uses default if None)

        Returns:
            Query result or None

        Raises:
            DatabaseQueryError: If query execution fails
        """
        query_timeout = timeout or self.query_timeout

        try:
            # Execute with timeout using asyncio.wait_for (compatible with all Python versions)
            async def _execute_query():
                async with self.get_session() as session:
                    result = await session.execute(text(query), parameters or {})

                    if fetch_one:
                        return result.fetchone()
                    elif fetch_all:
                        return result.fetchall()
                    else:
                        return result

            return await asyncio.wait_for(
                _execute_query(), timeout=query_timeout.total_seconds()
            )

        except SQLAlchemyError as e:
            # Specific SQLAlchemy error handling
            error_type = type(e).__name__
            logger.error(f"SQLAlchemy error ({error_type}) in query: {str(e)}")
            raise DatabaseQueryError(
                query=query,
                parameters=parameters,
                cause=e,
            )
        except asyncio.TimeoutError as e:
            logger.error(
                f"Query timeout after {query_timeout.total_seconds()}s: {query[:100]}..."
            )
            raise DatabaseQueryError(
                query=query,
                parameters=parameters,
                cause=e,
                error_type="TimeoutError",
                severity=ErrorSeverity.HIGH,
            )
        except Exception as e:
            logger.error(f"Unexpected database error: {str(e)}")
            raise DatabaseQueryError(
                query=query,
                parameters=parameters,
                cause=e,
            )

    async def execute_optimized_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_prepared_statement: bool = False,
        enable_query_cache: bool = True,
    ) -> Any:
        """
        Execute query with advanced optimization features.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            use_prepared_statement: Whether to use prepared statements
            enable_query_cache: Whether to enable query plan caching

        Returns:
            Query result
        """
        try:
            async with self.get_session() as session:
                # Enable query plan caching if requested
                if enable_query_cache:
                    await session.execute(
                        text("SET plan_cache_mode = force_generic_plan")
                    )

                # Prepare statement if requested (PostgreSQL specific)
                if use_prepared_statement and parameters:
                    # Create prepared statement name
                    statement_name = (
                        f"stmt_{hashlib.md5(query.encode()).hexdigest()[:8]}"
                    )

                    try:
                        # Prepare the statement
                        prepare_query = f"PREPARE {statement_name} AS {query}"
                        await session.execute(text(prepare_query))

                        # Execute prepared statement
                        param_values = [str(v) for v in parameters.values()]
                        execute_query = f"EXECUTE {statement_name}({', '.join(['%s'] * len(param_values))})"
                        result = await session.execute(
                            text(execute_query), param_values
                        )

                        # Deallocate prepared statement
                        await session.execute(text(f"DEALLOCATE {statement_name}"))

                    except SQLAlchemyError as e:
                        logger.warning(
                            f"Prepared statement failed, falling back to regular query: {e}"
                        )
                        result = await session.execute(text(query), parameters or {})
                else:
                    result = await session.execute(text(query), parameters or {})

                return result

        except SQLAlchemyError as e:
            logger.error(f"Optimized query execution failed: {e}")
            raise DatabaseQueryError(
                query=query,
                parameters=parameters,
                cause=e,
            )

    async def analyze_query_performance(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        include_execution_plan: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze query performance and provide optimization suggestions.

        Args:
            query: SQL query to analyze
            parameters: Query parameters
            include_execution_plan: Whether to include EXPLAIN ANALYZE results

        Returns:
            Dictionary with performance analysis
        """
        analysis = {
            "query": query[:200] + "..." if len(query) > 200 else query,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            async with self.get_session() as session:
                # Get execution plan
                if include_execution_plan:
                    explain_query = (
                        f"EXPLAIN (ANALYZE true, BUFFERS true, FORMAT json) {query}"
                    )

                    try:
                        result = await session.execute(
                            text(explain_query), parameters or {}
                        )
                        plan_data = result.fetchone()
                        if plan_data:
                            analysis["execution_plan"] = plan_data[0]
                    except SQLAlchemyError as e:
                        analysis["execution_plan_error"] = str(e)

                # Measure execution time
                start_time = time.time()

                try:
                    await session.execute(text(query), parameters or {})
                    execution_time = time.time() - start_time
                    analysis["execution_time_seconds"] = execution_time

                    # Performance assessment
                    if execution_time < 0.1:
                        analysis["performance_rating"] = "excellent"
                    elif execution_time < 1.0:
                        analysis["performance_rating"] = "good"
                    elif execution_time < 5.0:
                        analysis["performance_rating"] = "acceptable"
                    else:
                        analysis["performance_rating"] = "needs_optimization"

                except SQLAlchemyError as e:
                    analysis["execution_error"] = str(e)

                # Check for common performance issues
                analysis["optimization_suggestions"] = (
                    self._get_optimization_suggestions(query)
                )

        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    def _get_optimization_suggestions(self, query: str) -> List[str]:
        """Generate query optimization suggestions based on query analysis."""
        suggestions = []
        query_upper = query.upper()

        # Check for missing WHERE clauses on large tables
        if "FROM SENSOR_EVENTS" in query_upper and "WHERE" not in query_upper:
            suggestions.append("Add WHERE clause to filter sensor_events table")

        # Check for inefficient JOINs
        if "JOIN" in query_upper and "ON" not in query_upper:
            suggestions.append("Ensure JOINs have proper ON conditions")

        # Check for SELECT * usage
        if "SELECT *" in query_upper:
            suggestions.append("Specify explicit column names instead of SELECT *")

        # Check for ORDER BY without LIMIT
        if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            suggestions.append("Consider adding LIMIT clause with ORDER BY")

        # Check for potential N+1 queries
        if query.count("SELECT") > 1:
            suggestions.append(
                "Consider using JOINs instead of multiple SELECT statements"
            )

        return suggestions

    async def get_connection_pool_metrics(self) -> Dict[str, Any]:
        """
        Get detailed connection pool metrics using SQLAlchemy engine introspection.

        Returns:
            Dictionary with connection pool statistics
        """
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "pool_status": "unknown",
            "connection_stats": self._connection_stats.copy(),
        }

        if not self.engine:
            metrics["error"] = "Database engine not initialized"
            return metrics

        try:
            pool = self.engine.pool

            # Basic pool metrics
            metrics.update(
                {
                    "pool_size": getattr(pool, "_pool_size", 0),
                    "checked_out": getattr(pool, "_checked_out", 0),
                    "overflow": getattr(pool, "_overflow", 0),
                    "invalid_count": getattr(pool, "_invalidated", 0),
                }
            )

            # Calculate pool utilization
            total_connections = metrics["checked_out"] + metrics["overflow"]
            if metrics["pool_size"] > 0:
                metrics["utilization_percent"] = (
                    total_connections / metrics["pool_size"]
                ) * 100
            else:
                metrics["utilization_percent"] = 0

            # Pool health assessment
            if metrics["utilization_percent"] < 50:
                metrics["pool_status"] = "healthy"
            elif metrics["utilization_percent"] < 80:
                metrics["pool_status"] = "moderate"
            else:
                metrics["pool_status"] = "high_utilization"

            # Add recommendations
            if metrics["pool_status"] == "high_utilization":
                metrics["recommendations"] = [
                    "Consider increasing pool_size",
                    "Check for connection leaks",
                    "Optimize long-running queries",
                ]
            elif metrics["invalid_count"] > 0:
                metrics["recommendations"] = [
                    "Check database connectivity",
                    "Review connection timeout settings",
                ]
            else:
                metrics["recommendations"] = ["Pool operating normally"]

        except Exception as e:
            metrics["error"] = f"Failed to get pool metrics: {str(e)}"

        return metrics

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

                # Check TimescaleDB status and extract version information
                try:
                    result = await session.execute(
                        text("SELECT timescaledb_information.get_version_info()")
                    )

                    # Extract TimescaleDB version information from the result
                    version_info = {}
                    try:
                        version_row = result.fetchone()
                        if version_row and version_row[0]:
                            # Parse the version info (format: "TimescaleDB version X.Y.Z on PostgreSQL A.B.C")
                            version_string = str(version_row[0])
                            version_info["full_version"] = version_string

                            # Extract TimescaleDB version number
                            if "TimescaleDB version" in version_string:
                                # Extract version between "TimescaleDB version " and " on" (or end of string)
                                start = version_string.find("TimescaleDB version ") + 20
                                end = version_string.find(" on", start)
                                if end == -1:  # No " on" found, use end of string
                                    end = len(version_string)
                                if end > start:
                                    version_info["timescale_version"] = version_string[
                                        start:end
                                    ].strip()

                            # Extract PostgreSQL version
                            if "PostgreSQL" in version_string:
                                pg_start = version_string.find("PostgreSQL ") + 11
                                # Find end of version (next space or end of string)
                                pg_end = version_string.find(" ", pg_start)
                                if pg_end == -1:
                                    pg_end = len(version_string)
                                version_info["postgresql_version"] = version_string[
                                    pg_start:pg_end
                                ]

                    except Exception as parse_error:
                        logger.debug(
                            f"Failed to parse TimescaleDB version info: {parse_error}"
                        )
                        version_info["parse_error"] = str(parse_error)

                    health_status["timescale_status"] = "available"
                    health_status["timescale_version_info"] = version_info

                except Exception as timescale_error:
                    health_status["timescale_status"] = "unavailable"
                    health_status["timescale_version_info"] = {
                        "error": str(timescale_error)
                    }
                    logger.debug(f"TimescaleDB version check failed: {timescale_error}")

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
                                    pool.overflow() if hasattr(pool, "overflow") else 0
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
                            "message": self._connection_stats["last_connection_error"],
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
        """Background task for periodic health checks using configurable interval."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval.total_seconds())
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                # Wait 1 minute before retry on error
                await asyncio.sleep(60)

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
        statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

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
        result = await db_manager.execute_query("SELECT version()", fetch_one=True)
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
