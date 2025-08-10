#!/usr/bin/env python3
"""
Database setup script for the occupancy prediction system.

This script creates the database, user, installs TimescaleDB extension,
creates tables, sets up hypertables, indexes, and applies optimization policies.
It's designed to be idempotent and handle both fresh installs and updates.
"""

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import asyncpg

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.config import get_config
from core.exceptions import DatabaseConnectionError, DatabaseMigrationError
from data.storage.database import DatabaseManager
from data.storage.models import (
    Base,
    create_timescale_hypertables,
    optimize_database_performance,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """
    Handles database initialization and setup for the occupancy prediction system.
    """

    def __init__(self):
        """Initialize database setup with configuration."""
        self.config = get_config()
        self.db_config = self.config.database

        # Parse connection string components
        self.connection_parts = self._parse_connection_string()

        # Database setup configuration
        self.setup_config = {
            "database_name": "occupancy_prediction",
            "username": "occupancy_user",
            "password": "occupancy_pass",
            "host": "localhost",
            "port": 5432,
            "extensions": ["timescaledb"],
            "retention_days": 730,  # 2 years
            "compression_days": 7,
        }

        # Override with parsed values
        self.setup_config.update(self.connection_parts)

    def _parse_connection_string(self) -> Dict[str, Any]:
        """Parse database connection string to extract components."""
        import re

        # Remove async driver prefix if present
        conn_str = self.db_config.connection_string
        if conn_str.startswith("postgresql+asyncpg://"):
            conn_str = conn_str.replace("postgresql+asyncpg://", "postgresql://")

        # Parse connection string
        pattern = r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)"
        match = re.match(pattern, conn_str)

        if not match:
            raise ValueError(f"Invalid connection string format: {conn_str}")

        return {
            "username": match.group(1),
            "password": match.group(2),
            "host": match.group(3),
            "port": int(match.group(4)),
            "database_name": match.group(5),
        }

    async def setup_database(self, force_recreate: bool = False) -> None:
        """
        Complete database setup process.

        Args:
            force_recreate: If True, drop and recreate database
        """
        logger.info("Starting database setup...")

        try:
            # Step 1: Create database and user
            await self._create_database_and_user(force_recreate)

            # Step 2: Install extensions
            await self._install_extensions()

            # Step 3: Create tables and schema
            await self._create_schema()

            # Step 4: Setup TimescaleDB hypertables
            await self._setup_timescaledb()

            # Step 5: Create indexes and constraints
            await self._create_indexes()

            # Step 6: Setup policies and optimizations
            await self._setup_policies()

            # Step 7: Verify setup
            await self._verify_setup()

            logger.info("Database setup completed successfully!")

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise DatabaseMigrationError("database_setup", cause=e)

    async def _create_database_and_user(self, force_recreate: bool = False) -> None:
        """Create database and user if they don't exist."""
        logger.info("Creating database and user...")

        # Connect to PostgreSQL server (not specific database)
        try:
            conn = await asyncpg.connect(
                user="postgres",  # Assume postgres superuser exists
                password=os.getenv("POSTGRES_PASSWORD", "postgres"),
                host=self.setup_config["host"],
                port=self.setup_config["port"],
                database="postgres",
            )

            # Check if database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.setup_config["database_name"],
            )

            if force_recreate and db_exists:
                logger.warning(
                    f"Dropping existing database: {self.setup_config['database_name']}"
                )

                # Terminate connections to the database
                await conn.execute(
                    f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{self.setup_config['database_name']}'
                    AND pid <> pg_backend_pid()
                """
                )

                await conn.execute(
                    f"DROP DATABASE IF EXISTS {self.setup_config['database_name']}"
                )
                db_exists = False

            if not db_exists:
                logger.info(f"Creating database: {self.setup_config['database_name']}")
                await conn.execute(
                    f"CREATE DATABASE {self.setup_config['database_name']}"
                )
            else:
                logger.info(
                    f"Database already exists: {self.setup_config['database_name']}"
                )

            # Check if user exists
            user_exists = await conn.fetchval(
                "SELECT 1 FROM pg_user WHERE usename = $1",
                self.setup_config["username"],
            )

            if not user_exists:
                logger.info(f"Creating user: {self.setup_config['username']}")
                await conn.execute(
                    f"""
                    CREATE USER {self.setup_config['username']} 
                    WITH PASSWORD '{self.setup_config['password']}'
                """
                )
            else:
                logger.info(f"User already exists: {self.setup_config['username']}")

            # Grant privileges
            await conn.execute(
                f"""
                GRANT ALL PRIVILEGES ON DATABASE {self.setup_config['database_name']} 
                TO {self.setup_config['username']}
            """
            )

            await conn.close()

        except Exception as e:
            logger.error(f"Failed to create database and user: {e}")
            raise

    async def _install_extensions(self) -> None:
        """Install required PostgreSQL extensions."""
        logger.info("Installing PostgreSQL extensions...")

        conn = await asyncpg.connect(
            user=self.setup_config["username"],
            password=self.setup_config["password"],
            host=self.setup_config["host"],
            port=self.setup_config["port"],
            database=self.setup_config["database_name"],
        )

        try:
            for extension in self.setup_config["extensions"]:
                # Check if extension exists
                ext_exists = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = $1", extension
                )

                if not ext_exists:
                    logger.info(f"Installing extension: {extension}")
                    try:
                        await conn.execute(
                            f"CREATE EXTENSION IF NOT EXISTS {extension}"
                        )
                    except Exception as e:
                        if extension == "timescaledb":
                            logger.warning(f"TimescaleDB extension not available: {e}")
                            logger.warning("Some time-series features will be disabled")
                        else:
                            raise
                else:
                    logger.info(f"Extension already installed: {extension}")

        finally:
            await conn.close()

    async def _create_schema(self) -> None:
        """Create database schema from SQLAlchemy models."""
        logger.info("Creating database schema...")

        # Use our database manager to create tables
        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            # Create all tables
            async with db_manager.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")

        finally:
            await db_manager.close()

    async def _setup_timescaledb(self) -> None:
        """Setup TimescaleDB hypertables and configurations."""
        logger.info("Setting up TimescaleDB hypertables...")

        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            # Check if TimescaleDB is available
            timescale_version = await db_manager.execute_query(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'",
                fetch_one=True,
            )

            if not timescale_version:
                logger.warning("TimescaleDB not available, skipping hypertable setup")
                return

            logger.info(f"TimescaleDB version: {timescale_version[0]}")

            # Create hypertables and setup policies
            async with db_manager.get_session() as session:
                await create_timescale_hypertables(session)
                logger.info("TimescaleDB hypertables and policies created")

        finally:
            await db_manager.close()

    async def _create_indexes(self) -> None:
        """Create additional database indexes for optimization."""
        logger.info("Creating database indexes...")

        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            # Additional indexes for performance
            additional_indexes = [
                # Fast lookups for real-time predictions
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_events_realtime
                ON sensor_events (room_id, timestamp DESC, sensor_type)
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                """,
                # Optimize feature store queries
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_store_lookup
                ON feature_store (room_id, feature_timestamp DESC, feature_version)
                WHERE expires_at IS NULL OR expires_at > NOW()
                """,
                # Optimize prediction validation queries
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_validation
                ON predictions (room_id, predicted_transition_time, validation_timestamp)
                WHERE actual_transition_time IS NULL
                """,
                # Room state transitions
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_room_state_transitions
                ON room_states (room_id, timestamp DESC, is_occupied)
                WHERE transition_trigger IS NOT NULL
                """,
                # Model accuracy tracking
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_accuracy_trend
                ON model_accuracy (room_id, model_type, measurement_end DESC)
                """,
            ]

            for index_sql in additional_indexes:
                try:
                    await db_manager.execute_query(index_sql)
                    logger.debug("Created index successfully")
                except Exception as e:
                    # Index creation failures are often non-critical
                    logger.warning(f"Index creation warning: {e}")

            logger.info("Database indexes created")

        finally:
            await db_manager.close()

    async def _setup_policies(self) -> None:
        """Setup database policies and optimizations."""
        logger.info("Setting up database policies...")

        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            # Apply performance optimizations
            async with db_manager.get_session() as session:
                await optimize_database_performance(session)

            # Additional database settings for TimescaleDB
            optimization_queries = [
                # Optimize checkpoint and WAL settings
                "ALTER SYSTEM SET checkpoint_completion_target = 0.8",
                "ALTER SYSTEM SET wal_buffers = '16MB'",
                "ALTER SYSTEM SET shared_buffers = '256MB'",
                # Optimize for analytical workloads
                "ALTER SYSTEM SET effective_cache_size = '1GB'",
                "ALTER SYSTEM SET maintenance_work_mem = '64MB'",
                "ALTER SYSTEM SET work_mem = '8MB'",
                # Enable query optimization
                "ALTER SYSTEM SET enable_hashjoin = on",
                "ALTER SYSTEM SET enable_mergejoin = on",
                "ALTER SYSTEM SET enable_sort = on",
            ]

            for query in optimization_queries:
                try:
                    await db_manager.execute_query(query)
                except Exception as e:
                    # System settings might require superuser
                    logger.debug(f"System setting skipped: {e}")

            # Reload configuration
            try:
                await db_manager.execute_query("SELECT pg_reload_conf()")
            except Exception as e:
                logger.debug(f"Configuration reload skipped: {e}")

            logger.info("Database policies configured")

        finally:
            await db_manager.close()

    async def _verify_setup(self) -> None:
        """Verify database setup is working correctly."""
        logger.info("Verifying database setup...")

        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            # Test basic connectivity and tables
            tables_to_check = [
                "sensor_events",
                "room_states",
                "predictions",
                "model_accuracy",
                "feature_store",
            ]

            async with db_manager.get_session() as session:
                for table in tables_to_check:
                    result = await session.execute(
                        f"""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """
                    )
                    if result.scalar() == 0:
                        raise Exception(f"Table {table} not found")

                logger.info("All required tables verified")

                # Test TimescaleDB features
                try:
                    result = await session.execute(
                        """
                        SELECT * FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'sensor_events'
                    """
                    )
                    if result.fetchone():
                        logger.info("TimescaleDB hypertables verified")
                    else:
                        logger.warning("TimescaleDB hypertables not found")
                except Exception:
                    logger.info(
                        "TimescaleDB verification skipped (extension not available)"
                    )

                # Test sample data insertion
                await session.execute(
                    """
                    INSERT INTO sensor_events (
                        timestamp, room_id, sensor_id, sensor_type, state,
                        previous_state, is_human_triggered
                    ) VALUES (
                        NOW(), 'test_room', 'test_sensor', 'motion', 'on',
                        'off', true
                    ) ON CONFLICT DO NOTHING
                """
                )

                # Clean up test data
                await session.execute(
                    """
                    DELETE FROM sensor_events 
                    WHERE room_id = 'test_room' AND sensor_id = 'test_sensor'
                """
                )

                logger.info("Database write/read operations verified")

            # Perform health check
            health = await db_manager.health_check()
            if health["status"] != "healthy":
                logger.warning(
                    f"Database health check warnings: {health.get('errors', [])}"
                )
            else:
                logger.info("Database health check passed")

            logger.info("Database setup verification completed successfully")

        finally:
            await db_manager.close()

    async def create_sample_data(self) -> None:
        """Create sample data for testing (optional)."""
        logger.info("Creating sample data...")

        db_manager = DatabaseManager(self.db_config)
        await db_manager.initialize()

        try:
            from datetime import datetime, timedelta
            import random

            sample_rooms = ["living_room", "bedroom", "kitchen", "office"]
            sensor_types = ["motion", "door", "presence"]
            states = ["on", "off", "detected", "clear"]

            async with db_manager.get_session() as session:
                base_time = datetime.utcnow() - timedelta(days=7)

                for room in sample_rooms:
                    for i in range(100):  # 100 events per room
                        timestamp = base_time + timedelta(
                            minutes=random.randint(0, 10080)
                        )  # Week span

                        await session.execute(
                            """
                            INSERT INTO sensor_events (
                                timestamp, room_id, sensor_id, sensor_type, state,
                                previous_state, is_human_triggered, confidence_score
                            ) VALUES (
                                :timestamp, :room_id, :sensor_id, :sensor_type, :state,
                                :previous_state, :is_human_triggered, :confidence_score
                            )
                        """,
                            {
                                "timestamp": timestamp,
                                "room_id": room,
                                "sensor_id": f"{room}_sensor_{random.randint(1, 3)}",
                                "sensor_type": random.choice(sensor_types),
                                "state": random.choice(states),
                                "previous_state": random.choice(states),
                                "is_human_triggered": random.choice([True, False]),
                                "confidence_score": random.uniform(0.5, 1.0),
                            },
                        )

                logger.info(f"Created sample data for {len(sample_rooms)} rooms")

        finally:
            await db_manager.close()

    def get_setup_summary(self) -> Dict[str, Any]:
        """Get summary of setup configuration."""
        return {
            "database": self.setup_config["database_name"],
            "host": f"{self.setup_config['host']}:{self.setup_config['port']}",
            "user": self.setup_config["username"],
            "extensions": self.setup_config["extensions"],
            "retention_days": self.setup_config["retention_days"],
            "compression_days": self.setup_config["compression_days"],
        }


async def main():
    """Main setup function."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup occupancy prediction database")
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate database if it exists",
    )
    parser.add_argument(
        "--sample-data", action="store_true", help="Create sample data for testing"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing setup"
    )

    args = parser.parse_args()

    setup = DatabaseSetup()

    # Print setup summary
    summary = setup.get_setup_summary()
    logger.info("Database setup configuration:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    try:
        if args.verify_only:
            await setup._verify_setup()
        else:
            await setup.setup_database(force_recreate=args.force_recreate)

            if args.sample_data:
                await setup.create_sample_data()

        logger.info("Setup completed successfully!")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
