"""
Database compatibility layer for handling differences between SQLite and PostgreSQL/TimescaleDB.

This module provides utilities to configure SQLAlchemy models for different database backends,
ensuring that the same models work with both SQLite (for testing) and PostgreSQL/TimescaleDB (for production).
"""

from typing import Any, Dict

from sqlalchemy import Index, event, text
from sqlalchemy.engine import Engine


def is_sqlite_engine(engine_or_url) -> bool:
    """Check if the engine or URL is for SQLite."""
    if hasattr(engine_or_url, "url"):
        url_str = str(engine_or_url.url)
    else:
        url_str = str(engine_or_url)
    return "sqlite" in url_str.lower()


def is_postgresql_engine(engine_or_url) -> bool:
    """Check if the engine or URL is for PostgreSQL."""
    if hasattr(engine_or_url, "url"):
        url_str = str(engine_or_url.url)
    else:
        url_str = str(engine_or_url)
    return "postgresql" in url_str.lower()


def configure_sensor_event_model(base_model_class, engine):
    """
    Configure the SensorEvent model for database-specific requirements.

    Args:
        base_model_class: The base SensorEvent model class
        engine: SQLAlchemy engine

    Returns:
        Configured model class
    """

    if is_sqlite_engine(engine):
        # SQLite configuration: Single primary key with autoincrement
        # Add composite unique constraint to simulate composite primary key behavior

        # Modify the id column to have autoincrement
        if hasattr(base_model_class, "__table__"):
            table = base_model_class.__table__
            id_col = table.c.id
            id_col.autoincrement = True

            # Remove timestamp from primary key if it's there
            timestamp_col = table.c.timestamp
            if timestamp_col.primary_key:
                timestamp_col.primary_key = False

            # Add unique constraint on (id, timestamp) to maintain data integrity
            from sqlalchemy import UniqueConstraint

            table.append_constraint(
                UniqueConstraint(
                    "id", "timestamp", name="uq_sensor_events_id_timestamp"
                )
            )

    elif is_postgresql_engine(engine):
        # PostgreSQL/TimescaleDB configuration: Composite primary key
        if hasattr(base_model_class, "__table__"):
            table = base_model_class.__table__

            # Ensure both id and timestamp are in primary key
            id_col = table.c.id
            timestamp_col = table.c.timestamp

            id_col.primary_key = True
            id_col.autoincrement = True
            timestamp_col.primary_key = True

    return base_model_class


def create_database_specific_models(
    base_model_classes: Dict[str, Any], engine
) -> Dict[str, Any]:
    """
    Create database-specific model configurations.

    Args:
        base_model_classes: Dictionary of model class names to classes
        engine: SQLAlchemy engine

    Returns:
        Dictionary of configured model classes
    """
    configured_models = {}

    for name, model_class in base_model_classes.items():
        if name == "SensorEvent":
            configured_models[name] = configure_sensor_event_model(model_class, engine)
        else:
            # Other models don't need special configuration
            configured_models[name] = model_class

    return configured_models


# Event listeners for automatic model configuration
@event.listens_for(Engine, "connect")
def configure_sqlite_for_testing(dbapi_connection, connection_record):
    """Configure SQLite connections for testing."""
    if "sqlite" in str(connection_record.info.get("url", "")):
        # Enable foreign key constraints for SQLite
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@event.listens_for(Engine, "first_connect")
def configure_database_on_first_connect(dbapi_connection, connection_record):
    """Configure database-specific settings on first connection."""
    engine_url = str(connection_record.info.get("url", ""))

    if "sqlite" in engine_url.lower():
        # SQLite-specific configuration
        cursor = dbapi_connection.cursor()
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        # Set journal mode to WAL for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set synchronous mode to NORMAL for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    elif "postgresql" in engine_url.lower():
        # PostgreSQL-specific configuration
        # Can add PostgreSQL-specific settings here if needed
        pass


def patch_models_for_sqlite_compatibility():
    """
    Monkey-patch the existing models to work with SQLite.

    This is a temporary solution for testing environments.
    """
    from src.data.storage.models import SensorEvent

    # Temporarily modify the SensorEvent model for SQLite
    if hasattr(SensorEvent, "__table__"):
        table = SensorEvent.__table__

        # Remove timestamp from primary key constraints
        timestamp_col = table.c.timestamp
        if timestamp_col.primary_key:
            timestamp_col.primary_key = False

        # Ensure id column has autoincrement
        id_col = table.c.id
        id_col.autoincrement = True


def get_database_specific_table_args(engine, table_name: str) -> tuple:
    """
    Get database-specific table arguments.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table

    Returns:
        Tuple of table arguments
    """
    base_args = []

    if table_name == "sensor_events":
        if is_sqlite_engine(engine):
            # SQLite-specific indexes and constraints
            from sqlalchemy import UniqueConstraint

            base_args.extend(
                [
                    Index("idx_room_sensor_time", "room_id", "sensor_id", "timestamp"),
                    Index("idx_room_time_desc", "room_id", "timestamp"),
                    Index("idx_sensor_type_time", "sensor_type", "timestamp"),
                    Index("idx_human_triggered", "is_human_triggered", "timestamp"),
                    # Unique constraint to simulate composite primary key behavior
                    UniqueConstraint(
                        "id", "timestamp", name="uq_sensor_events_id_timestamp"
                    ),
                ]
            )
        elif is_postgresql_engine(engine):
            # PostgreSQL/TimescaleDB-specific indexes and constraints
            base_args.extend(
                [
                    Index("idx_room_sensor_time", "room_id", "sensor_id", "timestamp"),
                    Index(
                        "idx_room_time_desc",
                        "room_id",
                        "timestamp",
                        postgresql_using="btree",
                    ),
                    Index(
                        "idx_state_changes",
                        "room_id",
                        "timestamp",
                        postgresql_where=text("state != previous_state"),
                    ),
                    Index("idx_sensor_type_time", "sensor_type", "timestamp"),
                    Index("idx_human_triggered", "is_human_triggered", "timestamp"),
                ]
            )

    return tuple(base_args)
