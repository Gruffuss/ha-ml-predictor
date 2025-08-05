"""
Data storage module for the occupancy prediction system.

This module provides database models, connection management, and storage utilities
for handling time-series sensor data, predictions, and model performance tracking.
"""

from .database import (
    DatabaseManager,
    check_table_exists,
    close_database_manager,
    execute_sql_file,
    get_database_manager,
    get_database_version,
    get_db_session,
    get_timescaledb_version,
)
from .models import (
    Base,
    FeatureStore,
    ModelAccuracy,
    Prediction,
    RoomState,
    SensorEvent,
    create_timescale_hypertables,
    get_bulk_insert_query,
    optimize_database_performance,
)

__all__ = [
    # Models
    "Base",
    "SensorEvent",
    "RoomState",
    "Prediction",
    "ModelAccuracy",
    "FeatureStore",
    # Model utilities
    "create_timescale_hypertables",
    "optimize_database_performance",
    "get_bulk_insert_query",
    # Database management
    "DatabaseManager",
    "get_database_manager",
    "get_db_session",
    "close_database_manager",
    # Database utilities
    "execute_sql_file",
    "check_table_exists",
    "get_database_version",
    "get_timescaledb_version",
]
