"""
Data storage module for the occupancy prediction system.

This module provides database models, connection management, and storage utilities
for handling time-series sensor data, predictions, and model performance tracking.
"""

from .database import DatabaseManager
from .database import check_table_exists
from .database import close_database_manager
from .database import execute_sql_file
from .database import get_database_manager
from .database import get_database_version
from .database import get_db_session
from .database import get_timescaledb_version
from .models import Base
from .models import FeatureStore
from .models import ModelAccuracy
from .models import Prediction
from .models import RoomState
from .models import SensorEvent
from .models import create_timescale_hypertables
from .models import get_bulk_insert_query
from .models import optimize_database_performance

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
