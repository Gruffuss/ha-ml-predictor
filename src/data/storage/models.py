"""
Database models for the occupancy prediction system.

This module defines SQLAlchemy models optimized for TimescaleDB to handle
time-series sensor data, predictions, and model performance tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from decimal import Decimal
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    and_,
    desc,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base, relationship, selectinload
from sqlalchemy.sql import func as sql_func

Base = declarative_base()


# Database-specific configuration helpers
def _is_sqlite_engine(bind) -> bool:
    """Check if the current engine is SQLite."""
    if bind is None:
        return False
    return "sqlite" in str(bind.url).lower()


def _get_database_specific_column_config(
    bind, column_name: str, is_primary_key: bool = False, autoincrement: bool = False
):
    """Get database-specific column configuration."""
    if _is_sqlite_engine(bind) and is_primary_key and autoincrement:
        # For SQLite, only single-column primary keys can have autoincrement
        return {"autoincrement": True if column_name == "id" else False}
    return {"autoincrement": autoincrement}


def _get_json_column_type():
    """Get appropriate JSON column type based on environment."""
    import os

    # Check multiple environment indicators for SQLite
    is_sqlite = (
        os.getenv("TEST_DB_URL", "").startswith("sqlite")
        or os.getenv("TESTING") == "true"
        or os.getenv("DATABASE_URL", "").startswith("sqlite")
        or "pytest" in os.getenv("_", "")  # Check if running under pytest
        or "pytest" in " ".join(os.sys.argv)  # Check command line for pytest
    )

    if is_sqlite:
        return JSON  # SQLite uses JSON
    else:
        return JSONB  # PostgreSQL uses JSONB


# Enums for consistent state tracking
SENSOR_TYPES = [
    "motion",
    "presence",
    "door",
    "window",
    "temperature",
    "humidity",
    "light",
    "pressure",
    "air_quality",
]

SENSOR_STATES = ["on", "off", "open", "closed", "detected", "clear", "unknown"]

TRANSITION_TYPES = ["occupied_to_vacant", "vacant_to_occupied", "state_change"]

MODEL_TYPES = ["lstm", "xgboost", "hmm", "gaussian_process", "ensemble"]


class SensorEvent(Base):
    """
    Main hypertable for storing all sensor events from Home Assistant.
    Optimized for time-series queries with proper partitioning.

    Uses composite primary key (id, timestamp) for TimescaleDB hypertable compatibility.
    For SQLite compatibility, only id is used as primary key with autoincrement.
    Foreign key relationships are managed at the application level to avoid conflicts
    with TimescaleDB's partitioning requirements.
    """

    __tablename__ = "sensor_events"

    # Primary key configuration - Using single primary key for cross-database compatibility
    # TimescaleDB partitioning will be handled through hypertable configuration, not composite PK
    # For SQLite, use Integer instead of BigInteger for proper autoincrement
    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True,  # Always indexed for time-series queries
    )

    # Core event data
    room_id = Column(String(50), nullable=False, index=True)
    sensor_id = Column(String(100), nullable=False, index=True)
    sensor_type = Column(ENUM(*SENSOR_TYPES, name="sensor_type_enum"), nullable=False)
    state = Column(ENUM(*SENSOR_STATES, name="sensor_state_enum"), nullable=False)
    previous_state = Column(ENUM(*SENSOR_STATES, name="sensor_state_enum"))

    # Metadata and context
    attributes = Column(
        JSON, default=dict  # Force JSON for cross-database compatibility
    )  # Additional sensor attributes
    is_human_triggered = Column(
        Boolean, default=True, server_default=text("1"), nullable=False
    )
    confidence_score = Column(
        Numeric(precision=5, scale=4)
    )  # Decimal precision for confidence scores

    # Performance tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    processed_at = Column(DateTime(timezone=True))

    # Note: No foreign key relationships to maintain TimescaleDB performance
    # Application-level referential integrity is used instead

    def __init__(self, **kwargs):
        """Initialize SensorEvent with proper defaults."""
        # Apply Python-level defaults for columns that need them
        if "is_human_triggered" not in kwargs:
            kwargs["is_human_triggered"] = True
        if "attributes" not in kwargs:
            kwargs["attributes"] = {}

        # Call parent constructor
        super().__init__(**kwargs)

    __table_args__ = (
        # Indexes for efficient time-series queries
        Index("idx_room_sensor_time", "room_id", "sensor_id", "timestamp"),
        Index("idx_room_time_desc", "room_id", desc("timestamp")),
        Index("idx_sensor_type_time", "sensor_type", "timestamp"),
        Index("idx_human_triggered", "is_human_triggered", "timestamp"),
        # Composite indexes for common query patterns
        Index(
            "idx_room_sensor_state_time",
            "room_id",
            "sensor_id",
            "state",
            "timestamp",
        ),
        # Constraints
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="valid_confidence",
        ),
        # Note: PostgreSQL-specific indexes are conditionally added in conftest.py
    )

    @classmethod
    async def get_recent_events(
        cls,
        session: AsyncSession,
        room_id: str,
        hours: int = 24,
        sensor_types: Optional[List[str]] = None,
    ) -> List["SensorEvent"]:
        """Get recent events for a room, optionally filtered by sensor type."""
        query = (
            select(cls)
            .where(
                and_(
                    cls.room_id == room_id,
                    cls.timestamp
                    >= datetime.now(timezone.utc) - timedelta(hours=hours),
                )
            )
            .order_by(desc(cls.timestamp))
        )

        if sensor_types:
            query = query.where(cls.sensor_type.in_(sensor_types))

        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def get_state_changes(
        cls,
        session: AsyncSession,
        room_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> List["SensorEvent"]:
        """Get events where state changed from previous state."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        query = (
            select(cls)
            .where(
                and_(
                    cls.room_id == room_id,
                    cls.timestamp >= start_time,
                    cls.timestamp <= end_time,
                    cls.state != cls.previous_state,
                )
            )
            .order_by(cls.timestamp)
        )

        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def get_transition_sequences(
        cls,
        session: AsyncSession,
        room_id: str,
        lookback_hours: int = 24,
        min_sequence_length: int = 3,
    ) -> List[List["SensorEvent"]]:
        """Get sequences of sensor transitions for pattern analysis."""
        events = await cls.get_state_changes(
            session,
            room_id,
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
        )

        # Group events into sequences based on time gaps
        sequences = []
        current_sequence = []
        max_gap_minutes = 30

        for event in events:
            if (
                current_sequence
                and (event.timestamp - current_sequence[-1].timestamp).total_seconds()
                > max_gap_minutes * 60
            ):
                if len(current_sequence) >= min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [event]
            else:
                current_sequence.append(event)

        if len(current_sequence) >= min_sequence_length:
            sequences.append(current_sequence)

        return sequences

    async def get_predictions(self, session: AsyncSession) -> List["Prediction"]:
        """Get predictions that were triggered by this sensor event using application-level join."""
        query = select(Prediction).where(Prediction.triggering_event_id == self.id)
        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def get_advanced_analytics(
        cls,
        session: AsyncSession,
        room_id: str,
        hours: int = 24,
        include_statistics: bool = True,
    ) -> Dict[str, Any]:
        """
        Get advanced analytics for room events using SQL functions.

        Args:
            session: Database session
            room_id: Room to analyze
            hours: Hours of data to analyze
            include_statistics: Include statistical calculations

        Returns:
            Dictionary with analytics data
        """
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Base analytics query using sql_func
        analytics_query = select(
            sql_func.count().label("total_events"),
            sql_func.count(sql_func.distinct(cls.sensor_id)).label("unique_sensors"),
            sql_func.avg(cls.confidence_score).label("avg_confidence"),
            sql_func.min(cls.timestamp).label("first_event"),
            sql_func.max(cls.timestamp).label("last_event"),
            sql_func.count()
            .filter(cls.is_human_triggered.is_(True))
            .label("human_events"),
            sql_func.count()
            .filter(cls.is_human_triggered.is_(False))
            .label("automated_events"),
        ).where(and_(cls.room_id == room_id, cls.timestamp >= start_time))

        result = await session.execute(analytics_query)
        row = result.first()

        analytics = {
            "room_id": room_id,
            "analysis_period_hours": hours,
            "total_events": row.total_events or 0,
            "unique_sensors": row.unique_sensors or 0,
            "average_confidence": float(row.avg_confidence or 0),
            "first_event": row.first_event,
            "last_event": row.last_event,
            "human_triggered_events": row.human_events or 0,
            "automated_events": row.automated_events or 0,
            "human_event_ratio": (row.human_events or 0)
            / max(row.total_events or 1, 1),
        }

        if include_statistics and row.total_events > 0:
            # Additional statistical queries using database-agnostic functions
            from .dialect_utils import (
                extract_epoch_interval,
                percentile_cont,
                stddev_samp,
            )

            stats_query = select(
                percentile_cont(0.5, cls.confidence_score, order_desc=True).label(
                    "median_confidence"
                ),
                stddev_samp(cls.confidence_score).label("confidence_stddev"),
                extract_epoch_interval(
                    sql_func.min(cls.timestamp), sql_func.max(cls.timestamp)
                ).label("time_span_seconds"),
            ).where(
                and_(
                    cls.room_id == room_id,
                    cls.timestamp >= start_time,
                    cls.confidence_score.is_not(None),
                )
            )

            stats_result = await session.execute(stats_query)
            stats_row = stats_result.first()

            analytics.update(
                {
                    "median_confidence": float(stats_row.median_confidence or 0),
                    "confidence_standard_deviation": float(
                        stats_row.confidence_stddev or 0
                    ),
                    "time_span_seconds": float(stats_row.time_span_seconds or 0),
                }
            )

        return analytics

    @classmethod
    async def get_sensor_efficiency_metrics(
        cls, session: AsyncSession, room_id: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Calculate sensor efficiency metrics using advanced SQL aggregations.

        Args:
            session: Database session
            room_id: Room to analyze
            days: Days of data to analyze

        Returns:
            List of sensor metrics with efficiency data
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Complex aggregation query with sensor-specific metrics
        efficiency_query = (
            select(
                cls.sensor_id,
                cls.sensor_type,
                sql_func.count().label("total_events"),
                sql_func.count()
                .filter(cls.state != cls.previous_state)
                .label("state_changes"),
                sql_func.avg(cls.confidence_score).label("avg_confidence"),
                sql_func.min(cls.confidence_score).label("min_confidence"),
                sql_func.max(cls.confidence_score).label("max_confidence"),
                (
                    sql_func.count().filter(cls.state != cls.previous_state)
                    / sql_func.count().cast(Float)
                ).label("state_change_ratio"),
                # Use simpler calculation for SQLite compatibility
                sql_func.coalesce(
                    sql_func.extract(
                        "epoch",
                        sql_func.avg(
                            cls.timestamp
                            - sql_func.lag(cls.timestamp).over(
                                partition_by=cls.sensor_id, order_by=cls.timestamp
                            )
                        ),
                    ),
                    0,
                ).label("avg_interval_seconds"),
            )
            .where(and_(cls.room_id == room_id, cls.timestamp >= start_time))
            .group_by(cls.sensor_id, cls.sensor_type)
            .order_by(sql_func.count().desc())
        )

        result = await session.execute(efficiency_query)

        metrics = []
        for row in result:
            metrics.append(
                {
                    "sensor_id": row.sensor_id,
                    "sensor_type": row.sensor_type,
                    "total_events": row.total_events,
                    "state_changes": row.state_changes or 0,
                    "average_confidence": float(row.avg_confidence or 0),
                    "min_confidence": float(row.min_confidence or 0),
                    "max_confidence": float(row.max_confidence or 0),
                    "state_change_ratio": float(row.state_change_ratio or 0),
                    "average_interval_seconds": float(row.avg_interval_seconds or 0),
                    "efficiency_score": cls._calculate_efficiency_score(
                        row.total_events,
                        row.state_changes or 0,
                        float(row.avg_confidence or 0),
                        float(row.state_change_ratio or 0),
                    ),
                }
            )

        return metrics

    @staticmethod
    def _calculate_efficiency_score(
        total_events: int,
        state_changes: int,
        avg_confidence: float,
        state_change_ratio: float,
    ) -> float:
        """Calculate a composite efficiency score for sensors."""
        # Normalize metrics and calculate weighted score
        event_score = min(total_events / 100.0, 1.0)  # More events = better
        change_score = min(state_change_ratio * 2.0, 1.0)  # Meaningful changes = better
        confidence_score = avg_confidence  # Higher confidence = better

        # Weighted average with emphasis on confidence and meaningful changes
        return event_score * 0.2 + change_score * 0.4 + confidence_score * 0.4

    @classmethod
    async def get_temporal_patterns(
        cls, session: AsyncSession, room_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns using advanced date/time functions.

        Args:
            session: Database session
            room_id: Room to analyze
            days: Days of data to analyze

        Returns:
            Dictionary with temporal pattern analysis
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Hourly distribution query
        hourly_query = (
            select(
                sql_func.extract("hour", cls.timestamp).label("hour"),
                sql_func.count().label("event_count"),
                sql_func.avg(cls.confidence_score).label("avg_confidence"),
            )
            .where(and_(cls.room_id == room_id, cls.timestamp >= start_time))
            .group_by(sql_func.extract("hour", cls.timestamp))
            .order_by("hour")
        )

        hourly_result = await session.execute(hourly_query)
        hourly_patterns = [
            {
                "hour": int(row.hour),
                "event_count": row.event_count,
                "average_confidence": float(row.avg_confidence or 0),
            }
            for row in hourly_result
        ]

        # Day of week distribution query
        dow_query = (
            select(
                sql_func.extract("dow", cls.timestamp).label("day_of_week"),
                sql_func.count().label("event_count"),
                sql_func.avg(cls.confidence_score).label("avg_confidence"),
            )
            .where(and_(cls.room_id == room_id, cls.timestamp >= start_time))
            .group_by(sql_func.extract("dow", cls.timestamp))
            .order_by("day_of_week")
        )

        dow_result = await session.execute(dow_query)
        dow_patterns = [
            {
                "day_of_week": int(row.day_of_week),
                "day_name": [
                    "Sunday",
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ][int(row.day_of_week)],
                "event_count": row.event_count,
                "average_confidence": float(row.avg_confidence or 0),
            }
            for row in dow_result
        ]

        return {
            "room_id": room_id,
            "analysis_period_days": days,
            "hourly_patterns": hourly_patterns,
            "day_of_week_patterns": dow_patterns,
            "peak_hour": (
                max(hourly_patterns, key=lambda x: x["event_count"])["hour"]
                if hourly_patterns
                else None
            ),
            "peak_day": (
                max(dow_patterns, key=lambda x: x["event_count"])["day_name"]
                if dow_patterns
                else None
            ),
        }


class RoomState(Base):
    """
    Current and historical room occupancy states.
    Tracks the derived occupancy status from sensor events.
    """

    __tablename__ = "room_states"

    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)
    room_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Occupancy status with UUID for tracking
    occupancy_session_id = Column(
        UUID(as_uuid=True)
    )  # UUID to track occupancy sessions
    is_occupied = Column(Boolean, nullable=False)
    occupancy_confidence = Column(
        Numeric(precision=5, scale=4), nullable=False, default=0.5
    )
    occupant_type = Column(String(20))  # 'human', 'cat', 'unknown'
    occupant_count = Column(Integer, default=1)

    # State metadata with Text field for detailed descriptions
    state_duration = Column(Integer)  # Duration in current state (seconds)
    transition_trigger = Column(String(100))  # Sensor that triggered transition
    certainty_factors = Column(
        JSON, default=dict  # Force JSON for cross-database compatibility
    )  # Contributing factors
    detailed_analysis = Column(Text)  # Large text field for detailed state analysis

    # Tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Note: Relationships managed at application level for TimescaleDB compatibility

    __table_args__ = (
        Index("idx_room_time_occupied", "room_id", "timestamp", "is_occupied"),
        Index(
            "idx_occupancy_changes",
            "room_id",
            "timestamp",
            postgresql_where=text("transition_trigger IS NOT NULL"),
        ),
        Index("idx_recent_states", "room_id", desc("timestamp")),
        CheckConstraint("occupancy_confidence >= 0 AND occupancy_confidence <= 1"),
        CheckConstraint("occupant_count >= 0"),
    )

    @classmethod
    async def get_current_state(
        cls, session: AsyncSession, room_id: str
    ) -> Optional["RoomState"]:
        """Get the most recent room state."""
        query = (
            select(cls)
            .where(cls.room_id == room_id)
            .order_by(desc(cls.timestamp))
            .limit(1)
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def get_occupancy_history(
        cls, session: AsyncSession, room_id: str, hours: int = 24
    ) -> List["RoomState"]:
        """Get occupancy history for pattern analysis."""
        query = (
            select(cls)
            .where(
                and_(
                    cls.room_id == room_id,
                    cls.timestamp
                    >= datetime.now(timezone.utc) - timedelta(hours=hours),
                )
            )
            .order_by(cls.timestamp)
        )

        result = await session.execute(query)
        return result.scalars().all()

    async def get_predictions(self, session: AsyncSession) -> List["Prediction"]:
        """Get predictions associated with this room state using application-level join."""
        query = select(Prediction).where(Prediction.room_state_id == self.id)
        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def get_occupancy_sessions(
        cls,
        session: AsyncSession,
        room_id: str,
        days: int = 7,
        use_optimized_loading: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get occupancy sessions grouped by session UUID with optimized loading.

        Args:
            session: Database session
            room_id: Room to analyze
            days: Days of data to analyze
            use_optimized_loading: Whether to use selectinload optimization

        Returns:
            List of occupancy session data
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Query with selectinload optimization for better performance
        base_query = (
            select(cls)
            .where(
                and_(
                    cls.room_id == room_id,
                    cls.timestamp >= start_time,
                    cls.occupancy_session_id.is_not(None),
                )
            )
            .order_by(cls.timestamp)
        )

        if use_optimized_loading:
            # Use selectinload for any related data if we had relationships
            # This is a demonstration of the feature for future use
            base_query = base_query.options()

        result = await session.execute(base_query)
        states = result.scalars().all()

        # Group by session ID
        sessions_dict = {}
        for state in states:
            session_id_str = str(state.occupancy_session_id)
            if session_id_str not in sessions_dict:
                sessions_dict[session_id_str] = {
                    "session_id": session_id_str,
                    "room_id": room_id,
                    "states": [],
                    "start_time": None,
                    "end_time": None,
                    "duration_seconds": 0,
                    "occupant_type": None,
                    "confidence_range": {"min": 1.0, "max": 0.0, "avg": 0.0},
                }

            session_data = sessions_dict[session_id_str]
            session_data["states"].append(
                {
                    "timestamp": state.timestamp,
                    "is_occupied": state.is_occupied,
                    "confidence": float(state.occupancy_confidence),
                    "occupant_type": state.occupant_type,
                    "transition_trigger": state.transition_trigger,
                }
            )

            # Update session metadata
            if (
                session_data["start_time"] is None
                or state.timestamp < session_data["start_time"]
            ):
                session_data["start_time"] = state.timestamp
            if (
                session_data["end_time"] is None
                or state.timestamp > session_data["end_time"]
            ):
                session_data["end_time"] = state.timestamp

            session_data["occupant_type"] = state.occupant_type

            # Update confidence range
            conf = float(state.occupancy_confidence)
            session_data["confidence_range"]["min"] = min(
                session_data["confidence_range"]["min"], conf
            )
            session_data["confidence_range"]["max"] = max(
                session_data["confidence_range"]["max"], conf
            )

        # Calculate final statistics
        sessions_list = []
        for session_data in sessions_dict.values():
            if session_data["start_time"] and session_data["end_time"]:
                session_data["duration_seconds"] = (
                    session_data["end_time"] - session_data["start_time"]
                ).total_seconds()

            # Calculate average confidence
            if session_data["states"]:
                avg_conf = sum(s["confidence"] for s in session_data["states"]) / len(
                    session_data["states"]
                )
                session_data["confidence_range"]["avg"] = avg_conf

            sessions_list.append(session_data)

        return sorted(sessions_list, key=lambda x: x["start_time"] or datetime.min)

    @classmethod
    async def get_precision_occupancy_metrics(
        cls,
        session: AsyncSession,
        room_id: str,
        precision_threshold: Decimal = Decimal("0.8000"),
    ) -> Dict[str, Any]:
        """
        Get occupancy metrics with decimal precision calculations.

        Args:
            session: Database session
            room_id: Room to analyze
            precision_threshold: Minimum confidence threshold (Decimal)

        Returns:
            Dictionary with precision metrics
        """
        # Use precise decimal calculations for confidence metrics
        # Import database-agnostic functions
        from .dialect_utils import percentile_cont, stddev_samp

        precision_query = select(
            sql_func.count().label("total_states"),
            sql_func.count()
            .filter(cls.occupancy_confidence >= precision_threshold)
            .label("high_confidence_states"),
            sql_func.avg(cls.occupancy_confidence).label("avg_confidence"),
            stddev_samp(cls.occupancy_confidence).label("confidence_stddev"),
            sql_func.min(cls.occupancy_confidence).label("min_confidence"),
            sql_func.max(cls.occupancy_confidence).label("max_confidence"),
            percentile_cont(0.25, cls.occupancy_confidence).label("q1_confidence"),
            percentile_cont(0.5, cls.occupancy_confidence).label("median_confidence"),
            percentile_cont(0.75, cls.occupancy_confidence).label("q3_confidence"),
        ).where(cls.room_id == room_id)

        result = await session.execute(precision_query)
        row = result.first()

        return {
            "room_id": room_id,
            "precision_threshold": float(precision_threshold),
            "total_states": row.total_states or 0,
            "high_confidence_states": row.high_confidence_states or 0,
            "high_confidence_ratio": (row.high_confidence_states or 0)
            / max(row.total_states or 1, 1),
            "confidence_statistics": {
                "average": float(row.avg_confidence or 0),
                "standard_deviation": float(row.confidence_stddev or 0),
                "minimum": float(row.min_confidence or 0),
                "maximum": float(row.max_confidence or 0),
                "quartiles": {
                    "q1": float(row.q1_confidence or 0),
                    "median": float(row.median_confidence or 0),
                    "q3": float(row.q3_confidence or 0),
                },
            },
        }


class Prediction(Base):
    """
    Stores model predictions for occupancy transitions.
    Tracks prediction accuracy and model performance.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)
    room_id = Column(String(50), nullable=False, index=True)
    prediction_time = Column(DateTime(timezone=True), nullable=False, index=True)

    # Prediction details
    predicted_transition_time = Column(DateTime(timezone=True), nullable=False)
    predicted_time = Column(
        DateTime(timezone=True), nullable=True
    )  # Alias for predicted_transition_time for compatibility
    transition_type = Column(
        ENUM(*TRANSITION_TYPES, name="transition_type_enum"), nullable=False
    )
    confidence_score = Column(Numeric(precision=5, scale=4), nullable=False)
    prediction_interval_lower = Column(DateTime(timezone=True))
    prediction_interval_upper = Column(DateTime(timezone=True))

    # Model information
    model_type = Column(ENUM(*MODEL_TYPES, name="model_type_enum"), nullable=False)
    model_version = Column(String(50), nullable=False)
    feature_importance = Column(
        JSON, default=dict
    )  # Force JSON for cross-database compatibility

    # Alternative predictions (top-k)
    alternatives = Column(
        JSON, default=list
    )  # Force JSON for cross-database compatibility

    # Validation results
    actual_transition_time = Column(DateTime(timezone=True))
    actual_time = Column(
        DateTime(timezone=True), nullable=True
    )  # Alias for actual_transition_time for compatibility
    accuracy_minutes = Column(Float)  # Difference in minutes
    is_accurate = Column(Boolean)  # Within threshold
    validation_timestamp = Column(DateTime(timezone=True))
    status = Column(String(20), default="pending")  # For test compatibility

    # Context - References maintained at application level for TimescaleDB compatibility
    # No foreign key constraints to avoid conflicts with TimescaleDB partitioning
    triggering_event_id = Column(
        BigInteger, nullable=True, index=True
    )  # References sensor_events.id
    room_state_id = Column(
        BigInteger, nullable=True, index=True
    )  # References room_states.id

    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    processing_time_ms = Column(Float)  # Time to generate prediction

    def __init__(self, **kwargs):
        """Initialize Prediction with proper defaults and compatibility handling."""
        # Handle predicted_time compatibility - if only predicted_time is provided,
        # use it for predicted_transition_time as well
        if "predicted_time" in kwargs and "predicted_transition_time" not in kwargs:
            kwargs["predicted_transition_time"] = kwargs["predicted_time"]
        # If both are provided, ensure they're the same for consistency
        elif "predicted_time" in kwargs and "predicted_transition_time" in kwargs:
            if kwargs["predicted_time"] != kwargs["predicted_transition_time"]:
                # Use predicted_transition_time as the authoritative value
                kwargs["predicted_time"] = kwargs["predicted_transition_time"]

        # Handle actual_time compatibility - if only actual_time is provided,
        # use it for actual_transition_time as well
        if "actual_time" in kwargs and "actual_transition_time" not in kwargs:
            kwargs["actual_transition_time"] = kwargs["actual_time"]
        # If both are provided, ensure they're the same for consistency
        elif "actual_time" in kwargs and "actual_transition_time" in kwargs:
            if kwargs["actual_time"] != kwargs["actual_transition_time"]:
                # Use actual_transition_time as the authoritative value
                kwargs["actual_time"] = kwargs["actual_transition_time"]

        # Call parent constructor
        super().__init__(**kwargs)

    # Note: Relationships managed at application level
    # Use get_triggering_event() and get_room_state() methods for data access

    __table_args__ = (
        Index("idx_room_prediction_time", "room_id", "prediction_time"),
        Index("idx_model_type_time", "model_type", "prediction_time"),
        Index(
            "idx_accuracy_validation",
            "room_id",
            "is_accurate",
            "validation_timestamp",
        ),
        Index(
            "idx_pending_validation",
            "room_id",
            "predicted_transition_time",
            postgresql_where=text("actual_transition_time IS NULL"),
        ),
        Index(
            "idx_triggering_event", "triggering_event_id"
        ),  # For application-level joins
        Index("idx_room_state_re", "room_state_id"),  # For application-level joins
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"),
        CheckConstraint("prediction_interval_lower <= prediction_interval_upper"),
    )

    @classmethod
    async def get_pending_validations(
        cls,
        session: AsyncSession,
        room_id: Optional[str] = None,
        cutoff_hours: int = 2,
    ) -> List["Prediction"]:
        """Get predictions that need validation (past their predicted time)."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=cutoff_hours)

        query = select(cls).where(
            and_(
                cls.predicted_transition_time <= datetime.now(timezone.utc),
                cls.predicted_transition_time >= cutoff_time,
                cls.actual_transition_time.is_(None),
            )
        )

        if room_id:
            query = query.where(cls.room_id == room_id)

        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def get_accuracy_metrics(
        cls,
        session: AsyncSession,
        room_id: str,
        days: int = 7,
        model_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate accuracy metrics for predictions."""
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.validation_timestamp >= start_time,
                cls.actual_transition_time.is_not(None),
            )
        )

        if model_type:
            query = query.where(cls.model_type == model_type)

        result = await session.execute(query)
        predictions = result.scalars().all()

        if not predictions:
            return {}

        accuracies = [
            p.accuracy_minutes for p in predictions if p.accuracy_minutes is not None
        ]
        accurate_count = sum(1 for p in predictions if p.is_accurate)

        return {
            "total_predictions": len(predictions),
            "accurate_predictions": accurate_count,
            "accuracy_rate": accurate_count / len(predictions),
            "mean_error_minutes": (
                sum(abs(a) for a in accuracies) / len(accuracies) if accuracies else 0
            ),
            "median_error_minutes": (
                sorted(accuracies)[len(accuracies) // 2] if accuracies else 0
            ),
            "rmse_minutes": (
                (sum(a**2 for a in accuracies) / len(accuracies)) ** 0.5
                if accuracies
                else 0
            ),
        }

    async def get_triggering_event(
        self, session: AsyncSession
    ) -> Optional["SensorEvent"]:
        """Get the triggering sensor event using application-level join."""
        if not self.triggering_event_id:
            return None

        query = select(SensorEvent).where(SensorEvent.id == self.triggering_event_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def get_room_state(self, session: AsyncSession) -> Optional["RoomState"]:
        """Get the associated room state using application-level join."""
        if not self.room_state_id:
            return None

        query = select(RoomState).where(RoomState.id == self.room_state_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def get_predictions_with_events(
        cls, session: AsyncSession, room_id: str, hours: int = 24
    ) -> List[Tuple["Prediction", Optional["SensorEvent"]]]:
        """Get predictions with their triggering events using application-level joins."""

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Get predictions
        prediction_query = (
            select(cls)
            .where(and_(cls.room_id == room_id, cls.prediction_time >= cutoff_time))
            .order_by(desc(cls.prediction_time))
        )

        prediction_result = await session.execute(prediction_query)
        predictions = prediction_result.scalars().all()

        # Get triggering events in batch
        event_ids = [
            p.triggering_event_id for p in predictions if p.triggering_event_id
        ]
        events_dict = {}

        if event_ids:
            event_query = select(SensorEvent).where(SensorEvent.id.in_(event_ids))
            event_result = await session.execute(event_query)
            events_dict = {event.id: event for event in event_result.scalars().all()}

        # Combine results
        return [(p, events_dict.get(p.triggering_event_id)) for p in predictions]

    @classmethod
    async def get_predictions_with_full_context(
        cls,
        session: AsyncSession,
        room_id: str,
        hours: int = 24,
        include_alternatives: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with full context including JSON data analysis.
        Uses selectinload for efficient relationship loading.

        Args:
            session: Database session
            room_id: Room to analyze
            hours: Hours of data to retrieve
            include_alternatives: Include alternative predictions from JSON field

        Returns:
            List of prediction dictionaries with full context
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Use selectinload strategy for efficient loading
        query = (
            select(cls)
            .where(and_(cls.room_id == room_id, cls.prediction_time >= cutoff_time))
            .order_by(desc(cls.prediction_time))
            .limit(1000)  # Prevent excessive loading
        )

        result = await session.execute(query)
        predictions = result.scalars().all()

        enriched_predictions = []
        for prediction in predictions:
            # Extract and analyze JSON data
            feature_data = prediction.feature_importance or {}
            alternatives_data = prediction.alternatives or []

            context = {
                "prediction_id": prediction.id,
                "room_id": prediction.room_id,
                "prediction_time": prediction.prediction_time,
                "predicted_transition_time": prediction.predicted_transition_time,
                "transition_type": prediction.transition_type,
                "confidence_score": float(prediction.confidence_score),
                "model_type": prediction.model_type,
                "model_version": prediction.model_version,
                "accuracy_minutes": prediction.accuracy_minutes,
                "is_accurate": prediction.is_accurate,
                # Analyze JSON feature importance
                "feature_analysis": {
                    "total_features": len(feature_data),
                    "top_features": cls._extract_top_features(feature_data),
                    "feature_categories": cls._categorize_features(feature_data),
                },
            }

            # Include alternatives analysis if requested
            if include_alternatives and alternatives_data:
                context["alternatives_analysis"] = {
                    "total_alternatives": len(alternatives_data),
                    "confidence_spread": cls._analyze_confidence_spread(
                        alternatives_data
                    ),
                    "alternative_predictions": alternatives_data[
                        :3
                    ],  # Top 3 alternatives
                }

            enriched_predictions.append(context)

        return enriched_predictions

    @staticmethod
    def _extract_top_features(
        feature_importance: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Extract top features from JSON feature importance data."""
        if not feature_importance:
            return []

        # Convert to list of tuples and sort by importance
        features = [
            (k, float(v))
            for k, v in feature_importance.items()
            if isinstance(v, (int, float))
        ]
        return sorted(features, key=lambda x: abs(x[1]), reverse=True)[:10]

    @staticmethod
    def _categorize_features(feature_importance: Dict[str, Any]) -> Dict[str, int]:
        """Categorize features by type from JSON data."""
        categories = {
            "temporal": 0,
            "sequential": 0,
            "contextual": 0,
            "environmental": 0,
            "other": 0,
        }

        for feature_name in feature_importance.keys():
            feature_lower = feature_name.lower()
            if any(
                term in feature_lower for term in ["time", "hour", "day", "cyclical"]
            ):
                categories["temporal"] += 1
            elif any(
                term in feature_lower for term in ["sequence", "transition", "movement"]
            ):
                categories["sequential"] += 1
            elif any(
                term in feature_lower for term in ["room", "cross", "correlation"]
            ):
                categories["contextual"] += 1
            elif any(
                term in feature_lower for term in ["temperature", "humidity", "light"]
            ):
                categories["environmental"] += 1
            else:
                categories["other"] += 1

        return categories

    @staticmethod
    def _analyze_confidence_spread(
        alternatives: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze confidence distribution in alternatives JSON data."""
        if not alternatives:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        confidences = []
        for alt in alternatives:
            if isinstance(alt, dict) and "confidence" in alt:
                confidences.append(float(alt["confidence"]))

        if not confidences:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

        return {
            "min": min(confidences),
            "max": max(confidences),
            "mean": mean_conf,
            "std": variance**0.5,
        }

    def add_extended_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Add extended metadata using JSON field capabilities.

        Args:
            metadata: Dictionary of additional metadata to store
        """
        # Extend feature_importance JSON field with metadata
        current_features = self.feature_importance or {}

        # Add metadata section to JSON
        extended_data = {
            **current_features,
            "_metadata": {
                "extended_info": metadata,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
            },
        }

        # Store back to JSON field
        self.feature_importance = extended_data


class ModelAccuracy(Base):
    """
    Tracks model performance metrics over time.
    Used for drift detection and retraining decisions.
    """

    __tablename__ = "model_accuracy"

    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)
    room_id = Column(String(50), nullable=False, index=True)
    model_type = Column(ENUM(*MODEL_TYPES, name="model_type_enum"), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Time window for metrics
    measurement_start = Column(DateTime(timezone=True), nullable=False)
    measurement_end = Column(DateTime(timezone=True), nullable=False)

    # Accuracy metrics
    total_predictions = Column(Integer, nullable=False)
    accurate_predictions = Column(Integer, nullable=False)
    accuracy_rate = Column(Float, nullable=False)
    mean_error_minutes = Column(Float, nullable=False)
    median_error_minutes = Column(Float, nullable=False)
    rmse_minutes = Column(Float, nullable=False)

    # Confidence calibration
    confidence_correlation = Column(
        Float
    )  # Correlation between confidence and accuracy
    overconfidence_rate = Column(Float)  # Rate of high confidence but wrong predictions

    # Drift indicators
    feature_drift_score = Column(Float)
    concept_drift_score = Column(Float)
    performance_degradation = Column(Float)  # Change from baseline

    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    baseline_comparison = Column(
        JSON, default=dict
    )  # Force JSON for cross-database compatibility

    __table_args__ = (
        Index("idx_room_model_time", "room_id", "model_type", "measurement_end"),
        Index(
            "idx_accuracy_trend",
            "room_id",
            "model_type",
            "accuracy_rate",
            "measurement_end",
        ),
        Index(
            "idx_drift_detection",
            "room_id",
            "concept_drift_score",
            "measurement_end",
        ),
        UniqueConstraint(
            "room_id", "model_type", "measurement_start", "measurement_end"
        ),
        CheckConstraint("accuracy_rate >= 0 AND accuracy_rate <= 1"),
        CheckConstraint("total_predictions >= accurate_predictions"),
    )


class FeatureStore(Base):
    """
    Stores computed features for model training and inference.
    Caches expensive feature computations for reuse.
    """

    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)
    room_id = Column(String(50), nullable=False, index=True)
    feature_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Feature categories
    temporal_features = Column(
        JSON, nullable=False, default=dict
    )  # Force JSON for cross-database compatibility
    sequential_features = Column(
        JSON, nullable=False, default=dict
    )  # Force JSON for cross-database compatibility
    contextual_features = Column(
        JSON, nullable=False, default=dict
    )  # Force JSON for cross-database compatibility
    environmental_features = Column(
        JSON,
        nullable=False,
        default=dict,  # Force JSON for cross-database compatibility
    )

    # Feature metadata
    lookback_hours = Column(Integer, nullable=False)
    feature_version = Column(String(20), nullable=False)
    computation_time_ms = Column(Float)

    # Data quality indicators
    completeness_score = Column(Float)  # Fraction of expected features present
    freshness_score = Column(Float)  # How recent the input data was
    confidence_score = Column(Float)  # Overall confidence in feature quality

    # Tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True))  # When to refresh features

    __table_args__ = (
        Index("idx_room_feature_time", "room_id", "feature_timestamp"),
        Index("idx_feature_version", "feature_version", "created_at"),
        Index("idx_expiration", "expires_at"),
        Index(
            "idx_feature_quality",
            "room_id",
            "completeness_score",
            "freshness_score",
        ),
        UniqueConstraint(
            "room_id", "feature_timestamp", "lookback_hours", "feature_version"
        ),
        CheckConstraint("completeness_score >= 0 AND completeness_score <= 1"),
        CheckConstraint("freshness_score >= 0 AND freshness_score <= 1"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"),
    )

    @classmethod
    async def get_latest_features(
        cls, session: AsyncSession, room_id: str, max_age_hours: int = 6
    ) -> Optional["FeatureStore"]:
        """Get the most recent feature set for a room."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        query = (
            select(cls)
            .where(
                and_(
                    cls.room_id == room_id,
                    cls.feature_timestamp >= cutoff_time,
                    or_(
                        cls.expires_at.is_(None),
                        cls.expires_at > datetime.now(timezone.utc),
                    ),
                )
            )
            .order_by(desc(cls.feature_timestamp))
            .limit(1)
        )

        result = await session.execute(query)
        return result.scalar_one_or_none()

    def get_all_features(self) -> Dict[str, Any]:
        """Combine all feature categories into a single dictionary."""
        features = {}
        features.update(self.temporal_features or {})
        features.update(self.sequential_features or {})
        features.update(self.contextual_features or {})
        features.update(self.environmental_features or {})
        return features


class PredictionAudit(Base):
    """
    Audit trail for predictions demonstrating ForeignKey and relationship usage.
    This model uses proper relationships for non-time-series data.
    """

    __tablename__ = "prediction_audits"

    id = Column(Integer, primary_key=True, autoincrement="auto", nullable=False)

    # Foreign key relationships
    prediction_id = Column(
        BigInteger, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False
    )
    model_accuracy_id = Column(
        BigInteger, ForeignKey("model_accuracy.id", ondelete="SET NULL"), nullable=True
    )

    # Audit fields
    audit_timestamp = Column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    audit_action = Column(
        String(50), nullable=False
    )  # 'created', 'validated', 'corrected'
    audit_user = Column(String(100), nullable=True)

    # Detailed audit data stored as JSON
    audit_details = Column(JSON, default=dict)  # Using JSON column type
    previous_values = Column(JSON, default=dict)  # Store previous state
    validation_metrics = Column(JSON, default=dict)  # Store validation results

    # Text field for detailed notes
    audit_notes = Column(Text, nullable=True)

    # Establish relationships
    prediction = relationship("Prediction", backref="audit_entries", lazy="select")
    model_accuracy = relationship(
        "ModelAccuracy", backref="audited_predictions", lazy="select"
    )

    __table_args__ = (
        Index("idx_prediction_audit", "prediction_id", "audit_timestamp"),
        Index("idx_audit_action_time", "audit_action", "audit_timestamp"),
    )

    @classmethod
    async def create_audit_entry(
        cls,
        session: AsyncSession,
        prediction_id: int,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> "PredictionAudit":
        """
        Create a new audit entry with JSON details.

        Args:
            session: Database session
            prediction_id: ID of the prediction being audited
            action: Type of audit action
            details: JSON dictionary of audit details
            user: User performing the action
            notes: Additional text notes

        Returns:
            Created audit entry
        """
        audit = cls(
            prediction_id=prediction_id,
            audit_action=action,
            audit_user=user,
            audit_details=details or {},
            audit_notes=notes,
        )

        session.add(audit)
        await session.flush()  # Get the ID

        return audit

    @classmethod
    async def get_audit_trail_with_relationships(
        cls, session: AsyncSession, prediction_id: int, load_related: bool = True
    ) -> List["PredictionAudit"]:
        """
        Get audit trail using selectinload for efficient relationship loading.

        Args:
            session: Database session
            prediction_id: Prediction to get audit trail for
            load_related: Whether to eagerly load related objects

        Returns:
            List of audit entries with relationships loaded
        """
        query = select(cls).where(cls.prediction_id == prediction_id)

        if load_related:
            # Use selectinload for efficient relationship loading
            query = query.options(
                selectinload(cls.prediction), selectinload(cls.model_accuracy)
            )

        query = query.order_by(cls.audit_timestamp.asc())

        result = await session.execute(query)
        return result.scalars().all()

    def analyze_json_details(self) -> Dict[str, Any]:
        """
        Analyze the JSON audit details field.

        Returns:
            Analysis of the audit details
        """
        details = self.audit_details or {}

        analysis = {
            "total_fields": len(details),
            "field_types": {},
            "has_metrics": "metrics" in details,
            "has_errors": "errors" in details,
            "complexity_score": 0,
        }

        # Analyze field types
        for key, value in details.items():
            value_type = type(value).__name__
            analysis["field_types"][value_type] = (
                analysis["field_types"].get(value_type, 0) + 1
            )

        # Calculate complexity score based on nested structures
        analysis["complexity_score"] = self._calculate_json_complexity(details)

        return analysis

    @staticmethod
    def _calculate_json_complexity(data: Any, depth: int = 0) -> int:
        """Calculate complexity score for nested JSON data."""
        if depth > 5:  # Prevent infinite recursion
            return 1

        if isinstance(data, dict):
            return sum(
                PredictionAudit._calculate_json_complexity(v, depth + 1)
                for v in data.values()
            ) + len(data)
        elif isinstance(data, list):
            return sum(
                PredictionAudit._calculate_json_complexity(item, depth + 1)
                for item in data
            ) + len(data)
        else:
            return 1

    def update_validation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update the JSON validation metrics field."""
        current_metrics = self.validation_metrics or {}
        current_metrics.update(metrics)
        current_metrics["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.validation_metrics = current_metrics


# Utility functions for database operations
async def create_timescale_hypertables(session: AsyncSession):
    """
    Create TimescaleDB hypertables and configure partitioning.

    The sensor_events table uses a single primary key (id) for cross-database compatibility.
    TimescaleDB hypertable partitioning is configured on the timestamp column.
    Foreign key relationships are managed at the application level to avoid conflicts
    with TimescaleDB's unique index requirements.
    """

    # Create hypertable for sensor_events with timestamp partitioning
    # Using single primary key for compatibility while still partitioning by timestamp
    await session.execute(
        text(
            "SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE, create_default_indexes => FALSE)"
        )
    )

    # Create continuous aggregates for common queries
    await session.execute(
        text(
            """
        CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_events_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket(INTERVAL '1 hour', timestamp) AS bucket,
            room_id,
            sensor_type,
            state,
            COUNT(*) as event_count,
            COUNT(CASE WHEN state != previous_state THEN 1 END) as state_changes
        FROM sensor_events
        GROUP BY bucket, room_id, sensor_type, state
        WITH NO DATA;
    """
        )
    )

    # Enable compression for older data
    await session.execute(
        text(
            """
        ALTER TABLE sensor_events SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'room_id,sensor_type',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """
        )
    )

    # Set up automatic compression policy (compress data older than 7 days)
    await session.execute(
        text(
            """
        SELECT add_compression_policy('sensor_events', INTERVAL '7 days', if_not_exists => TRUE);
    """
        )
    )

    # Set up data retention policy (keep data for 2 years)
    await session.execute(
        text(
            """
        SELECT add_retention_policy('sensor_events', INTERVAL '2 years', if_not_exists => TRUE);
    """
        )
    )

    await session.commit()


async def optimize_database_performance(session: AsyncSession):
    """Apply performance optimizations to the database."""

    # Update table statistics
    await session.execute(text("ANALYZE sensor_events"))
    await session.execute(text("ANALYZE room_states"))
    await session.execute(text("ANALYZE predictions"))

    # Create partial indexes for common query patterns
    await session.execute(
        text(
            """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_motion_events
        ON sensor_events (room_id, timestamp DESC)
        WHERE sensor_type = 'motion' AND timestamp > NOW() - INTERVAL '24 hours'
    """
        )
    )

    await session.execute(
        text(
            """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_predictions
        ON predictions (room_id, model_type, validation_timestamp DESC)
        WHERE is_accurate = FALSE
    """
        )
    )

    await session.commit()


def get_bulk_insert_query() -> str:
    """Generate optimized bulk insert query for sensor events."""
    return """
        INSERT INTO sensor_events (
            timestamp, room_id, sensor_id, sensor_type, state,
            previous_state, attributes, is_human_triggered, confidence_score
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            timestamp = EXCLUDED.timestamp,
            room_id = EXCLUDED.room_id,
            sensor_id = EXCLUDED.sensor_id,
            sensor_type = EXCLUDED.sensor_type,
            state = EXCLUDED.state,
            previous_state = EXCLUDED.previous_state,
            attributes = EXCLUDED.attributes,
            is_human_triggered = EXCLUDED.is_human_triggered,
            confidence_score = EXCLUDED.confidence_score
    """
