"""
Database models for the occupancy prediction system.

This module defines SQLAlchemy models optimized for TimescaleDB to handle
time-series sensor data, predictions, and model performance tracking.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, Index, Integer, 
    String, Text, JSON, ForeignKey, UniqueConstraint, CheckConstraint,
    func, text, select, and_, or_, desc
)
from sqlalchemy.dialects.postgresql import JSONB, ENUM, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, selectinload
from sqlalchemy.sql import func as sql_func

Base = declarative_base()

# Enums for consistent state tracking
SENSOR_TYPES = [
    'motion', 'presence', 'door', 'window', 'temperature', 
    'humidity', 'light', 'pressure', 'air_quality'
]

SENSOR_STATES = ['on', 'off', 'open', 'closed', 'detected', 'clear', 'unknown']

TRANSITION_TYPES = ['occupied_to_vacant', 'vacant_to_occupied', 'state_change']

MODEL_TYPES = ['lstm', 'xgboost', 'hmm', 'gaussian_process', 'ensemble']


class SensorEvent(Base):
    """
    Main hypertable for storing all sensor events from Home Assistant.
    Optimized for time-series queries with proper partitioning.
    
    Uses composite primary key (id, timestamp) for TimescaleDB hypertable compatibility
    while maintaining id uniqueness for foreign key relationships.
    """
    __tablename__ = 'sensor_events'
    
    # Composite primary key for TimescaleDB hypertable compatibility
    # id remains unique across the table via UniqueConstraint for foreign key compatibility
    # autoincrement=True ensures id values are unique across all partitions
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False, default=func.now())
    
    # Core event data
    room_id = Column(String(50), nullable=False, index=True)
    sensor_id = Column(String(100), nullable=False, index=True)
    sensor_type = Column(ENUM(*SENSOR_TYPES, name='sensor_type_enum'), nullable=False)
    state = Column(ENUM(*SENSOR_STATES, name='sensor_state_enum'), nullable=False)
    previous_state = Column(ENUM(*SENSOR_STATES, name='sensor_state_enum'))
    
    # Metadata and context
    attributes = Column(JSONB, default=dict)  # Additional sensor attributes
    is_human_triggered = Column(Boolean, default=True, nullable=False)
    confidence_score = Column(Float)  # Confidence in human vs automation classification
    
    # Performance tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="triggering_event")
    
    __table_args__ = (
        # Unique constraint on id for foreign key compatibility
        # This ensures id remains unique across all partitions
        UniqueConstraint('id', name='uq_sensor_event_id'),
        
        # Indexes for efficient time-series queries
        Index('idx_room_sensor_time', 'room_id', 'sensor_id', 'timestamp'),
        Index('idx_room_time_desc', 'room_id', desc('timestamp')),
        Index('idx_state_changes', 'room_id', 'timestamp', 
              postgresql_where=text("state != previous_state")),
        Index('idx_sensor_type_time', 'sensor_type', 'timestamp'),
        Index('idx_human_triggered', 'is_human_triggered', 'timestamp'),
        
        # Composite indexes for common query patterns
        Index('idx_room_sensor_state_time', 'room_id', 'sensor_id', 'state', 'timestamp'),
        Index('idx_motion_events', 'room_id', 'timestamp', 
              postgresql_where=text("sensor_type = 'motion'")),
        
        # Constraints
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', 
                       name='valid_confidence'),
    )
    
    @classmethod
    async def get_recent_events(
        cls, 
        session: AsyncSession, 
        room_id: str, 
        hours: int = 24,
        sensor_types: Optional[List[str]] = None
    ) -> List['SensorEvent']:
        """Get recent events for a room, optionally filtered by sensor type."""
        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.timestamp >= datetime.utcnow() - timedelta(hours=hours)
            )
        ).order_by(desc(cls.timestamp))
        
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
        end_time: Optional[datetime] = None
    ) -> List['SensorEvent']:
        """Get events where state changed from previous state."""
        if end_time is None:
            end_time = datetime.utcnow()
            
        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.timestamp >= start_time,
                cls.timestamp <= end_time,
                cls.state != cls.previous_state
            )
        ).order_by(cls.timestamp)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def get_transition_sequences(
        cls,
        session: AsyncSession,
        room_id: str,
        lookback_hours: int = 24,
        min_sequence_length: int = 3
    ) -> List[List['SensorEvent']]:
        """Get sequences of sensor transitions for pattern analysis."""
        events = await cls.get_state_changes(
            session, room_id, datetime.utcnow() - timedelta(hours=lookback_hours)
        )
        
        # Group events into sequences based on time gaps
        sequences = []
        current_sequence = []
        max_gap_minutes = 30
        
        for event in events:
            if (current_sequence and 
                (event.timestamp - current_sequence[-1].timestamp).total_seconds() > max_gap_minutes * 60):
                if len(current_sequence) >= min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [event]
            else:
                current_sequence.append(event)
        
        if len(current_sequence) >= min_sequence_length:
            sequences.append(current_sequence)
            
        return sequences


class RoomState(Base):
    """
    Current and historical room occupancy states.
    Tracks the derived occupancy status from sensor events.
    """
    __tablename__ = 'room_states'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    room_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Occupancy status
    is_occupied = Column(Boolean, nullable=False)
    occupancy_confidence = Column(Float, nullable=False, default=0.5)
    occupant_type = Column(String(20))  # 'human', 'cat', 'unknown'
    occupant_count = Column(Integer, default=1)
    
    # State metadata
    state_duration = Column(Integer)  # Duration in current state (seconds)
    transition_trigger = Column(String(100))  # Sensor that triggered transition
    certainty_factors = Column(JSONB, default=dict)  # Contributing factors
    
    # Tracking
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="room_state")
    
    __table_args__ = (
        Index('idx_room_time_occupied', 'room_id', 'timestamp', 'is_occupied'),
        Index('idx_occupancy_changes', 'room_id', 'timestamp', 
              postgresql_where=text("transition_trigger IS NOT NULL")),
        Index('idx_recent_states', 'room_id', desc('timestamp')),
        CheckConstraint('occupancy_confidence >= 0 AND occupancy_confidence <= 1'),
        CheckConstraint('occupant_count >= 0'),
    )
    
    @classmethod
    async def get_current_state(cls, session: AsyncSession, room_id: str) -> Optional['RoomState']:
        """Get the most recent room state."""
        query = select(cls).where(cls.room_id == room_id).order_by(desc(cls.timestamp)).limit(1)
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_occupancy_history(
        cls,
        session: AsyncSession,
        room_id: str,
        hours: int = 24
    ) -> List['RoomState']:
        """Get occupancy history for pattern analysis."""
        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.timestamp >= datetime.utcnow() - timedelta(hours=hours)
            )
        ).order_by(cls.timestamp)
        
        result = await session.execute(query)
        return result.scalars().all()


class Prediction(Base):
    """
    Stores model predictions for occupancy transitions.
    Tracks prediction accuracy and model performance.
    """
    __tablename__ = 'predictions'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    room_id = Column(String(50), nullable=False, index=True)
    prediction_time = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Prediction details
    predicted_transition_time = Column(DateTime(timezone=True), nullable=False)
    transition_type = Column(ENUM(*TRANSITION_TYPES, name='transition_type_enum'), nullable=False)
    confidence_score = Column(Float, nullable=False)
    prediction_interval_lower = Column(DateTime(timezone=True))
    prediction_interval_upper = Column(DateTime(timezone=True))
    
    # Model information
    model_type = Column(ENUM(*MODEL_TYPES, name='model_type_enum'), nullable=False)
    model_version = Column(String(50), nullable=False)
    feature_importance = Column(JSONB, default=dict)
    
    # Alternative predictions (top-k)
    alternatives = Column(JSONB, default=list)
    
    # Validation results
    actual_transition_time = Column(DateTime(timezone=True))
    accuracy_minutes = Column(Float)  # Difference in minutes
    is_accurate = Column(Boolean)  # Within threshold
    validation_timestamp = Column(DateTime(timezone=True))
    
    # Context - Foreign keys with proper nullable constraints
    # Note: References sensor_events.id which has a unique constraint for FK compatibility
    triggering_event_id = Column(BigInteger, ForeignKey('sensor_events.id', ondelete='SET NULL'), nullable=True)
    room_state_id = Column(BigInteger, ForeignKey('room_states.id', ondelete='SET NULL'), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    processing_time_ms = Column(Float)  # Time to generate prediction
    
    # Relationships
    triggering_event = relationship("SensorEvent", back_populates="predictions")
    room_state = relationship("RoomState", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_room_prediction_time', 'room_id', 'prediction_time'),
        Index('idx_model_type_time', 'model_type', 'prediction_time'),
        Index('idx_accuracy_validation', 'room_id', 'is_accurate', 'validation_timestamp'),
        Index('idx_pending_validation', 'room_id', 'predicted_transition_time',
              postgresql_where=text("actual_transition_time IS NULL")),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
        CheckConstraint('prediction_interval_lower <= prediction_interval_upper'),
    )
    
    @classmethod
    async def get_pending_validations(
        cls,
        session: AsyncSession,
        room_id: Optional[str] = None,
        cutoff_hours: int = 2
    ) -> List['Prediction']:
        """Get predictions that need validation (past their predicted time)."""
        cutoff_time = datetime.utcnow() - timedelta(hours=cutoff_hours)
        
        query = select(cls).where(
            and_(
                cls.predicted_transition_time <= datetime.utcnow(),
                cls.predicted_transition_time >= cutoff_time,
                cls.actual_transition_time.is_(None)
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
        model_type: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate accuracy metrics for predictions."""
        start_time = datetime.utcnow() - timedelta(days=days)
        
        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.validation_timestamp >= start_time,
                cls.actual_transition_time.is_not(None)
            )
        )
        
        if model_type:
            query = query.where(cls.model_type == model_type)
            
        result = await session.execute(query)
        predictions = result.scalars().all()
        
        if not predictions:
            return {}
        
        accuracies = [p.accuracy_minutes for p in predictions if p.accuracy_minutes is not None]
        accurate_count = sum(1 for p in predictions if p.is_accurate)
        
        return {
            'total_predictions': len(predictions),
            'accurate_predictions': accurate_count,
            'accuracy_rate': accurate_count / len(predictions),
            'mean_error_minutes': sum(abs(a) for a in accuracies) / len(accuracies) if accuracies else 0,
            'median_error_minutes': sorted(accuracies)[len(accuracies)//2] if accuracies else 0,
            'rmse_minutes': (sum(a**2 for a in accuracies) / len(accuracies))**0.5 if accuracies else 0,
        }


class ModelAccuracy(Base):
    """
    Tracks model performance metrics over time.
    Used for drift detection and retraining decisions.
    """
    __tablename__ = 'model_accuracy'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    room_id = Column(String(50), nullable=False, index=True)
    model_type = Column(ENUM(*MODEL_TYPES, name='model_type_enum'), nullable=False)
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
    confidence_correlation = Column(Float)  # Correlation between confidence and accuracy
    overconfidence_rate = Column(Float)  # Rate of high confidence but wrong predictions
    
    # Drift indicators
    feature_drift_score = Column(Float)
    concept_drift_score = Column(Float)
    performance_degradation = Column(Float)  # Change from baseline
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    baseline_comparison = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_room_model_time', 'room_id', 'model_type', 'measurement_end'),
        Index('idx_accuracy_trend', 'room_id', 'model_type', 'accuracy_rate', 'measurement_end'),
        Index('idx_drift_detection', 'room_id', 'concept_drift_score', 'measurement_end'),
        UniqueConstraint('room_id', 'model_type', 'measurement_start', 'measurement_end'),
        CheckConstraint('accuracy_rate >= 0 AND accuracy_rate <= 1'),
        CheckConstraint('total_predictions >= accurate_predictions'),
    )


class FeatureStore(Base):
    """
    Stores computed features for model training and inference.
    Caches expensive feature computations for reuse.
    """
    __tablename__ = 'feature_store'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    room_id = Column(String(50), nullable=False, index=True)
    feature_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Feature categories
    temporal_features = Column(JSONB, nullable=False, default=dict)
    sequential_features = Column(JSONB, nullable=False, default=dict)
    contextual_features = Column(JSONB, nullable=False, default=dict)
    environmental_features = Column(JSONB, nullable=False, default=dict)
    
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
        Index('idx_room_feature_time', 'room_id', 'feature_timestamp'),
        Index('idx_feature_version', 'feature_version', 'created_at'),
        Index('idx_expiration', 'expires_at'),
        Index('idx_feature_quality', 'room_id', 'completeness_score', 'freshness_score'),
        UniqueConstraint('room_id', 'feature_timestamp', 'lookback_hours', 'feature_version'),
        CheckConstraint('completeness_score >= 0 AND completeness_score <= 1'),
        CheckConstraint('freshness_score >= 0 AND freshness_score <= 1'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
    )
    
    @classmethod
    async def get_latest_features(
        cls,
        session: AsyncSession,
        room_id: str,
        max_age_hours: int = 6
    ) -> Optional['FeatureStore']:
        """Get the most recent feature set for a room."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        query = select(cls).where(
            and_(
                cls.room_id == room_id,
                cls.feature_timestamp >= cutoff_time,
                or_(cls.expires_at.is_(None), cls.expires_at > datetime.utcnow())
            )
        ).order_by(desc(cls.feature_timestamp)).limit(1)
        
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


# Utility functions for database operations
async def create_timescale_hypertables(session: AsyncSession):
    """
    Create TimescaleDB hypertables and configure partitioning.
    
    The sensor_events table uses a composite primary key (id, timestamp) to satisfy
    TimescaleDB's requirement that the partitioning column be part of the primary key.
    A separate unique constraint on 'id' ensures foreign key relationships work properly.
    """
    
    # Create hypertable for sensor_events with timestamp partitioning
    # The composite primary key (id, timestamp) allows this to work
    await session.execute(
        text("SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE, create_default_indexes => FALSE)")
    )
    
    # Create continuous aggregates for common queries
    await session.execute(text("""
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
    """))
    
    # Enable compression for older data
    await session.execute(text("""
        ALTER TABLE sensor_events SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'room_id,sensor_type',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """))
    
    # Set up automatic compression policy (compress data older than 7 days)
    await session.execute(text("""
        SELECT add_compression_policy('sensor_events', INTERVAL '7 days', if_not_exists => TRUE);
    """))
    
    # Set up data retention policy (keep data for 2 years)
    await session.execute(text("""
        SELECT add_retention_policy('sensor_events', INTERVAL '2 years', if_not_exists => TRUE);
    """))
    
    await session.commit()


async def optimize_database_performance(session: AsyncSession):
    """Apply performance optimizations to the database."""
    
    # Update table statistics
    await session.execute(text("ANALYZE sensor_events"))
    await session.execute(text("ANALYZE room_states"))
    await session.execute(text("ANALYZE predictions"))
    
    # Create partial indexes for common query patterns
    await session.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_motion_events 
        ON sensor_events (room_id, timestamp DESC) 
        WHERE sensor_type = 'motion' AND timestamp > NOW() - INTERVAL '24 hours'
    """))
    
    await session.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_predictions
        ON predictions (room_id, model_type, validation_timestamp DESC)
        WHERE is_accurate = FALSE
    """))
    
    await session.commit()


def get_bulk_insert_query() -> str:
    """Generate optimized bulk insert query for sensor events."""
    return """
        INSERT INTO sensor_events (
            timestamp, room_id, sensor_id, sensor_type, state, 
            previous_state, attributes, is_human_triggered, confidence_score
        ) VALUES %s
        ON CONFLICT (id, timestamp) DO UPDATE SET
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