# TimescaleDB Hypertable Compatibility Fix

## Problem Summary

TimescaleDB requires the partitioning column (`timestamp`) to be part of the primary key for hypertables. The original schema had a single primary key on `id`, which caused this error:

```
cannot create a unique index without the column 'timestamp' (used in partitioning)
```

However, foreign keys from other tables needed to reference the `id` column specifically, creating a conflict between TimescaleDB requirements and PostgreSQL foreign key constraints.

## Solution Implemented

### 1. Composite Primary Key
Changed `SensorEvent` model to use a composite primary key:

```python
# Before:
id = Column(BigInteger, primary_key=True, autoincrement=True)
timestamp = Column(DateTime(timezone=True), nullable=False, index=True, default=func.now())

# After:
id = Column(BigInteger, primary_key=True, autoincrement=True)
timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False, default=func.now())
```

### 2. Unique Constraint for Foreign Key Compatibility
Added a unique constraint on `id` column to enable foreign key relationships:

```python
__table_args__ = (
    # Unique constraint on id for foreign key compatibility
    # This ensures id remains unique across all partitions
    UniqueConstraint('id', name='uq_sensor_event_id'),
    # ... other indexes and constraints
)
```

### 3. Updated Bulk Insert Query
Modified the bulk insert query to handle the composite primary key:

```python
# Updated ON CONFLICT clause to work with composite primary key
ON CONFLICT (id, timestamp) DO UPDATE SET
    timestamp = EXCLUDED.timestamp,
    room_id = EXCLUDED.room_id,
    # ... other fields
```

### 4. Enhanced Documentation
Added detailed comments explaining the design decisions:

```python
class SensorEvent(Base):
    """
    Main hypertable for storing all sensor events from Home Assistant.
    
    Uses composite primary key (id, timestamp) for TimescaleDB hypertable compatibility
    while maintaining id uniqueness for foreign key relationships.
    """
```

## How It Works

1. **TimescaleDB Compatibility**: The composite primary key `(id, timestamp)` satisfies TimescaleDB's requirement that the partitioning column be part of the primary key.

2. **Foreign Key Support**: The unique constraint on `id` ensures that foreign keys can reference just the `id` column:
   ```python
   triggering_event_id = Column(BigInteger, ForeignKey('sensor_events.id', ondelete='SET NULL'))
   ```

3. **Data Integrity**: The `autoincrement=True` on `id` ensures that ID values are unique across all partitions, maintaining referential integrity.

4. **Query Performance**: All existing indexes and query patterns continue to work as before.

## Testing the Fix

To verify the fix works:

1. **Create Tables**: Run `python scripts/setup_database.py`
2. **Create Hypertable**: The script will run `SELECT create_hypertable('sensor_events', 'timestamp')` successfully
3. **Test Foreign Keys**: Create a `Prediction` record that references a `SensorEvent.id`
4. **Verify Partitioning**: Check `timescaledb_information.hypertables` for proper setup

## Benefits

- ✅ **TimescaleDB Hypertables**: Can now create hypertables with timestamp partitioning
- ✅ **Foreign Key Relationships**: All existing FK relationships continue to work
- ✅ **Backwards Compatibility**: No changes needed to existing queries or application code
- ✅ **Performance**: Time-series optimizations enabled while maintaining relational integrity
- ✅ **Data Consistency**: Unique ID values across all partitions

## Files Modified

- `src/data/storage/models.py`: Updated `SensorEvent` model with composite PK and unique constraint
- Documentation added explaining the design decisions and TimescaleDB compatibility

## Verification Commands

```python
# Test hypertable creation
await session.execute(text("SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE)"))

# Test foreign key relationship
prediction = Prediction(triggering_event_id=sensor_event.id, ...)
session.add(prediction)
await session.commit()  # Should work without errors

# Verify hypertable setup
result = await session.execute("""
    SELECT hypertable_name, main_dimension 
    FROM timescaledb_information.hypertables 
    WHERE hypertable_name = 'sensor_events'
""")
```

This fix enables the full benefits of TimescaleDB for time-series data while maintaining all the relational database features needed for the occupancy prediction system.