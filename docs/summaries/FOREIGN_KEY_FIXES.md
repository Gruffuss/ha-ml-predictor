# Foreign Key Constraint Fixes

## Issues Identified and Fixed

### 1. **Primary Key Structure Issues**

**Problem**: The original `SensorEvent` table had a composite primary key `(id, timestamp)`, but foreign keys in the `Prediction` table only referenced `sensor_events.id`. PostgreSQL requires foreign keys to match the complete primary key structure.

**Fix**: 
- Changed `SensorEvent` to use a simple primary key on `id` only
- Kept `timestamp` as a regular indexed column for TimescaleDB partitioning
- Removed redundant `unique=True` constraint on primary key columns

### 2. **Missing Unique Constraints for Foreign Key References**

**Problem**: PostgreSQL foreign keys require the referenced columns to have unique constraints, but the original models didn't explicitly ensure this.

**Fix**:
- Primary key columns are inherently unique, so explicit unique constraints were unnecessary
- Removed redundant `UniqueConstraint` declarations that were conflicting with primary key definitions

### 3. **Foreign Key Nullability and Cascade Options**

**Problem**: Foreign key columns weren't properly configured for optional relationships and cascading deletes.

**Fix**:
- Made `triggering_event_id` and `room_state_id` nullable (`nullable=True`)
- Added `ondelete='SET NULL'` to foreign key constraints for proper cascade behavior
- This allows predictions to exist even if their associated events or room states are deleted

## Model Structure After Fixes

### SensorEvent Table
```sql
CREATE TABLE sensor_events (
    id BIGSERIAL PRIMARY KEY,  -- Simple primary key, inherently unique
    timestamp TIMESTAMPTZ NOT NULL,  -- Regular column for TimescaleDB partitioning
    room_id VARCHAR(50) NOT NULL,
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type sensor_type_enum NOT NULL,
    state sensor_state_enum NOT NULL,
    previous_state sensor_state_enum,
    attributes JSONB DEFAULT '{}',
    is_human_triggered BOOLEAN DEFAULT TRUE NOT NULL,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);
```

### RoomState Table
```sql
CREATE TABLE room_states (
    id BIGSERIAL PRIMARY KEY,  -- Simple primary key, inherently unique
    room_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    is_occupied BOOLEAN NOT NULL,
    occupancy_confidence FLOAT DEFAULT 0.5 NOT NULL,
    occupant_type VARCHAR(20),
    occupant_count INTEGER DEFAULT 1,
    state_duration INTEGER,
    transition_trigger VARCHAR(100),
    certainty_factors JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);
```

### Prediction Table with Fixed Foreign Keys
```sql
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    room_id VARCHAR(50) NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    predicted_transition_time TIMESTAMPTZ NOT NULL,
    transition_type transition_type_enum NOT NULL,
    confidence_score FLOAT NOT NULL,
    prediction_interval_lower TIMESTAMPTZ,
    prediction_interval_upper TIMESTAMPTZ,
    model_type model_type_enum NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    feature_importance JSONB DEFAULT '{}',
    alternatives JSONB DEFAULT '[]',
    actual_transition_time TIMESTAMPTZ,
    accuracy_minutes FLOAT,
    is_accurate BOOLEAN,
    validation_timestamp TIMESTAMPTZ,
    
    -- Fixed foreign key constraints
    triggering_event_id BIGINT REFERENCES sensor_events(id) ON DELETE SET NULL,
    room_state_id BIGINT REFERENCES room_states(id) ON DELETE SET NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processing_time_ms FLOAT
);
```

## Key Changes Made

1. **Simplified Primary Keys**: Changed from composite to simple primary keys to avoid foreign key complexity
2. **Proper Cascade Behavior**: Added `ON DELETE SET NULL` to foreign key constraints
3. **Nullable Foreign Keys**: Made foreign key columns nullable to support optional relationships
4. **Removed Redundant Constraints**: Eliminated unnecessary unique constraints that conflicted with primary keys

## TimescaleDB Compatibility

The fixes maintain full compatibility with TimescaleDB:
- `sensor_events` can still be converted to a hypertable partitioned by `timestamp`
- Primary key on `id` ensures proper foreign key references
- All time-series optimizations and indexing strategies remain intact

## Database Creation Order

With these fixes, tables can be created in any order since foreign key constraints are properly defined:

1. `sensor_events` (no dependencies)
2. `room_states` (no dependencies) 
3. `predictions` (references both above tables)
4. `model_accuracy` (no foreign key dependencies)
5. `feature_store` (no foreign key dependencies)

## Validation

The fixed models should now pass PostgreSQL's foreign key constraint validation:
- ✅ All foreign key targets have unique constraints (primary keys)
- ✅ Foreign key columns are properly typed and nullable
- ✅ Cascade options are configured for data integrity
- ✅ No circular dependencies or constraint conflicts

These fixes ensure that the database schema creation will succeed without foreign key constraint errors.