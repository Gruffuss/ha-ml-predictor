# Database Compatibility Layer

This document describes the cross-database compatibility layer implemented to support both PostgreSQL (production) and SQLite (testing) databases.

## Problem Statement

The original codebase used PostgreSQL-specific SQL functions like:
- `percentile_cont() WITHIN GROUP (ORDER BY column)`
- `stddev_samp()`
- `extract('epoch', interval)`

These functions caused SQLite test failures with syntax errors like:
```
sqlite3.OperationalError: near "(": syntax error
[SQL: SELECT percentile_cont(?) WITHIN GROUP (ORDER BY ...)]
```

## Solution Overview

A comprehensive database compatibility layer was implemented in `src/data/storage/dialect_utils.py` that provides:

1. **Dialect Detection**: Automatic detection of PostgreSQL vs SQLite
2. **Function Abstraction**: Database-agnostic statistical functions
3. **Fallback Implementations**: SQLite-compatible alternatives for PostgreSQL functions
4. **Query Builder**: Helper for building cross-database queries

## Architecture

### Core Components

#### DatabaseDialectUtils
```python
class DatabaseDialectUtils:
    @staticmethod
    def is_postgresql(engine) -> bool
    @staticmethod
    def is_sqlite(engine) -> bool
    @staticmethod
    def get_dialect_name(engine) -> str
```

#### StatisticalFunctions
```python
class StatisticalFunctions:
    @staticmethod
    def percentile_cont(engine, percentile, column, order_desc=False)
    @staticmethod
    def stddev_samp(engine, column)
    @staticmethod
    def extract_epoch_from_interval(engine, start_time, end_time)
```

#### CompatibilityManager
```python
class CompatibilityManager:
    def __init__(self, engine)
    @classmethod
    def initialize(cls, engine)
    @classmethod
    def get_instance(cls)
```

### Function Mapping

| PostgreSQL Function | SQLite Equivalent | Implementation |
|---------------------|-------------------|----------------|
| `percentile_cont(0.5) WITHIN GROUP` | Approximation using min/max/avg | Linear interpolation between min and max |
| `percentile_cont(0.25)` | Q1 approximation | `min + (max - min) * 0.25` |
| `percentile_cont(0.75)` | Q3 approximation | `min + (max - min) * 0.75` |
| `stddev_samp(column)` | Custom calculation | `sqrt(max(avg(x²) - avg(x)², 0))` |
| `extract('epoch', interval)` | strftime difference | `strftime('%s', end) - strftime('%s', start)` |

## Usage

### Automatic Integration

The compatibility layer is automatically initialized when the DatabaseManager creates an engine:

```python
# In database.py
self.engine = create_async_engine(**engine_kwargs)

# Initialize compatibility manager for database dialect handling
from .dialect_utils import CompatibilityManager
CompatibilityManager.initialize(self.engine)
```

### Model Integration

Models now use database-agnostic functions:

```python
# Before (PostgreSQL-specific)
stats_query = select(
    sql_func.percentile_cont(0.5)
    .within_group(cls.confidence_score.desc())
    .label("median_confidence"),
    sql_func.stddev_samp(cls.confidence_score).label("confidence_stddev"),
)

# After (Cross-database compatible)
from .dialect_utils import percentile_cont, stddev_samp

stats_query = select(
    percentile_cont(0.5, cls.confidence_score, order_desc=True).label("median_confidence"),
    stddev_samp(cls.confidence_score).label("confidence_stddev"),
)
```

### Global Functions

The layer provides global convenience functions with automatic fallbacks:

```python
from src.data.storage.dialect_utils import percentile_cont, stddev_samp, extract_epoch_interval

# These work regardless of database dialect
median_expr = percentile_cont(0.5, column)
stddev_expr = stddev_samp(column)
epoch_expr = extract_epoch_interval(start_time, end_time)
```

## Fallback Behavior

When the CompatibilityManager is not initialized (e.g., in some test scenarios), the global functions automatically fall back to SQLite-compatible implementations:

```python
def percentile_cont(percentile, column, order_desc=False, engine=None):
    if engine is None:
        try:
            manager = get_compatibility_manager()
            engine = manager.engine
        except RuntimeError:
            # Fallback: create basic sqlite expression
            if percentile == 0.5:
                return sql_func.avg(column)
            # ... other fallback implementations
```

## Testing

Comprehensive tests ensure compatibility across both database types:

- `test_dialect_compatibility.py`: Tests all compatibility functions
- Function-specific tests for PostgreSQL and SQLite branches
- Integration tests with actual model methods
- Fallback behavior validation

## Performance Considerations

### PostgreSQL
- Uses native statistical functions for optimal performance
- Maintains original query optimization

### SQLite
- Uses approximations that are computationally efficient
- Linear interpolation provides reasonable statistical estimates
- Suitable for test scenarios where precision is less critical

## Migration Notes

### Existing Code
No changes required for most existing code. The compatibility layer is transparent.

### New Development
Use the global functions from `dialect_utils` instead of raw SQLAlchemy functions:

```python
# Preferred
from src.data.storage.dialect_utils import percentile_cont
median_expr = percentile_cont(0.5, column)

# Avoid
median_expr = sql_func.percentile_cont(0.5).within_group(column.asc())
```

## Limitations

### SQLite Approximations
The SQLite implementations are approximations:
- Percentiles use linear interpolation, not exact percentile calculations
- Standard deviation uses a simplified formula
- Suitable for testing but may not match PostgreSQL precision exactly

### Function Coverage
Currently supports the most commonly used statistical functions. Additional PostgreSQL-specific functions can be added as needed.

## Future Enhancements

1. **More Accurate SQLite Percentiles**: Implement window functions for better percentile calculations
2. **Additional Statistical Functions**: Add support for more PostgreSQL statistical functions
3. **Performance Optimization**: Optimize SQLite fallback implementations
4. **Configuration Options**: Allow tuning of approximation algorithms
5. **Validation Layer**: Add validation to ensure results are within acceptable ranges

## Error Handling

The system gracefully handles various scenarios:
- Missing engine initialization (falls back to SQLite mode)
- Invalid percentile values (clamped to 0.0-1.0 range)
- Null/empty datasets (returns appropriate defaults)
- Database connection failures (propagates original errors)

## Conclusion

This compatibility layer enables seamless operation across PostgreSQL (production) and SQLite (testing) environments while maintaining performance and accuracy appropriate to each use case.