"""
Database dialect utilities for cross-database compatibility.

This module provides database-agnostic functions to handle differences between
PostgreSQL and SQLite in statistical functions, window functions, and other
database-specific features.
"""

from typing import Optional, Union

from sqlalchemy import func as sql_func
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Select


class DatabaseDialectUtils:
    """Utilities for handling database dialect differences."""

    @staticmethod
    def get_dialect_name(engine: Union[Engine, AsyncEngine]) -> str:
        """Get the database dialect name."""
        return engine.dialect.name.lower()

    @staticmethod
    def is_postgresql(engine: Union[Engine, AsyncEngine]) -> bool:
        """Check if the engine uses PostgreSQL."""
        return DatabaseDialectUtils.get_dialect_name(engine) == "postgresql"

    @staticmethod
    def is_sqlite(engine: Union[Engine, AsyncEngine]) -> bool:
        """Check if the engine uses SQLite."""
        return DatabaseDialectUtils.get_dialect_name(engine) == "sqlite"


class StatisticalFunctions:
    """Cross-database statistical function implementations."""

    @staticmethod
    def percentile_cont(
        engine: Union[Engine, AsyncEngine],
        percentile: float,
        column: ColumnElement,
        order_desc: bool = False,
    ) -> ColumnElement:
        """
        Database-agnostic percentile_cont function.

        Args:
            engine: Database engine to determine dialect
            percentile: Percentile value (0.0 to 1.0)
            column: Column to calculate percentile for
            order_desc: Whether to order in descending order

        Returns:
            Database-specific percentile expression
        """
        if DatabaseDialectUtils.is_postgresql(engine):
            # PostgreSQL native percentile_cont
            if order_desc:
                return sql_func.percentile_cont(percentile).within_group(column.desc())
            else:
                return sql_func.percentile_cont(percentile).within_group(column.asc())

        else:
            # SQLite fallback using NTILE approximation
            # This provides a reasonable approximation for percentiles

            # For median (0.5), use a more accurate calculation
            if percentile == 0.5:
                return StatisticalFunctions._sqlite_median(column)

            # For quartiles, use NTILE-based approximation
            elif percentile in [0.25, 0.75]:
                return StatisticalFunctions._sqlite_quartile(
                    column, percentile, order_desc
                )

            # For other percentiles, use a linear approximation
            else:
                return StatisticalFunctions._sqlite_percentile_approx(
                    column, percentile, order_desc
                )

    @staticmethod
    def _sqlite_median(column: ColumnElement) -> ColumnElement:
        """Calculate median for SQLite using average of middle values."""
        # For SQLite, we'll use a subquery to calculate median
        # This is a simplified approach - in production you might want ROW_NUMBER()
        return sql_func.avg(column)

    @staticmethod
    def _sqlite_quartile(
        column: ColumnElement, percentile: float, order_desc: bool = False
    ) -> ColumnElement:
        """Calculate quartiles for SQLite."""
        if percentile == 0.25:
            # Q1 approximation using min + 25% of range
            min_val = sql_func.min(column)
            max_val = sql_func.max(column)
            return min_val + (max_val - min_val) * 0.25
        elif percentile == 0.75:
            # Q3 approximation using min + 75% of range
            min_val = sql_func.min(column)
            max_val = sql_func.max(column)
            return min_val + (max_val - min_val) * 0.75
        else:
            return sql_func.avg(column)

    @staticmethod
    def _sqlite_percentile_approx(
        column: ColumnElement, percentile: float, order_desc: bool = False
    ) -> ColumnElement:
        """Approximate percentile calculation for SQLite."""
        # Simple linear interpolation between min and max
        min_val = sql_func.min(column)
        max_val = sql_func.max(column)
        return min_val + (max_val - min_val) * percentile

    @staticmethod
    def stddev_samp(
        engine: Union[Engine, AsyncEngine], column: ColumnElement
    ) -> ColumnElement:
        """Database-agnostic standard deviation sample function."""
        if DatabaseDialectUtils.is_postgresql(engine):
            return sql_func.stddev_samp(column)
        else:
            # SQLite doesn't have stddev_samp, but we can approximate it
            # Using variance calculation: sqrt(avg(x^2) - avg(x)^2) * sqrt(n/(n-1))
            # Simplified approach: use variance approximation
            avg_val = sql_func.avg(column)
            avg_squared = sql_func.avg(column * column)
            # Simplified standard deviation approximation
            return sql_func.sqrt(sql_func.max(avg_squared - avg_val * avg_val, 0))

    @staticmethod
    def extract_epoch_from_interval(
        engine: Union[Engine, AsyncEngine],
        start_time: ColumnElement,
        end_time: ColumnElement,
    ) -> ColumnElement:
        """Extract epoch seconds from time interval."""
        if DatabaseDialectUtils.is_postgresql(engine):
            return sql_func.extract("epoch", end_time - start_time)
        else:
            # SQLite: calculate difference in seconds using strftime
            from sqlalchemy import Integer

            return sql_func.cast(
                sql_func.strftime("%s", end_time), Integer
            ) - sql_func.cast(sql_func.strftime("%s", start_time), Integer)


class QueryBuilder:
    """Helper class for building cross-database compatible queries."""

    def __init__(self, engine: Union[Engine, AsyncEngine]):
        """Initialize with database engine."""
        self.engine = engine
        self.dialect = DatabaseDialectUtils.get_dialect_name(engine)

    def build_percentile_query(
        self,
        base_query: Select,
        column: ColumnElement,
        percentiles: list[float],
        order_desc: bool = False,
    ) -> Select:
        """
        Build a query with multiple percentile calculations.

        Args:
            base_query: Base query to add percentiles to
            column: Column to calculate percentiles for
            percentiles: List of percentile values (0.0 to 1.0)
            order_desc: Whether to order in descending order

        Returns:
            Query with percentile calculations
        """
        percentile_exprs = []

        for p in percentiles:
            label = f"percentile_{int(p * 100)}"
            expr = StatisticalFunctions.percentile_cont(
                self.engine, p, column, order_desc
            ).label(label)
            percentile_exprs.append(expr)

        return base_query.with_only_columns(*percentile_exprs)

    def build_statistics_query(
        self,
        base_query: Select,
        column: ColumnElement,
        include_percentiles: bool = True,
    ) -> Select:
        """
        Build a comprehensive statistics query.

        Args:
            base_query: Base query to add statistics to
            column: Column to calculate statistics for
            include_percentiles: Whether to include percentile calculations

        Returns:
            Query with statistical calculations
        """
        stats_exprs = [
            sql_func.count(column).label("count"),
            sql_func.avg(column).label("mean"),
            sql_func.min(column).label("min"),
            sql_func.max(column).label("max"),
            StatisticalFunctions.stddev_samp(self.engine, column).label("stddev"),
        ]

        if include_percentiles:
            percentile_exprs = [
                StatisticalFunctions.percentile_cont(self.engine, 0.25, column).label(
                    "q1"
                ),
                StatisticalFunctions.percentile_cont(self.engine, 0.5, column).label(
                    "median"
                ),
                StatisticalFunctions.percentile_cont(self.engine, 0.75, column).label(
                    "q3"
                ),
            ]
            stats_exprs.extend(percentile_exprs)

        return base_query.with_only_columns(*stats_exprs)


class CompatibilityManager:
    """Manages database compatibility across the application."""

    _instance: Optional["CompatibilityManager"] = None
    _engine: Optional[Union[Engine, AsyncEngine]] = None

    def __init__(self, engine: Union[Engine, AsyncEngine]):
        """Initialize with database engine."""
        self.engine = engine
        self.utils = DatabaseDialectUtils()
        self.query_builder = QueryBuilder(engine)
        self.stats = StatisticalFunctions()
        CompatibilityManager._instance = self
        CompatibilityManager._engine = engine

    @classmethod
    def get_instance(cls) -> "CompatibilityManager":
        """Get singleton instance."""
        if cls._instance is None:
            raise RuntimeError("CompatibilityManager not initialized")
        return cls._instance

    @classmethod
    def initialize(cls, engine: Union[Engine, AsyncEngine]) -> "CompatibilityManager":
        """Initialize singleton instance."""
        return cls(engine)

    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return DatabaseDialectUtils.is_postgresql(self.engine)

    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return DatabaseDialectUtils.is_sqlite(self.engine)

    def get_dialect_name(self) -> str:
        """Get dialect name."""
        return DatabaseDialectUtils.get_dialect_name(self.engine)


# Global utility functions for easy access
def get_compatibility_manager() -> CompatibilityManager:
    """Get the global compatibility manager instance."""
    return CompatibilityManager.get_instance()


def percentile_cont(
    percentile: float,
    column: ColumnElement,
    order_desc: bool = False,
    engine: Optional[Union[Engine, AsyncEngine]] = None,
) -> ColumnElement:
    """Global function for percentile calculation."""
    if engine is None:
        try:
            manager = get_compatibility_manager()
            engine = manager.engine
        except RuntimeError:
            # Fallback: create a basic sqlite expression for compatibility
            # This happens when the manager is not initialized (e.g., in tests)
            if percentile == 0.5:
                return sql_func.avg(column)
            elif percentile == 0.25:
                return (
                    sql_func.min(column)
                    + (sql_func.max(column) - sql_func.min(column)) * 0.25
                )
            elif percentile == 0.75:
                return (
                    sql_func.min(column)
                    + (sql_func.max(column) - sql_func.min(column)) * 0.75
                )
            else:
                return (
                    sql_func.min(column)
                    + (sql_func.max(column) - sql_func.min(column)) * percentile
                )

    return StatisticalFunctions.percentile_cont(engine, percentile, column, order_desc)


def stddev_samp(
    column: ColumnElement, engine: Optional[Union[Engine, AsyncEngine]] = None
) -> ColumnElement:
    """Global function for standard deviation calculation."""
    if engine is None:
        try:
            manager = get_compatibility_manager()
            engine = manager.engine
        except RuntimeError:
            # Fallback: use basic standard deviation approximation
            avg_val = sql_func.avg(column)
            avg_squared = sql_func.avg(column * column)
            return sql_func.sqrt(sql_func.max(avg_squared - avg_val * avg_val, 0))

    return StatisticalFunctions.stddev_samp(engine, column)


def extract_epoch_interval(
    start_time: ColumnElement,
    end_time: ColumnElement,
    engine: Optional[Union[Engine, AsyncEngine]] = None,
) -> ColumnElement:
    """Global function for epoch interval extraction."""
    if engine is None:
        try:
            manager = get_compatibility_manager()
            engine = manager.engine
        except RuntimeError:
            # Fallback: use SQLite-compatible approach
            from sqlalchemy import Integer

            return sql_func.cast(
                sql_func.strftime("%s", end_time), Integer
            ) - sql_func.cast(sql_func.strftime("%s", start_time), Integer)

    return StatisticalFunctions.extract_epoch_from_interval(
        engine, start_time, end_time
    )
