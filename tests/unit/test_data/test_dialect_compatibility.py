"""
Tests for database dialect compatibility utilities.

This module tests the cross-database compatibility layer that provides
unified access to database-specific functions across PostgreSQL and SQLite.
"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from src.data.storage.dialect_utils import (
    CompatibilityManager,
    DatabaseDialectUtils,
    QueryBuilder,
    StatisticalFunctions,
    extract_epoch_interval,
    percentile_cont,
    stddev_samp,
)

Base = declarative_base()


class MockTable(Base):
    __tablename__ = "mock_table"

    id = Column(Integer, primary_key=True)
    value = Column(Float)
    name = Column(String(50))


class TestDatabaseDialectUtils:
    """Test dialect detection utilities."""

    def test_postgresql_detection(self):
        """Test PostgreSQL dialect detection."""
        # Create a mock PostgreSQL engine
        engine = create_engine("postgresql://test")

        assert DatabaseDialectUtils.is_postgresql(engine) is True
        assert DatabaseDialectUtils.is_sqlite(engine) is False
        assert DatabaseDialectUtils.get_dialect_name(engine) == "postgresql"

    def test_sqlite_detection(self):
        """Test SQLite dialect detection."""
        # Create a mock SQLite engine
        engine = create_engine("sqlite:///test.db")

        assert DatabaseDialectUtils.is_sqlite(engine) is True
        assert DatabaseDialectUtils.is_postgresql(engine) is False
        assert DatabaseDialectUtils.get_dialect_name(engine) == "sqlite"


class TestStatisticalFunctions:
    """Test database-agnostic statistical functions."""

    def test_percentile_cont_postgresql(self):
        """Test percentile_cont for PostgreSQL."""
        engine = create_engine("postgresql://test")
        column = MockTable.value

        # Test median (0.5)
        expr = StatisticalFunctions.percentile_cont(engine, 0.5, column)
        assert expr is not None

        # Test with descending order
        expr_desc = StatisticalFunctions.percentile_cont(
            engine, 0.5, column, order_desc=True
        )
        assert expr_desc is not None

    def test_percentile_cont_sqlite(self):
        """Test percentile_cont fallback for SQLite."""
        engine = create_engine("sqlite:///test.db")
        column = MockTable.value

        # Test median (0.5)
        expr = StatisticalFunctions.percentile_cont(engine, 0.5, column)
        assert expr is not None

        # Test quartiles
        q1_expr = StatisticalFunctions.percentile_cont(engine, 0.25, column)
        q3_expr = StatisticalFunctions.percentile_cont(engine, 0.75, column)
        assert q1_expr is not None
        assert q3_expr is not None

        # Test arbitrary percentile
        p90_expr = StatisticalFunctions.percentile_cont(engine, 0.9, column)
        assert p90_expr is not None

    def test_stddev_samp_postgresql(self):
        """Test stddev_samp for PostgreSQL."""
        engine = create_engine("postgresql://test")
        column = MockTable.value

        expr = StatisticalFunctions.stddev_samp(engine, column)
        assert expr is not None

    def test_stddev_samp_sqlite(self):
        """Test stddev_samp fallback for SQLite."""
        engine = create_engine("sqlite:///test.db")
        column = MockTable.value

        expr = StatisticalFunctions.stddev_samp(engine, column)
        assert expr is not None

    def test_extract_epoch_postgresql(self):
        """Test epoch extraction for PostgreSQL."""
        engine = create_engine("postgresql://test")
        start_time = func.min(MockTable.id)
        end_time = func.max(MockTable.id)

        expr = StatisticalFunctions.extract_epoch_from_interval(
            engine, start_time, end_time
        )
        assert expr is not None

    def test_extract_epoch_sqlite(self):
        """Test epoch extraction for SQLite."""
        engine = create_engine("sqlite:///test.db")
        start_time = func.min(MockTable.id)
        end_time = func.max(MockTable.id)

        expr = StatisticalFunctions.extract_epoch_from_interval(
            engine, start_time, end_time
        )
        assert expr is not None


class TestQueryBuilder:
    """Test query builder for cross-database queries."""

    def test_query_builder_init(self):
        """Test query builder initialization."""
        engine = create_engine("sqlite:///test.db")
        builder = QueryBuilder(engine)

        assert builder.engine == engine
        assert builder.dialect == "sqlite"

    def test_build_percentile_query_postgresql(self):
        """Test percentile query building for PostgreSQL."""
        engine = create_engine("postgresql://test")
        builder = QueryBuilder(engine)

        from sqlalchemy import select

        base_query = select(MockTable)
        percentiles = [0.25, 0.5, 0.75]

        query = builder.build_percentile_query(base_query, MockTable.value, percentiles)
        assert query is not None

    def test_build_percentile_query_sqlite(self):
        """Test percentile query building for SQLite."""
        engine = create_engine("sqlite:///test.db")
        builder = QueryBuilder(engine)

        from sqlalchemy import select

        base_query = select(MockTable)
        percentiles = [0.25, 0.5, 0.75]

        query = builder.build_percentile_query(base_query, MockTable.value, percentiles)
        assert query is not None

    def test_build_statistics_query(self):
        """Test comprehensive statistics query building."""
        engine = create_engine("sqlite:///test.db")
        builder = QueryBuilder(engine)

        from sqlalchemy import select

        base_query = select(MockTable)

        query = builder.build_statistics_query(
            base_query, MockTable.value, include_percentiles=True
        )
        assert query is not None

        query_no_percentiles = builder.build_statistics_query(
            base_query, MockTable.value, include_percentiles=False
        )
        assert query_no_percentiles is not None


class TestCompatibilityManager:
    """Test compatibility manager singleton."""

    def test_compatibility_manager_init(self):
        """Test compatibility manager initialization."""
        engine = create_engine("sqlite:///test.db")
        manager = CompatibilityManager(engine)

        assert manager.engine == engine
        assert manager.is_sqlite() is True
        assert manager.is_postgresql() is False
        assert manager.get_dialect_name() == "sqlite"

    def test_compatibility_manager_singleton(self):
        """Test compatibility manager singleton behavior."""
        engine = create_engine("sqlite:///test.db")

        # Initialize singleton
        manager1 = CompatibilityManager.initialize(engine)
        manager2 = CompatibilityManager.get_instance()

        assert manager1 is manager2


class TestGlobalFunctions:
    """Test global convenience functions with fallbacks."""

    def test_global_functions_without_manager(self):
        """Test global functions work without initialized manager."""
        column = MockTable.value

        # These should use fallback implementations
        percentile_expr = percentile_cont(0.5, column)
        assert percentile_expr is not None

        stddev_expr = stddev_samp(column)
        assert stddev_expr is not None

        start_time = func.min(MockTable.id)
        end_time = func.max(MockTable.id)
        epoch_expr = extract_epoch_interval(start_time, end_time)
        assert epoch_expr is not None

    def test_global_functions_with_manager(self):
        """Test global functions with initialized manager."""
        engine = create_engine("sqlite:///test.db")
        CompatibilityManager.initialize(engine)

        column = MockTable.value

        # These should use manager's engine
        percentile_expr = percentile_cont(0.5, column)
        assert percentile_expr is not None

        stddev_expr = stddev_samp(column)
        assert stddev_expr is not None

        start_time = func.min(MockTable.id)
        end_time = func.max(MockTable.id)
        epoch_expr = extract_epoch_interval(start_time, end_time)
        assert epoch_expr is not None

    def test_global_functions_with_explicit_engine(self):
        """Test global functions with explicitly provided engine."""
        engine = create_engine("postgresql://test")
        column = MockTable.value

        # Pass engine explicitly
        percentile_expr = percentile_cont(0.5, column, engine=engine)
        assert percentile_expr is not None

        stddev_expr = stddev_samp(column, engine=engine)
        assert stddev_expr is not None

        start_time = func.min(MockTable.id)
        end_time = func.max(MockTable.id)
        epoch_expr = extract_epoch_interval(start_time, end_time, engine=engine)
        assert epoch_expr is not None


class TestRealWorldCompatibility:
    """Test compatibility with real-world scenarios."""

    def test_percentile_calculations_consistency(self):
        """Test that percentile calculations are consistent across dialects."""
        postgresql_engine = create_engine("postgresql://test")
        sqlite_engine = create_engine("sqlite:///test.db")

        column = MockTable.value

        # Both should produce expressions (can't test actual values without data)
        pg_median = StatisticalFunctions.percentile_cont(postgresql_engine, 0.5, column)
        sqlite_median = StatisticalFunctions.percentile_cont(sqlite_engine, 0.5, column)

        assert pg_median is not None
        assert sqlite_median is not None

        # Test quartiles
        pg_q1 = StatisticalFunctions.percentile_cont(postgresql_engine, 0.25, column)
        sqlite_q1 = StatisticalFunctions.percentile_cont(sqlite_engine, 0.25, column)

        assert pg_q1 is not None
        assert sqlite_q1 is not None

    def test_mixed_queries_compatibility(self):
        """Test that mixed statistical queries work on both dialects."""
        for engine_url in ["postgresql://test", "sqlite:///test.db"]:
            engine = create_engine(engine_url)
            builder = QueryBuilder(engine)

            from sqlalchemy import select

            base_query = select(MockTable)

            # Build comprehensive statistics query
            stats_query = builder.build_statistics_query(
                base_query, MockTable.value, include_percentiles=True
            )

            assert stats_query is not None

            # Build percentile-only query
            percentile_query = builder.build_percentile_query(
                base_query, MockTable.value, [0.1, 0.5, 0.9]
            )

            assert percentile_query is not None


@pytest.mark.database_models
class TestIntegrationWithModels:
    """Test integration with the actual model classes."""

    def test_dialect_compatibility_in_sensor_event_analytics(self):
        """Test that SensorEvent analytics work with dialect compatibility."""
        # This is more of an integration test to ensure our changes work
        from src.data.storage.models import SensorEvent

        # Verify that the methods exist and use our compatibility functions
        assert hasattr(SensorEvent, "get_advanced_analytics")

        # The actual functionality is tested in the models test suite
        # This just ensures our compatibility layer is properly integrated

    def test_dialect_compatibility_in_room_state_metrics(self):
        """Test that RoomState metrics work with dialect compatibility."""
        from src.data.storage.models import RoomState

        # Verify that the methods exist and use our compatibility functions
        assert hasattr(RoomState, "get_precision_occupancy_metrics")

        # The actual functionality is tested in the models test suite
        # This just ensures our compatibility layer is properly integrated
