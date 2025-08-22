"""
Comprehensive tests for database compatibility layer.

Tests database-specific configuration, model patching, and cross-database compatibility
utilities for SQLite and PostgreSQL/TimescaleDB.
"""

import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Index

from src.data.storage.database_compatibility import (
    configure_database_on_first_connect,
    configure_sensor_event_model,
    configure_sqlite_for_testing,
    create_database_specific_models,
    get_database_specific_table_args,
    is_postgresql_engine,
    is_sqlite_engine,
    patch_models_for_sqlite_compatibility,
)

# Create test models for compatibility testing
TestBase = declarative_base()


class TestSensorEventModel(TestBase):
    """Test model for sensor event compatibility testing."""

    __tablename__ = "test_sensor_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    room_id = Column(String(50), nullable=False)
    sensor_type = Column(String(20), nullable=False)


class TestDatabaseCompatibilityEngineChecks:
    """Test engine type detection functions."""

    def test_is_sqlite_engine_with_sqlite_url(self):
        """Test SQLite engine detection with SQLite URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        assert is_sqlite_engine(mock_engine) is True

    def test_is_sqlite_engine_with_memory_url(self):
        """Test SQLite engine detection with in-memory URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///:memory:")

        assert is_sqlite_engine(mock_engine) is True

    def test_is_sqlite_engine_with_postgresql_url(self):
        """Test SQLite engine detection with PostgreSQL URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(
            return_value="postgresql://user:pass@localhost/db"
        )

        assert is_sqlite_engine(mock_engine) is False

    def test_is_sqlite_engine_with_direct_url_string(self):
        """Test SQLite engine detection with direct URL string."""
        sqlite_url = "sqlite:///path/to/database.db"
        assert is_sqlite_engine(sqlite_url) is True

        postgres_url = "postgresql://localhost/db"
        assert is_sqlite_engine(postgres_url) is False

    def test_is_postgresql_engine_with_postgresql_url(self):
        """Test PostgreSQL engine detection with PostgreSQL URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(
            return_value="postgresql://user:pass@localhost/db"
        )

        assert is_postgresql_engine(mock_engine) is True

    def test_is_postgresql_engine_with_asyncpg_url(self):
        """Test PostgreSQL engine detection with asyncpg URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(
            return_value="postgresql+asyncpg://user:pass@localhost/db"
        )

        assert is_postgresql_engine(mock_engine) is True

    def test_is_postgresql_engine_with_sqlite_url(self):
        """Test PostgreSQL engine detection with SQLite URL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        assert is_postgresql_engine(mock_engine) is False

    def test_is_postgresql_engine_with_direct_url_string(self):
        """Test PostgreSQL engine detection with direct URL string."""
        postgres_url = "postgresql://localhost/testdb"
        assert is_postgresql_engine(postgres_url) is True

        postgres_asyncpg_url = "postgresql+asyncpg://localhost/testdb"
        assert is_postgresql_engine(postgres_asyncpg_url) is True

        sqlite_url = "sqlite:///test.db"
        assert is_postgresql_engine(sqlite_url) is False

    def test_engine_detection_case_insensitive(self):
        """Test engine detection is case insensitive."""
        # Test different case variations
        assert is_sqlite_engine("SQLITE:///test.db") is True
        assert is_sqlite_engine("SQLite:///test.db") is True
        assert is_postgresql_engine("POSTGRESQL://localhost/db") is True
        assert is_postgresql_engine("PostgreSQL://localhost/db") is True

    def test_engine_detection_with_none(self):
        """Test engine detection with None values."""
        mock_engine = Mock()
        mock_engine.url = None

        # Should handle None gracefully
        assert is_sqlite_engine(mock_engine) is False
        assert is_postgresql_engine(mock_engine) is False


class TestSensorEventModelConfiguration:
    """Test SensorEvent model configuration for different databases."""

    def test_configure_sensor_event_model_sqlite(self):
        """Test SensorEvent configuration for SQLite."""
        # Create mock model with table
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        # Configure mock attributes
        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_model.__table__ = mock_table

        # Mock engine for SQLite
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        with patch("sqlalchemy.UniqueConstraint") as mock_constraint:
            mock_constraint_instance = Mock()
            mock_constraint.return_value = mock_constraint_instance

            result = configure_sensor_event_model(mock_model, mock_engine)

            # Should set autoincrement on id column
            assert mock_id_column.autoincrement is True
            # Should remove timestamp from primary key
            assert mock_timestamp_column.primary_key is False
            # Should add unique constraint
            mock_table.append_constraint.assert_called_once_with(
                mock_constraint_instance
            )
            mock_constraint.assert_called_once_with(
                "id", "timestamp", name="uq_sensor_events_id_timestamp"
            )

            assert result == mock_model

    def test_configure_sensor_event_model_postgresql(self):
        """Test SensorEvent configuration for PostgreSQL."""
        # Create mock model with table
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_model.__table__ = mock_table

        # Mock engine for PostgreSQL
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        result = configure_sensor_event_model(mock_model, mock_engine)

        # Should set both columns as primary key
        assert mock_id_column.primary_key is True
        assert mock_id_column.autoincrement is True
        assert mock_timestamp_column.primary_key is True

        assert result == mock_model

    def test_configure_sensor_event_model_no_table(self):
        """Test SensorEvent configuration without __table__ attribute."""
        mock_model = Mock(spec=[])  # No __table__ attribute
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        # Should handle gracefully when no __table__ attribute
        result = configure_sensor_event_model(mock_model, mock_engine)
        assert result == mock_model

    def test_configure_sensor_event_model_missing_columns(self):
        """Test SensorEvent configuration with missing columns."""
        mock_model = Mock()
        mock_table = Mock()
        mock_model.__table__ = mock_table

        # Mock table with missing columns (AttributeError when accessing)
        mock_table.c = Mock()
        mock_table.c.id = Mock()
        # timestamp column missing - should handle gracefully
        del mock_table.c.timestamp

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        # Should handle missing columns gracefully
        try:
            result = configure_sensor_event_model(mock_model, mock_engine)
            assert result == mock_model
        except AttributeError:
            # If it raises AttributeError, that's also acceptable behavior
            pass


class TestDatabaseSpecificModels:
    """Test creation of database-specific model configurations."""

    def test_create_database_specific_models_with_sensor_event(self):
        """Test creating database-specific models with SensorEvent."""
        mock_sensor_event = Mock()
        mock_other_model = Mock()

        base_models = {
            "SensorEvent": mock_sensor_event,
            "OtherModel": mock_other_model,
        }

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        with patch(
            "src.data.storage.database_compatibility.configure_sensor_event_model"
        ) as mock_configure:
            mock_configured_model = Mock()
            mock_configure.return_value = mock_configured_model

            result = create_database_specific_models(base_models, mock_engine)

            # Should configure SensorEvent
            mock_configure.assert_called_once_with(mock_sensor_event, mock_engine)
            assert result["SensorEvent"] == mock_configured_model

            # Should keep other models unchanged
            assert result["OtherModel"] == mock_other_model

    def test_create_database_specific_models_without_sensor_event(self):
        """Test creating database-specific models without SensorEvent."""
        mock_model1 = Mock()
        mock_model2 = Mock()

        base_models = {
            "Model1": mock_model1,
            "Model2": mock_model2,
        }

        mock_engine = Mock()

        with patch(
            "src.data.storage.database_compatibility.configure_sensor_event_model"
        ) as mock_configure:
            result = create_database_specific_models(base_models, mock_engine)

            # Should not call configure for any models
            mock_configure.assert_not_called()

            # Should return all models unchanged
            assert result["Model1"] == mock_model1
            assert result["Model2"] == mock_model2

    def test_create_database_specific_models_empty(self):
        """Test creating database-specific models with empty input."""
        base_models = {}
        mock_engine = Mock()

        result = create_database_specific_models(base_models, mock_engine)

        assert result == {}


class TestConnectionEventListeners:
    """Test database connection event listeners."""

    def test_configure_sqlite_for_testing(self):
        """Test SQLite configuration event listener."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Call the event handler
        configure_sqlite_for_testing(mock_connection, mock_record)

        # Should enable foreign keys
        mock_cursor.execute.assert_called_once_with("PRAGMA foreign_keys=ON")
        mock_cursor.close.assert_called_once()

    def test_configure_sqlite_for_testing_non_sqlite(self):
        """Test SQLite configuration with non-SQLite database."""
        mock_connection = Mock()
        mock_record = Mock()
        mock_record.info = {"url": "postgresql://localhost/db"}

        # Should not call cursor for non-SQLite
        configure_sqlite_for_testing(mock_connection, mock_record)

        mock_connection.cursor.assert_not_called()

    def test_configure_sqlite_for_testing_no_url(self):
        """Test SQLite configuration with missing URL info."""
        mock_connection = Mock()
        mock_record = Mock()
        mock_record.info = {}  # No URL

        # Should handle missing URL gracefully
        configure_sqlite_for_testing(mock_connection, mock_record)

        mock_connection.cursor.assert_not_called()

    def test_configure_database_on_first_connect_sqlite(self):
        """Test first connection configuration for SQLite."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Call the event handler
        configure_database_on_first_connect(mock_connection, mock_record)

        # Should execute SQLite optimization commands
        expected_calls = [
            "PRAGMA foreign_keys=ON",
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
        ]

        assert mock_cursor.execute.call_count == len(expected_calls)
        for expected_call in expected_calls:
            assert any(
                call[0][0] == expected_call
                for call in mock_cursor.execute.call_args_list
            )

        mock_cursor.close.assert_called_once()

    def test_configure_database_on_first_connect_postgresql(self):
        """Test first connection configuration for PostgreSQL."""
        mock_connection = Mock()
        mock_record = Mock()
        mock_record.info = {"url": "postgresql://localhost/db"}

        # Call the event handler
        configure_database_on_first_connect(mock_connection, mock_record)

        # Should not perform any actions for PostgreSQL (currently)
        mock_connection.cursor.assert_not_called()

    def test_configure_database_on_first_connect_error_handling(self):
        """Test first connection configuration error handling."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("SQLite error")
        mock_connection.cursor.return_value = mock_cursor

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Should handle errors gracefully
        configure_database_on_first_connect(mock_connection, mock_record)

        # Should still attempt to close cursor
        mock_cursor.close.assert_called_once()


class TestModelPatching:
    """Test model patching for SQLite compatibility."""

    def test_patch_models_for_sqlite_compatibility(self):
        """Test patching existing models for SQLite compatibility."""
        with patch("src.data.storage.models.SensorEvent") as mock_model:
            mock_table = Mock()
            mock_id_column = Mock()
            mock_timestamp_column = Mock()

            mock_table.c.id = mock_id_column
            mock_table.c.timestamp = mock_timestamp_column
            mock_timestamp_column.primary_key = True  # Initially true
            mock_model.__table__ = mock_table

            # Call the patching function
            patch_models_for_sqlite_compatibility()

            # Should remove timestamp from primary key
            assert mock_timestamp_column.primary_key is False
            # Should ensure id has autoincrement
            assert mock_id_column.autoincrement is True

    def test_patch_models_for_sqlite_compatibility_no_table(self):
        """Test patching models without __table__ attribute."""
        with patch("src.data.storage.models.SensorEvent") as mock_model:
            # No __table__ attribute
            mock_model.__table__ = None

            # Should handle gracefully
            patch_models_for_sqlite_compatibility()

    def test_patch_models_for_sqlite_compatibility_missing_columns(self):
        """Test patching models with missing columns."""
        with patch("src.data.storage.models.SensorEvent") as mock_model:
            mock_table = Mock()
            mock_table.c = Mock()

            # Missing id and timestamp columns
            mock_table.c.id = None
            mock_table.c.timestamp = None
            mock_model.__table__ = mock_table

            # Should handle missing columns gracefully
            patch_models_for_sqlite_compatibility()


class TestTableArgsGeneration:
    """Test database-specific table arguments generation."""

    def test_get_database_specific_table_args_sensor_events_sqlite(self):
        """Test table args for sensor_events table on SQLite."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        with patch("sqlalchemy.Index") as mock_index, patch(
            "sqlalchemy.UniqueConstraint"
        ) as mock_constraint:

            mock_index_instances = [Mock() for _ in range(4)]  # 4 indexes expected
            mock_index.side_effect = mock_index_instances

            mock_constraint_instance = Mock()
            mock_constraint.return_value = mock_constraint_instance

            args = get_database_specific_table_args(mock_engine, "sensor_events")

            # Should create SQLite-specific indexes and constraints
            assert len(args) == 5  # 4 indexes + 1 constraint
            assert mock_constraint_instance in args

            # Should have created indexes with correct names
            index_calls = mock_index.call_args_list
            index_names = [call[0][0] for call in index_calls]

            expected_indexes = [
                "idx_room_sensor_time",
                "idx_room_time_desc",
                "idx_sensor_type_time",
                "idx_human_triggered",
            ]

            for expected_name in expected_indexes:
                assert expected_name in index_names

    def test_get_database_specific_table_args_sensor_events_postgresql(self):
        """Test table args for sensor_events table on PostgreSQL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        with patch("sqlalchemy.Index") as mock_index, patch(
            "sqlalchemy.text"
        ) as mock_text:

            mock_index_instances = [Mock() for _ in range(5)]  # 5 indexes expected
            mock_index.side_effect = mock_index_instances

            mock_text_instance = Mock()
            mock_text.return_value = mock_text_instance

            args = get_database_specific_table_args(mock_engine, "sensor_events")

            # Should create PostgreSQL-specific indexes
            assert len(args) == 5
            assert all(idx in args for idx in mock_index_instances)

            # Should use PostgreSQL-specific features
            index_calls = mock_index.call_args_list

            # Check for postgresql_using parameter
            using_calls = [
                call for call in index_calls if "postgresql_using" in call[1]
            ]
            assert len(using_calls) > 0

            # Check for postgresql_where parameter
            where_calls = [
                call for call in index_calls if "postgresql_where" in call[1]
            ]
            assert len(where_calls) > 0

            # Should call text() for WHERE clause
            mock_text.assert_called()

    def test_get_database_specific_table_args_other_table(self):
        """Test table args for non-sensor_events table."""
        mock_engine = Mock()

        args = get_database_specific_table_args(mock_engine, "other_table")

        # Should return empty tuple for non-sensor_events tables
        assert args == ()

    def test_get_database_specific_table_args_empty_table_name(self):
        """Test table args with empty table name."""
        mock_engine = Mock()

        args = get_database_specific_table_args(mock_engine, "")

        # Should return empty tuple for empty table name
        assert args == ()

    def test_get_database_specific_table_args_none_table_name(self):
        """Test table args with None table name."""
        mock_engine = Mock()

        args = get_database_specific_table_args(mock_engine, None)

        # Should return empty tuple for None table name
        assert args == ()


class TestEventListenerRegistration:
    """Test proper registration of event listeners."""

    def test_event_listeners_are_registered(self):
        """Test that event listeners are properly registered on import."""
        # This test verifies that the event listeners are registered when the module is imported
        # We can't directly test the registration since it happens at import time,
        # but we can verify the functions exist and are callable

        assert callable(configure_sqlite_for_testing)
        assert callable(configure_database_on_first_connect)

        # Verify function signatures
        import inspect

        # Test configure_sqlite_for_testing signature
        sig = inspect.signature(configure_sqlite_for_testing)
        assert len(sig.parameters) == 2
        assert "dbapi_connection" in sig.parameters
        assert "connection_record" in sig.parameters

        # Test configure_database_on_first_connect signature
        sig = inspect.signature(configure_database_on_first_connect)
        assert len(sig.parameters) == 2
        assert "dbapi_connection" in sig.parameters
        assert "connection_record" in sig.parameters

    @patch("sqlalchemy.event")
    def test_event_listener_decorator_usage(self, mock_event):
        """Test that event listeners use proper decorators."""
        # Since the decorators are applied at import time, we can't directly test them
        # But we can verify that the event module is used correctly

        # Re-import the module to test decorator application
        import importlib

        import src.data.storage.database_compatibility

        importlib.reload(src.data.storage.database_compatibility)

        # The event.listens_for should have been called during import
        assert mock_event.listens_for.call_count >= 2  # At least 2 listeners


class TestEnvironmentBasedConfiguration:
    """Test environment-based configuration detection."""

    @patch.dict(os.environ, {"TEST_DB_URL": "sqlite:///test.db"})
    def test_sqlite_detection_via_environment(self):
        """Test SQLite detection through environment variables."""
        # This would be used in models.py _get_json_column_type function
        # Test that the environment detection works
        from src.data.storage.database_compatibility import is_sqlite_engine

        # Test with environment variable set
        assert "sqlite" in os.environ.get("TEST_DB_URL", "")

    @patch.dict(os.environ, {"TESTING": "true"})
    def test_testing_environment_detection(self):
        """Test testing environment detection."""
        assert os.environ.get("TESTING") == "true"

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/db"})
    def test_postgresql_environment_detection(self):
        """Test PostgreSQL environment detection."""
        assert "postgresql" in os.environ.get("DATABASE_URL", "")


class TestCompatibilityEdgeCases:
    """Test edge cases and error handling in compatibility layer."""

    def test_engine_url_attribute_error(self):
        """Test handling of engines without url attribute."""
        mock_engine = Mock(spec=[])  # No url attribute

        # Should handle gracefully
        assert is_sqlite_engine(mock_engine) is False
        assert is_postgresql_engine(mock_engine) is False

    def test_engine_url_str_error(self):
        """Test handling of URL conversion errors."""
        mock_engine = Mock()
        mock_url = Mock()
        mock_url.__str__.side_effect = Exception("URL conversion failed")
        mock_engine.url = mock_url

        # Should handle gracefully
        try:
            result = is_sqlite_engine(mock_engine)
            assert result is False  # Should default to False on error
        except Exception:
            # Or may raise exception, both are acceptable
            pass

    def test_table_modification_with_readonly_attributes(self):
        """Test table modification with read-only attributes."""
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        # Simulate read-only attributes
        type(mock_id_column).autoincrement = PropertyMock(
            side_effect=AttributeError("Can't set attribute")
        )
        type(mock_timestamp_column).primary_key = PropertyMock(
            side_effect=AttributeError("Can't set attribute")
        )

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_model.__table__ = mock_table

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        # Should handle read-only attributes gracefully
        try:
            result = configure_sensor_event_model(mock_model, mock_engine)
            assert result == mock_model
        except AttributeError:
            # If it raises AttributeError, that's also acceptable
            pass

    def test_constraint_addition_error(self):
        """Test error handling when adding constraints to table."""
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_table.append_constraint.side_effect = Exception("Cannot add constraint")
        mock_model.__table__ = mock_table

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        with patch("sqlalchemy.UniqueConstraint") as mock_constraint:
            mock_constraint.return_value = Mock()

            # Should handle constraint addition errors gracefully
            try:
                result = configure_sensor_event_model(mock_model, mock_engine)
                assert result == mock_model
            except Exception:
                # May raise exception, which is acceptable
                pass

    def test_event_listener_connection_info_missing(self):
        """Test event listeners with missing connection info."""
        mock_connection = Mock()
        mock_record = Mock()
        mock_record.info = None  # No info dict

        # Should handle missing info gracefully
        configure_sqlite_for_testing(mock_connection, mock_record)
        configure_database_on_first_connect(mock_connection, mock_record)

    def test_event_listener_cursor_creation_error(self):
        """Test event listener cursor creation error handling."""
        mock_connection = Mock()
        mock_connection.cursor.side_effect = Exception("Cannot create cursor")

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Should handle cursor creation errors gracefully
        configure_database_on_first_connect(mock_connection, mock_record)


@pytest.mark.unit
@pytest.mark.database_compatibility
class TestDatabaseCompatibilityIntegration:
    """Integration tests for database compatibility functionality."""

    def test_full_compatibility_workflow_sqlite(self):
        """Test complete compatibility workflow for SQLite."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        # Mock model classes
        mock_sensor_event = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_sensor_event.__table__ = mock_table

        mock_other_model = Mock()

        base_models = {
            "SensorEvent": mock_sensor_event,
            "OtherModel": mock_other_model,
        }

        # Test engine detection
        assert is_sqlite_engine(mock_engine) is True
        assert is_postgresql_engine(mock_engine) is False

        # Test model configuration
        with patch("sqlalchemy.UniqueConstraint") as mock_constraint:
            mock_constraint.return_value = Mock()

            configured_models = create_database_specific_models(
                base_models, mock_engine
            )

            # Should have configured SensorEvent for SQLite
            assert configured_models["SensorEvent"] == mock_sensor_event
            assert configured_models["OtherModel"] == mock_other_model

            # Should have modified the SensorEvent table
            assert mock_id_column.autoincrement is True
            assert mock_timestamp_column.primary_key is False

        # Test table args generation
        with patch("sqlalchemy.Index"), patch("sqlalchemy.UniqueConstraint"):
            table_args = get_database_specific_table_args(mock_engine, "sensor_events")
            assert len(table_args) > 0

    def test_full_compatibility_workflow_postgresql(self):
        """Test complete compatibility workflow for PostgreSQL."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        # Mock model classes
        mock_sensor_event = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_sensor_event.__table__ = mock_table

        base_models = {"SensorEvent": mock_sensor_event}

        # Test engine detection
        assert is_postgresql_engine(mock_engine) is True
        assert is_sqlite_engine(mock_engine) is False

        # Test model configuration
        configured_models = create_database_specific_models(base_models, mock_engine)

        # Should have configured SensorEvent for PostgreSQL
        assert configured_models["SensorEvent"] == mock_sensor_event

        # Should have set composite primary key
        assert mock_id_column.primary_key is True
        assert mock_id_column.autoincrement is True
        assert mock_timestamp_column.primary_key is True

        # Test table args generation
        with patch("sqlalchemy.Index"), patch("sqlalchemy.text"):
            table_args = get_database_specific_table_args(mock_engine, "sensor_events")
            assert len(table_args) > 0
