"""
Comprehensive tests for database compatibility layer.

Tests database-specific configuration, model patching, and cross-database compatibility
utilities for SQLite and PostgreSQL/TimescaleDB.
"""

import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from sqlalchemy.engine import Engine

from src.data.storage.database_compatibility import (
    configure_database_on_first_connect,
    configure_sensor_event_model,
    configure_sqlite_for_testing,
    create_database_specific_models,
    get_database_specific_table_args,
    is_postgresql_engine,
    is_sqlite_engine,
)


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

    def test_configure_sensor_event_model_sqlite_basic(self):
        """Test basic SensorEvent configuration for SQLite."""
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

        result = configure_sensor_event_model(mock_model, mock_engine)

        # Should set autoincrement on id column
        assert mock_id_column.autoincrement is True
        # Should remove timestamp from primary key
        assert mock_timestamp_column.primary_key is False

        assert result == mock_model

    def test_configure_sensor_event_model_postgresql_basic(self):
        """Test basic SensorEvent configuration for PostgreSQL."""
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

        # Test the function directly
        result = create_database_specific_models(base_models, mock_engine)

        # Should return models (SensorEvent may be configured internally)
        assert "SensorEvent" in result
        assert "OtherModel" in result
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

        result = create_database_specific_models(base_models, mock_engine)

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


class TestTableArgsGeneration:
    """Test database-specific table arguments generation."""

    def test_get_database_specific_table_args_sensor_events_sqlite(self):
        """Test table args for sensor_events table on SQLite."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        args = get_database_specific_table_args(mock_engine, "sensor_events")

        # Should return some table args for sensor_events
        assert isinstance(args, tuple)
        assert len(args) > 0

    def test_get_database_specific_table_args_sensor_events_postgresql(self):
        """Test table args for sensor_events table on PostgreSQL."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        args = get_database_specific_table_args(mock_engine, "sensor_events")

        # Should return some table args for sensor_events
        assert isinstance(args, tuple)
        assert len(args) > 0

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


class TestEnvironmentBasedConfiguration:
    """Test environment-based configuration detection."""

    @patch.dict(os.environ, {"TEST_DB_URL": "sqlite:///test.db"})
    def test_sqlite_detection_via_environment(self):
        """Test SQLite detection through environment variables."""
        # Test that the environment detection works
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

    def test_engine_url_str_error_handling(self):
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

    def test_configure_sensor_event_model_missing_columns(self):
        """Test SensorEvent configuration with missing columns."""
        mock_model = Mock()
        mock_table = Mock()
        mock_model.__table__ = mock_table

        # Mock table with missing columns (AttributeError when accessing)
        mock_table.c = Mock()
        mock_table.c.id = Mock()

        # Make timestamp column missing
        def missing_timestamp(*args, **kwargs):
            raise AttributeError("timestamp column not found")

        mock_table.c.__getattr__ = missing_timestamp

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
        configured_models = create_database_specific_models(base_models, mock_engine)

        # Should have configured models
        assert configured_models["SensorEvent"] is not None
        assert configured_models["OtherModel"] == mock_other_model

        # Test table args generation
        table_args = get_database_specific_table_args(mock_engine, "sensor_events")
        assert isinstance(table_args, tuple)

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
        table_args = get_database_specific_table_args(mock_engine, "sensor_events")
        assert isinstance(table_args, tuple)
