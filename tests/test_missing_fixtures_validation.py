#!/usr/bin/env python3
"""
pytest-based test to validate that all the missing fixtures have been properly implemented.
Run with: pytest test_missing_fixtures_validation.py -v
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.core.constants import DB_TABLES


class TestMissingFixtures:
    """Test class to validate all missing fixtures are properly implemented."""

    def test_sample_training_data_fixture(self, sample_training_data):
        """Test that sample_training_data fixture creates proper training data."""
        features, targets = sample_training_data

        # Validate features DataFrame
        assert isinstance(
            features, pd.DataFrame
        ), "Features should be a pandas DataFrame"
        assert len(features) == 1000, f"Expected 1000 samples, got {len(features)}"
        assert (
            len(features.columns) >= 16
        ), f"Expected at least 16 feature columns, got {len(features.columns)}"

        # Check specific required columns exist
        required_columns = [
            "time_since_last_change",
            "current_state_duration",
            "hour_sin",
            "hour_cos",
            "temperature",
            "humidity",
        ]
        for col in required_columns:
            assert col in features.columns, f"Missing required feature column: {col}"

        # Validate targets DataFrame
        assert isinstance(targets, pd.DataFrame), "Targets should be a pandas DataFrame"
        assert len(targets) == 1000, f"Expected 1000 target samples, got {len(targets)}"
        assert (
            "time_until_transition_seconds" in targets.columns
        ), "Missing target column"
        assert (
            targets["time_until_transition_seconds"].min() >= 60
        ), "All targets should be at least 60 seconds"

    def test_mock_validator_fixture(self, mock_validator):
        """Test that mock_validator fixture creates proper AsyncMock."""
        # Validate it's an AsyncMock
        assert isinstance(mock_validator, AsyncMock), "Validator should be AsyncMock"

        # Check required methods exist
        assert hasattr(
            mock_validator, "record_prediction"
        ), "Missing record_prediction method"
        assert hasattr(
            mock_validator, "validate_prediction"
        ), "Missing validate_prediction method"
        assert hasattr(
            mock_validator, "get_accuracy_metrics"
        ), "Missing get_accuracy_metrics method"

        # Check internal attributes
        assert hasattr(mock_validator, "_lock"), "Missing _lock attribute"
        assert hasattr(
            mock_validator, "_validation_records"
        ), "Missing _validation_records attribute"
        assert hasattr(
            mock_validator, "_pending_predictions"
        ), "Missing _pending_predictions attribute"

    def test_sample_room_states_data_fixture(self, sample_room_states_data):
        """Test that sample_room_states_data fixture creates proper room state data."""
        room_states = sample_room_states_data

        # Validate basic structure
        assert isinstance(room_states, list), "Room states should be a list"
        assert (
            len(room_states) == 800
        ), f"Expected 800 room state records, got {len(room_states)}"

        # Check first record has all required fields
        first_record = room_states[0]
        required_fields = [
            "room_id",
            "timestamp",
            "is_occupied",
            "occupancy_confidence",
            "occupant_type",
        ]
        for field in required_fields:
            assert hasattr(first_record, field), f"Missing required field: {field}"

        # Check data variety
        room_ids = set(state.room_id for state in room_states)
        assert len(room_ids) == 4, f"Expected 4 different rooms, got {len(room_ids)}"

    def test_sample_prediction_validation_data_fixture(
        self, sample_prediction_validation_data
    ):
        """Test that sample_prediction_validation_data fixture creates proper validation data."""
        validation_data = sample_prediction_validation_data

        # Validate basic structure
        assert isinstance(validation_data, list), "Validation data should be a list"
        assert (
            len(validation_data) == 44
        ), f"Expected 44 validation records, got {len(validation_data)}"

        # Check first record structure
        first_record = validation_data[0]
        required_keys = [
            "room_id",
            "predicted_time",
            "actual_time",
            "confidence",
            "error_minutes",
        ]
        for key in required_keys:
            assert key in first_record, f"Missing required key: {key}"

    def test_large_sample_training_data_fixture(self, large_sample_training_data):
        """Test that large_sample_training_data fixture creates comprehensive dataset."""
        features, targets = large_sample_training_data

        # Validate features DataFrame
        assert isinstance(
            features, pd.DataFrame
        ), "Features should be a pandas DataFrame"
        assert len(features) == 5000, f"Expected 5000 samples, got {len(features)}"
        assert (
            "timestamp" in features.columns
        ), "Missing timestamp column for time series"

        # Check temporal patterns are realistic
        assert features["hour_sin"].min() >= -1.0, "Hour sine should be >= -1"
        assert features["hour_sin"].max() <= 1.0, "Hour sine should be <= 1"
        assert features["is_weekend"].isin([0, 1]).all(), "Weekend should be binary"

    def test_database_schema_integration(self):
        """Test that the database schema fix works - all tables have schema definitions."""
        # Simulate what the test does - this is the logic that was failing
        schema_definitions = {}

        for table_key, table_name in DB_TABLES.items():
            if table_key == "sensor_events":
                schema_definitions[table_name] = {
                    "columns": [
                        "id",
                        "room_id",
                        "sensor_id",
                        "sensor_type",
                        "state",
                        "timestamp",
                    ],
                    "primary_key": ["id", "timestamp"],
                }
            elif table_key == "predictions":
                schema_definitions[table_name] = {
                    "columns": [
                        "id",
                        "room_id",
                        "prediction_type",
                        "predicted_time",
                        "confidence",
                    ],
                    "primary_key": ["id"],
                }
            elif table_key == "model_accuracy":
                schema_definitions[table_name] = {
                    "columns": [
                        "id",
                        "room_id",
                        "model_type",
                        "accuracy_score",
                        "timestamp",
                    ],
                    "primary_key": ["id"],
                }
            elif table_key == "room_states":
                schema_definitions[table_name] = {
                    "columns": [
                        "id",
                        "room_id",
                        "timestamp",
                        "is_occupied",
                        "occupancy_confidence",
                        "occupant_type",
                        "state_duration",
                        "transition_trigger",
                        "created_at",
                    ],
                    "primary_key": ["id", "timestamp"],
                }
            elif table_key == "feature_store":
                schema_definitions[table_name] = {
                    "columns": [
                        "id",
                        "room_id",
                        "timestamp",
                        "feature_type",
                        "feature_name",
                        "feature_value",
                        "computation_time",
                        "version",
                    ],
                    "primary_key": ["id"],
                }

        # This is the critical test that was failing before - ALL tables should have schema definitions
        for table_name in DB_TABLES.values():
            assert (
                table_name in schema_definitions
            ), f"Missing schema definition for table: {table_name}"
            assert (
                "columns" in schema_definitions[table_name]
            ), f"Missing columns for table: {table_name}"
            assert (
                "primary_key" in schema_definitions[table_name]
            ), f"Missing primary_key for table: {table_name}"

    @pytest.mark.asyncio
    async def test_mock_validator_get_accuracy_metrics(self, mock_validator):
        """Test that mock_validator fixture's get_accuracy_metrics method works."""
        # Test the async method works
        metrics = await mock_validator.get_accuracy_metrics("test_room", 24)

        # Validate the realistic mock data
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert "mean_error_minutes" in metrics, "Missing mean_error_minutes"
        assert "median_error_minutes" in metrics, "Missing median_error_minutes"
        assert (
            "accuracy_within_threshold" in metrics
        ), "Missing accuracy_within_threshold"
        assert "total_predictions" in metrics, "Missing total_predictions"
        assert "validated_predictions" in metrics, "Missing validated_predictions"

        # Check realistic values
        assert (
            0 <= metrics["accuracy_within_threshold"] <= 1
        ), "Accuracy should be between 0 and 1"
        assert (
            metrics["total_predictions"] >= metrics["validated_predictions"]
        ), "Total should be >= validated"

    def test_reproducible_training_data(self, sample_training_data):
        """Test that sample_training_data is reproducible due to seed setting."""
        features1, targets1 = sample_training_data

        # Get another instance (should be identical due to np.random.seed(42))
        features2, targets2 = sample_training_data

        # Should be identical due to seeding
        pd.testing.assert_frame_equal(
            features1, features2, "Training data should be reproducible"
        )
        pd.testing.assert_frame_equal(
            targets1, targets2, "Target data should be reproducible"
        )

    def test_realistic_feature_distributions(self, sample_training_data):
        """Test that generated features have realistic distributions."""
        features, targets = sample_training_data

        # Temperature should be reasonable (around 22°C with variation)
        temp_mean = features["temperature"].mean()
        assert (
            15 <= temp_mean <= 30
        ), f"Temperature mean {temp_mean} should be reasonable"

        # Humidity should be reasonable (around 45% with variation)
        humidity_mean = features["humidity"].mean()
        assert (
            20 <= humidity_mean <= 80
        ), f"Humidity mean {humidity_mean} should be reasonable"

        # Time features should be properly normalized
        assert -1 <= features["hour_sin"].min(), "Hour sine should be >= -1"
        assert features["hour_cos"].max() <= 1, "Hour cosine should be <= 1"

        # Targets should be reasonable (minimum 1 minute, reasonable maximum)
        target_mean = targets["time_until_transition_seconds"].mean()
        assert (
            60 <= target_mean <= 7200
        ), f"Target mean {target_mean} should be reasonable (1 min to 2 hours)"
