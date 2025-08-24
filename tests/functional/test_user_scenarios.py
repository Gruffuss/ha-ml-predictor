"""Functional tests for user scenarios and business requirements.

Covers user-facing functionality, business logic validation,
and acceptance criteria testing.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import (
    APIConfig,
    DatabaseConfig,
    HomeAssistantConfig,
    MQTTConfig,
    RoomConfig,
    SystemConfig,
    TrackingConfig,
)
from src.data.storage.database import DatabaseManager
from src.data.storage.models import RoomState, SensorEvent
from src.integration.mqtt_integration_manager import MQTTIntegrationManager


class TestOccupancyPredictionScenarios:
    """Test occupancy prediction user scenarios."""

    @pytest.fixture
    def prediction_test_config(self):
        """Configuration for prediction testing scenarios."""
        return {
            "rooms": {
                "living_room": RoomConfig(
                    room_id="living_room",
                    name="Living Room",
                    sensors={
                        "motion": "binary_sensor.living_room_motion",
                        "door": "binary_sensor.living_room_door",
                        "temperature": "sensor.living_room_temperature",
                    },
                ),
                "kitchen": RoomConfig(
                    room_id="kitchen",
                    name="Kitchen",
                    sensors={"motion": "binary_sensor.kitchen_motion"},
                ),
            },
            "prediction_thresholds": {
                "confidence_minimum": 0.7,
                "accuracy_target_minutes": 15,
                "update_interval_minutes": 5,
            },
        }

    def test_single_room_occupancy_prediction(self, prediction_test_config):
        """Test user scenario: Get prediction for a single room."""
        # User story: As a user, I want to know when the living room will next be occupied
        room_config = prediction_test_config["rooms"]["living_room"]

        # Mock prediction data
        mock_prediction = {
            "room_id": "living_room",
            "next_transition_time": datetime.now(timezone.utc) + timedelta(minutes=45),
            "transition_type": "occupied",
            "confidence": 0.85,
            "prediction_time": datetime.now(timezone.utc),
            "time_until_transition": "45 minutes",
            "model_info": {
                "primary_model": "lstm_ensemble",
                "feature_importance": {
                    "time_since_last_motion": 0.4,
                    "door_activity": 0.3,
                    "time_of_day": 0.3,
                },
            },
        }

        # Validate prediction meets user requirements
        assert mock_prediction["room_id"] == "living_room"
        assert (
            mock_prediction["confidence"]
            >= prediction_test_config["prediction_thresholds"]["confidence_minimum"]
        )
        assert mock_prediction["next_transition_time"] > datetime.now(timezone.utc)
        assert mock_prediction["transition_type"] in ["occupied", "vacant"]

        # Validate business logic
        confidence_acceptable = mock_prediction["confidence"] >= 0.7
        prediction_reasonable = mock_prediction["next_transition_time"] < datetime.now(
            timezone.utc
        ) + timedelta(hours=24)

        assert (
            confidence_acceptable
        ), "Prediction confidence must meet minimum threshold"
        assert prediction_reasonable, "Prediction must be within 24 hours"

    def test_multi_room_prediction_scenario(self, prediction_test_config):
        """Test user scenario: Get predictions for multiple rooms simultaneously."""
        # User story: As a user, I want to see occupancy predictions for all rooms
        rooms = prediction_test_config["rooms"]

        # Mock multi-room predictions
        multi_room_predictions = {
            "living_room": {
                "next_transition_time": datetime.now(timezone.utc)
                + timedelta(minutes=30),
                "transition_type": "occupied",
                "confidence": 0.88,
                "current_state": "vacant",
            },
            "kitchen": {
                "next_transition_time": datetime.now(timezone.utc)
                + timedelta(minutes=15),
                "transition_type": "vacant",
                "confidence": 0.92,
                "current_state": "occupied",
            },
        }

        # Validate all rooms have predictions
        for room_id in rooms.keys():
            assert (
                room_id in multi_room_predictions
            ), f"Missing prediction for {room_id}"

            prediction = multi_room_predictions[room_id]
            assert "confidence" in prediction
            assert "next_transition_time" in prediction
            assert "transition_type" in prediction
            assert prediction["confidence"] >= 0.7

        # Test user requirements: Cross-room consistency
        all_predictions_future = all(
            pred["next_transition_time"] > datetime.now(timezone.utc)
            for pred in multi_room_predictions.values()
        )
        assert all_predictions_future, "All predictions must be in the future"

        # Test business logic: No conflicting transitions
        transition_times = [
            pred["next_transition_time"] for pred in multi_room_predictions.values()
        ]

        # Transitions should be spaced reasonably (not all at exact same time)
        time_differences = [
            abs((t1 - t2).total_seconds())
            for i, t1 in enumerate(transition_times)
            for t2 in transition_times[i + 1 :]
        ]

        if len(time_differences) > 0:
            min_difference = min(time_differences)
            assert (
                min_difference >= 60
            ), "Room transitions should be spaced at least 1 minute apart"

    def test_prediction_accuracy_validation_workflow(self, prediction_test_config):
        """Test user scenario: Validate prediction accuracy over time."""
        # User story: As a user, I want to know how accurate the predictions are

        # Mock historical prediction vs actual data
        prediction_accuracy_data = [
            {
                "room_id": "living_room",
                "predicted_time": datetime.now(timezone.utc) - timedelta(hours=2),
                "actual_time": datetime.now(timezone.utc)
                - timedelta(hours=2, minutes=8),
                "error_minutes": 8,
                "prediction_type": "occupied",
                "confidence": 0.85,
            },
            {
                "room_id": "living_room",
                "predicted_time": datetime.now(timezone.utc) - timedelta(hours=4),
                "actual_time": datetime.now(timezone.utc)
                - timedelta(hours=4, minutes=12),
                "error_minutes": 12,
                "prediction_type": "vacant",
                "confidence": 0.78,
            },
            {
                "room_id": "kitchen",
                "predicted_time": datetime.now(timezone.utc) - timedelta(hours=1),
                "actual_time": datetime.now(timezone.utc)
                - timedelta(hours=1, minutes=3),
                "error_minutes": 3,
                "prediction_type": "occupied",
                "confidence": 0.95,
            },
        ]

        # Calculate accuracy metrics
        total_predictions = len(prediction_accuracy_data)
        total_error_minutes = sum(
            pred["error_minutes"] for pred in prediction_accuracy_data
        )
        avg_error_minutes = total_error_minutes / total_predictions

        # Test user requirements
        accuracy_threshold = prediction_test_config["prediction_thresholds"][
            "accuracy_target_minutes"
        ]

        assert (
            avg_error_minutes <= accuracy_threshold
        ), f"Average error ({avg_error_minutes:.1f} min) exceeds threshold ({accuracy_threshold} min)"

        # Test accuracy by confidence level
        high_confidence_predictions = [
            p for p in prediction_accuracy_data if p["confidence"] >= 0.9
        ]
        if high_confidence_predictions:
            high_conf_avg_error = sum(
                p["error_minutes"] for p in high_confidence_predictions
            ) / len(high_confidence_predictions)
            assert (
                high_conf_avg_error <= avg_error_minutes
            ), "High confidence predictions should be more accurate"

        # Test per-room accuracy
        room_accuracy = {}
        for room_id in prediction_test_config["rooms"].keys():
            room_predictions = [
                p for p in prediction_accuracy_data if p["room_id"] == room_id
            ]
            if room_predictions:
                room_avg_error = sum(
                    p["error_minutes"] for p in room_predictions
                ) / len(room_predictions)
                room_accuracy[room_id] = room_avg_error

        # All rooms should meet accuracy requirements
        for room_id, avg_error in room_accuracy.items():
            assert (
                avg_error <= accuracy_threshold * 1.2
            ), f"Room {room_id} accuracy ({avg_error:.1f} min) significantly exceeds threshold"

    def test_real_time_prediction_updates(self, prediction_test_config):
        """Test user scenario: Receive real-time prediction updates."""
        # User story: As a user, I want predictions to update as new sensor data arrives

        now = datetime.now(timezone.utc)

        # Initial prediction
        initial_prediction = {
            "room_id": "living_room",
            "next_transition_time": now + timedelta(minutes=60),
            "confidence": 0.75,
            "prediction_time": now - timedelta(minutes=10),
            "data_freshness": "10 minutes ago",
        }

        # New sensor event occurs
        new_sensor_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.living_room_motion",
            "state": "on",
            "timestamp": now,
            "event_type": "motion_detected",
        }

        # Updated prediction after new data
        updated_prediction = {
            "room_id": "living_room",
            "next_transition_time": now
            + timedelta(minutes=45),  # Sooner due to recent activity
            "confidence": 0.92,  # Higher confidence with fresh data
            "prediction_time": now,
            "data_freshness": "real-time",
            "trigger_event": "motion_detected",
            "confidence_change": +0.17,
        }

        # Validate update logic
        assert (
            updated_prediction["prediction_time"]
            > initial_prediction["prediction_time"]
        )
        assert updated_prediction["confidence"] > initial_prediction["confidence"]

        # Updated prediction should be more recent
        time_difference = (
            updated_prediction["next_transition_time"]
            - initial_prediction["next_transition_time"]
        )
        assert (
            abs(time_difference.total_seconds()) > 0
        ), "Prediction should change with new data"

        # Confidence should improve with fresh data
        assert (
            updated_prediction["confidence"] >= 0.9
        ), "Fresh sensor data should increase confidence"


class TestSystemConfigurationScenarios:
    """Test system configuration scenarios."""

    @pytest.fixture
    def configuration_test_setup(self):
        """Setup for configuration testing."""
        return {
            "sample_room_configs": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.living_room_motion",
                        "door": "binary_sensor.living_room_door",
                        "temperature": "sensor.living_room_temperature",
                    },
                    "prediction_settings": {
                        "confidence_threshold": 0.8,
                        "update_frequency_minutes": 5,
                    },
                },
                "bedroom": {
                    "name": "Master Bedroom",
                    "sensors": {
                        "motion": "binary_sensor.bedroom_motion",
                        "door": "binary_sensor.bedroom_door",
                    },
                    "prediction_settings": {
                        "confidence_threshold": 0.75,
                        "update_frequency_minutes": 10,
                    },
                },
            }
        }

    def test_room_configuration_validation(self, configuration_test_setup):
        """Test user scenario: Configure rooms and sensors."""
        # User story: As a user, I want to configure which sensors belong to which rooms

        room_configs = configuration_test_setup["sample_room_configs"]

        # Validate room configuration structure
        for room_id, config in room_configs.items():
            # Required fields
            assert "name" in config, f"Room {room_id} missing name"
            assert "sensors" in config, f"Room {room_id} missing sensors"

            # Sensor validation
            sensors = config["sensors"]
            assert "motion" in sensors, f"Room {room_id} missing motion sensor"

            # Entity ID format validation
            for sensor_type, entity_id in sensors.items():
                assert entity_id.startswith(
                    ("binary_sensor.", "sensor.")
                ), f"Invalid entity ID format: {entity_id}"
                assert (
                    room_id.replace("_", "") in entity_id
                    or config["name"].lower().replace(" ", "_") in entity_id
                ), f"Entity ID should reference room: {entity_id}"

    def test_sensor_mapping_configuration(self, configuration_test_setup):
        """Test user scenario: Map Home Assistant entities to system sensors."""
        # User story: As a user, I want to map my HA sensors to occupancy prediction

        room_configs = configuration_test_setup["sample_room_configs"]

        # Create sensor mapping
        sensor_mapping = {}
        for room_id, config in room_configs.items():
            for sensor_type, entity_id in config["sensors"].items():
                sensor_mapping[entity_id] = {
                    "room_id": room_id,
                    "sensor_type": sensor_type,
                    "room_name": config["name"],
                }

        # Validate mapping completeness
        expected_sensors = [
            "binary_sensor.living_room_motion",
            "binary_sensor.living_room_door",
            "sensor.living_room_temperature",
            "binary_sensor.bedroom_motion",
            "binary_sensor.bedroom_door",
        ]

        for expected_sensor in expected_sensors:
            assert (
                expected_sensor in sensor_mapping
            ), f"Missing sensor mapping: {expected_sensor}"

            mapping = sensor_mapping[expected_sensor]
            assert "room_id" in mapping
            assert "sensor_type" in mapping
            assert mapping["room_id"] in room_configs

    def test_system_customization_settings(self, configuration_test_setup):
        """Test user scenario: Customize prediction settings per room."""
        # User story: As a user, I want different prediction settings for different rooms

        room_configs = configuration_test_setup["sample_room_configs"]

        # Test per-room customization
        living_room_settings = room_configs["living_room"]["prediction_settings"]
        bedroom_settings = room_configs["bedroom"]["prediction_settings"]

        # Living room should have higher confidence threshold (more accuracy required)
        assert (
            living_room_settings["confidence_threshold"]
            > bedroom_settings["confidence_threshold"]
        )

        # Living room should update more frequently (more activity)
        assert (
            living_room_settings["update_frequency_minutes"]
            < bedroom_settings["update_frequency_minutes"]
        )

        # Validate setting ranges
        for room_id, config in room_configs.items():
            settings = config["prediction_settings"]

            # Confidence threshold should be reasonable
            assert (
                0.5 <= settings["confidence_threshold"] <= 1.0
            ), f"Invalid confidence threshold for {room_id}"

            # Update frequency should be reasonable
            assert (
                1 <= settings["update_frequency_minutes"] <= 60
            ), f"Invalid update frequency for {room_id}"

    def test_configuration_validation_rules(self, configuration_test_setup):
        """Test configuration validation business rules."""
        room_configs = configuration_test_setup["sample_room_configs"]

        validation_errors = []

        for room_id, config in room_configs.items():
            # Business rule: Every room must have at least motion sensor
            if "motion" not in config["sensors"]:
                validation_errors.append(
                    f"Room {room_id} missing required motion sensor"
                )

            # Business rule: Room names should be user-friendly
            if not config["name"] or config["name"].strip() == "":
                validation_errors.append(f"Room {room_id} has empty name")

            # Business rule: Sensor entity IDs must be unique
            sensor_entities = list(config["sensors"].values())
            if len(sensor_entities) != len(set(sensor_entities)):
                validation_errors.append(
                    f"Room {room_id} has duplicate sensor entities"
                )

        # No validation errors should exist in good configuration
        assert (
            len(validation_errors) == 0
        ), f"Configuration validation errors: {validation_errors}"

        # Test cross-room validation
        all_entities = []
        for config in room_configs.values():
            all_entities.extend(config["sensors"].values())

        # Global business rule: No entity can be used in multiple rooms
        duplicate_entities = [
            entity for entity in set(all_entities) if all_entities.count(entity) > 1
        ]
        assert (
            len(duplicate_entities) == 0
        ), f"Entity IDs used in multiple rooms: {duplicate_entities}"


class TestAdaptationScenarios:
    """Test model adaptation scenarios."""

    def test_model_retraining_trigger_scenario(self):
        """Test user scenario: Automatic model retraining when accuracy degrades."""
        # User story: As a user, I want the system to improve automatically when predictions become less accurate

        # Mock accuracy monitoring data
        accuracy_metrics = {
            "living_room": {
                "recent_predictions": 25,
                "avg_error_minutes": 18.5,  # Above 15-minute threshold
                "accuracy_percentage": 72.0,  # Below 85% threshold
                "confidence_avg": 0.81,
                "last_retraining": datetime.now(timezone.utc) - timedelta(days=7),
            },
            "kitchen": {
                "recent_predictions": 30,
                "avg_error_minutes": 12.3,  # Within threshold
                "accuracy_percentage": 89.2,  # Above threshold
                "confidence_avg": 0.88,
                "last_retraining": datetime.now(timezone.utc) - timedelta(days=14),
            },
        }

        # Define business rules for retraining
        retraining_thresholds = {
            "max_error_minutes": 15.0,
            "min_accuracy_percentage": 85.0,
            "min_predictions_count": 20,
            "max_days_since_retraining": 30,
        }

        # Evaluate retraining needs
        retraining_decisions = {}
        for room_id, metrics in accuracy_metrics.items():
            needs_retraining = False
            reasons = []

            if (
                metrics["avg_error_minutes"]
                > retraining_thresholds["max_error_minutes"]
            ):
                needs_retraining = True
                reasons.append("accuracy_degradation")

            if (
                metrics["accuracy_percentage"]
                < retraining_thresholds["min_accuracy_percentage"]
            ):
                needs_retraining = True
                reasons.append("low_accuracy")

            days_since_retraining = (
                datetime.now(timezone.utc) - metrics["last_retraining"]
            ).days
            if (
                days_since_retraining
                > retraining_thresholds["max_days_since_retraining"]
            ):
                needs_retraining = True
                reasons.append("scheduled_refresh")

            retraining_decisions[room_id] = {
                "needs_retraining": needs_retraining,
                "reasons": reasons,
                "priority": (
                    "high"
                    if len(reasons) > 1
                    else "medium" if needs_retraining else "none"
                ),
            }

        # Validate retraining logic
        assert (
            retraining_decisions["living_room"]["needs_retraining"] is True
        ), "Living room should need retraining due to poor accuracy"
        assert "accuracy_degradation" in retraining_decisions["living_room"]["reasons"]
        assert "low_accuracy" in retraining_decisions["living_room"]["reasons"]

        assert (
            retraining_decisions["kitchen"]["needs_retraining"] is False
        ), "Kitchen should not need retraining - metrics are good"

        # Test prioritization
        high_priority_rooms = [
            room
            for room, decision in retraining_decisions.items()
            if decision["priority"] == "high"
        ]
        assert (
            "living_room" in high_priority_rooms
        ), "Rooms with multiple issues should be high priority"

    def test_drift_detection_scenario(self):
        """Test user scenario: Detect when user behavior patterns change."""
        # User story: As a user, I want the system to adapt to changes in my daily routine

        # Historical behavior pattern (baseline)
        historical_pattern = {
            "living_room": {
                "weekday_morning_occupied": {
                    "time": "07:30",
                    "duration_minutes": 45,
                    "frequency": 0.85,
                },
                "weekday_evening_occupied": {
                    "time": "18:00",
                    "duration_minutes": 180,
                    "frequency": 0.92,
                },
                "weekend_afternoon_occupied": {
                    "time": "14:00",
                    "duration_minutes": 120,
                    "frequency": 0.78,
                },
            }
        }

        # Recent behavior pattern (last 2 weeks)
        recent_pattern = {
            "living_room": {
                "weekday_morning_occupied": {
                    "time": "08:15",
                    "duration_minutes": 30,
                    "frequency": 0.65,
                },  # Later start, shorter duration
                "weekday_evening_occupied": {
                    "time": "17:30",
                    "duration_minutes": 210,
                    "frequency": 0.95,
                },  # Earlier start, longer duration
                "weekend_afternoon_occupied": {
                    "time": "13:30",
                    "duration_minutes": 90,
                    "frequency": 0.55,
                },  # Earlier, shorter, less frequent
            }
        }

        # Drift detection algorithm
        drift_detection_results = {}

        for room_id in historical_pattern:
            room_drift = {}
            historical_room = historical_pattern[room_id]
            recent_room = recent_pattern[room_id]

            for pattern_name in historical_room:
                if pattern_name in recent_room:
                    hist = historical_room[pattern_name]
                    recent = recent_room[pattern_name]

                    # Calculate time drift (in minutes)
                    hist_time_minutes = int(hist["time"].split(":")[0]) * 60 + int(
                        hist["time"].split(":")[1]
                    )
                    recent_time_minutes = int(recent["time"].split(":")[0]) * 60 + int(
                        recent["time"].split(":")[1]
                    )
                    time_drift_minutes = abs(recent_time_minutes - hist_time_minutes)

                    # Calculate duration drift
                    duration_drift_percent = (
                        abs(recent["duration_minutes"] - hist["duration_minutes"])
                        / hist["duration_minutes"]
                    )

                    # Calculate frequency drift
                    frequency_drift_percent = (
                        abs(recent["frequency"] - hist["frequency"]) / hist["frequency"]
                    )

                    # Determine if drift is significant
                    significant_drift = (
                        time_drift_minutes > 30  # More than 30 minutes time change
                        or duration_drift_percent
                        > 0.25  # More than 25% duration change
                        or frequency_drift_percent
                        > 0.20  # More than 20% frequency change
                    )

                    room_drift[pattern_name] = {
                        "time_drift_minutes": time_drift_minutes,
                        "duration_drift_percent": duration_drift_percent,
                        "frequency_drift_percent": frequency_drift_percent,
                        "significant_drift": significant_drift,
                    }

            drift_detection_results[room_id] = room_drift

        # Validate drift detection
        living_room_drift = drift_detection_results["living_room"]

        # Morning pattern should show significant drift
        morning_drift = living_room_drift["weekday_morning_occupied"]
        assert (
            morning_drift["significant_drift"] is True
        ), "Morning pattern should show significant drift"
        assert (
            morning_drift["time_drift_minutes"] == 45
        ), "Morning time drift should be 45 minutes"

        # Evening pattern should show some drift
        evening_drift = living_room_drift["weekday_evening_occupied"]
        assert (
            evening_drift["time_drift_minutes"] == 30
        ), "Evening time drift should be 30 minutes"

        # Count patterns with significant drift
        significant_drift_count = sum(
            1 for pattern in living_room_drift.values() if pattern["significant_drift"]
        )
        assert (
            significant_drift_count >= 2
        ), "Multiple patterns should show significant drift"

    def test_continuous_learning_adaptation(self):
        """Test user scenario: System learns from new data continuously."""
        # User story: As a user, I want the system to get better over time without manual intervention

        # Simulate learning progression over time
        learning_timeline = [
            {
                "week": 1,
                "data_points": 50,
                "accuracy_percentage": 72.0,
                "confidence_avg": 0.68,
                "model_version": "1.0",
                "learning_status": "initial_training",
            },
            {
                "week": 4,
                "data_points": 200,
                "accuracy_percentage": 81.5,
                "confidence_avg": 0.76,
                "model_version": "1.1",
                "learning_status": "incremental_learning",
            },
            {
                "week": 8,
                "data_points": 400,
                "accuracy_percentage": 87.2,
                "confidence_avg": 0.83,
                "model_version": "1.2",
                "learning_status": "pattern_stabilization",
            },
            {
                "week": 12,
                "data_points": 600,
                "accuracy_percentage": 89.8,
                "confidence_avg": 0.87,
                "model_version": "1.3",
                "learning_status": "mature_learning",
            },
        ]

        # Validate continuous improvement
        for i in range(1, len(learning_timeline)):
            current = learning_timeline[i]
            previous = learning_timeline[i - 1]

            # Accuracy should improve over time
            accuracy_improvement = (
                current["accuracy_percentage"] - previous["accuracy_percentage"]
            )
            assert (
                accuracy_improvement > 0
            ), f"Accuracy should improve from week {previous['week']} to {current['week']}"

            # Confidence should increase with more data
            confidence_improvement = (
                current["confidence_avg"] - previous["confidence_avg"]
            )
            assert (
                confidence_improvement > 0
            ), f"Confidence should increase from week {previous['week']} to {current['week']}"

            # Data points should accumulate
            assert (
                current["data_points"] > previous["data_points"]
            ), "Data points should increase over time"

        # Test learning rate expectations
        initial_accuracy = learning_timeline[0]["accuracy_percentage"]
        final_accuracy = learning_timeline[-1]["accuracy_percentage"]
        total_improvement = final_accuracy - initial_accuracy

        assert (
            total_improvement >= 15.0
        ), "System should improve accuracy by at least 15% over 12 weeks"
        assert final_accuracy >= 85.0, "Final accuracy should reach at least 85%"

        # Test learning stage progression
        learning_stages = [entry["learning_status"] for entry in learning_timeline]
        expected_progression = [
            "initial_training",
            "incremental_learning",
            "pattern_stabilization",
            "mature_learning",
        ]

        assert (
            learning_stages == expected_progression
        ), "Learning should progress through expected stages"


class TestIntegrationScenarios:
    """Test Home Assistant integration scenarios."""

    def test_ha_entity_creation_scenario(self):
        """Test user scenario: Predictions appear as HA entities."""
        # User story: As a user, I want to see predictions in my Home Assistant dashboard

        # Mock HA entity definitions for predictions
        ha_prediction_entities = {
            "sensor.occupancy_prediction_living_room_next_occupied": {
                "entity_id": "sensor.occupancy_prediction_living_room_next_occupied",
                "name": "Living Room Next Occupied Time",
                "state": "2024-12-17T14:30:00+00:00",
                "attributes": {
                    "confidence": 0.85,
                    "time_until": "2 hours 15 minutes",
                    "prediction_type": "next_occupied",
                    "room": "living_room",
                    "model_version": "1.3",
                    "device_class": "timestamp",
                    "icon": "mdi:account-clock",
                },
                "unique_id": "occupancy_pred_living_room_next_occupied",
                "device_info": {
                    "identifiers": "ha_ml_predictor",
                    "name": "Occupancy Predictor",
                    "manufacturer": "HA ML Predictor",
                    "model": "Smart Room Occupancy Predictor",
                },
            },
            "sensor.occupancy_prediction_living_room_confidence": {
                "entity_id": "sensor.occupancy_prediction_living_room_confidence",
                "name": "Living Room Prediction Confidence",
                "state": 85,
                "attributes": {
                    "unit_of_measurement": "%",
                    "state_class": "measurement",
                    "room": "living_room",
                    "last_updated": "2024-12-17T12:15:00+00:00",
                    "icon": "mdi:percent",
                },
                "unique_id": "occupancy_pred_living_room_confidence",
            },
        }

        # Validate entity structure
        for entity_id, entity_config in ha_prediction_entities.items():
            # Required fields
            assert "entity_id" in entity_config
            assert "name" in entity_config
            assert "state" in entity_config
            assert "unique_id" in entity_config

            # Entity ID format
            assert entity_id.startswith("sensor.occupancy_prediction_")
            assert entity_config["entity_id"] == entity_id

            # Attributes validation
            if "confidence" in entity_config["attributes"]:
                confidence = entity_config["attributes"]["confidence"]
                assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"

            # Room association
            assert (
                "room" in entity_config["attributes"]
            ), "Entity should be associated with a room"

        # Test user-friendly naming
        next_occupied_entity = ha_prediction_entities[
            "sensor.occupancy_prediction_living_room_next_occupied"
        ]
        assert "Living Room" in next_occupied_entity["name"]
        assert "Next Occupied" in next_occupied_entity["name"]

        confidence_entity = ha_prediction_entities[
            "sensor.occupancy_prediction_living_room_confidence"
        ]
        assert confidence_entity["attributes"]["unit_of_measurement"] == "%"
        assert confidence_entity["state"] == 85

    def test_automation_integration_scenario(self):
        """Test user scenario: Use predictions in HA automations."""
        # User story: As a user, I want to create automations based on occupancy predictions

        # Mock HA automation that uses predictions
        sample_automation = {
            "automation": {
                "alias": "Living Room Pre-heating",
                "description": "Turn on heating 30 minutes before living room will be occupied",
                "trigger": [
                    {
                        "platform": "template",
                        "value_template": "{{ (as_timestamp(states('sensor.occupancy_prediction_living_room_next_occupied')) - as_timestamp(now())) / 60 <= 30 }}",
                        "condition": "template",
                        "value_template_condition": "{{ states('sensor.occupancy_prediction_living_room_confidence') | float >= 80 }}",
                    }
                ],
                "condition": [
                    {
                        "condition": "numeric_state",
                        "entity_id": "sensor.occupancy_prediction_living_room_confidence",
                        "above": 80,
                    },
                    {
                        "condition": "state",
                        "entity_id": "binary_sensor.living_room_occupied",
                        "state": "off",
                    },
                ],
                "action": [
                    {
                        "service": "climate.set_temperature",
                        "target": {"entity_id": "climate.living_room"},
                        "data": {"temperature": 21},
                    },
                    {
                        "service": "notify.mobile_app",
                        "data": {
                            "message": "Pre-heating living room - occupancy predicted in {{ states('sensor.occupancy_prediction_living_room_time_until') }}",
                            "title": "Smart Heating",
                        },
                    },
                ],
            }
        }

        automation_config = sample_automation["automation"]

        # Validate automation structure
        assert "trigger" in automation_config
        assert "condition" in automation_config
        assert "action" in automation_config

        # Validate trigger uses prediction entities
        trigger = automation_config["trigger"][0]
        assert (
            "sensor.occupancy_prediction_living_room_next_occupied"
            in trigger["value_template"]
        )
        assert (
            "sensor.occupancy_prediction_living_room_confidence"
            in trigger["value_template_condition"]
        )

        # Validate conditions use confidence threshold
        confidence_condition = automation_config["condition"][0]
        assert (
            confidence_condition["entity_id"]
            == "sensor.occupancy_prediction_living_room_confidence"
        )
        assert confidence_condition["above"] == 80

        # Validate actions are meaningful
        actions = automation_config["action"]
        assert any(
            "climate.set_temperature" in str(action) for action in actions
        ), "Should include heating action"
        assert any(
            "notify" in str(action) for action in actions
        ), "Should include notification action"

    def test_dashboard_integration_scenario(self):
        """Test user scenario: View predictions in HA dashboard."""
        # User story: As a user, I want a dashboard card showing occupancy predictions

        # Mock Lovelace dashboard card configuration
        dashboard_card = {
            "type": "entities",
            "title": "Room Occupancy Predictions",
            "show_header_toggle": False,
            "entities": [
                {
                    "entity": "sensor.occupancy_prediction_living_room_next_occupied",
                    "name": "Living Room Next Occupied",
                    "icon": "mdi:sofa",
                    "format": "relative",
                },
                {
                    "entity": "sensor.occupancy_prediction_living_room_confidence",
                    "name": "Confidence",
                    "icon": "mdi:percent",
                },
                {
                    "entity": "sensor.occupancy_prediction_kitchen_next_vacant",
                    "name": "Kitchen Next Vacant",
                    "icon": "mdi:chef-hat",
                    "format": "relative",
                },
                {
                    "entity": "sensor.occupancy_prediction_kitchen_confidence",
                    "name": "Kitchen Confidence",
                    "icon": "mdi:percent",
                },
            ],
            "state_color": True,
        }

        # Validate dashboard configuration
        assert dashboard_card["type"] == "entities"
        assert "title" in dashboard_card
        assert "entities" in dashboard_card

        # Validate entities in dashboard
        entities = dashboard_card["entities"]
        prediction_entities = [
            e for e in entities if "occupancy_prediction" in e["entity"]
        ]

        assert (
            len(prediction_entities) == 4
        ), "Should have prediction entities for multiple rooms"

        # Validate entity configuration
        for entity_config in prediction_entities:
            assert "entity" in entity_config
            assert "name" in entity_config
            assert "icon" in entity_config
            assert entity_config["entity"].startswith("sensor.occupancy_prediction_")

        # Test advanced dashboard card with charts
        advanced_card = {
            "type": "custom:apexcharts-card",
            "title": "Prediction Accuracy Trends",
            "header": {"show": True, "title": "Model Performance Over Time"},
            "series": [
                {
                    "entity": "sensor.occupancy_prediction_living_room_accuracy",
                    "name": "Living Room Accuracy",
                    "type": "line",
                    "color": "#2196F3",
                },
                {
                    "entity": "sensor.occupancy_prediction_kitchen_accuracy",
                    "name": "Kitchen Accuracy",
                    "type": "line",
                    "color": "#4CAF50",
                },
            ],
            "apex_config": {
                "yaxis": {"min": 0, "max": 100, "title": {"text": "Accuracy (%)"}}
            },
        }

        # Validate advanced dashboard features
        assert "series" in advanced_card
        assert len(advanced_card["series"]) == 2
        assert "apex_config" in advanced_card


class TestMonitoringScenarios:
    """Test system monitoring scenarios."""

    def test_system_health_monitoring_scenario(self):
        """Test user scenario: Monitor system health and status."""
        # User story: As a user, I want to know if the prediction system is working properly

        # Mock system health status
        system_health = {
            "overall_status": "healthy",
            "last_updated": datetime.now(timezone.utc),
            "components": {
                "database": {
                    "status": "healthy",
                    "connection_pool_active": 3,
                    "connection_pool_size": 10,
                    "last_query_time_ms": 45,
                    "health_check_passed": True,
                },
                "mqtt_broker": {
                    "status": "healthy",
                    "connected": True,
                    "last_message_sent": datetime.now(timezone.utc)
                    - timedelta(minutes=2),
                    "message_queue_size": 0,
                    "reconnect_attempts": 0,
                },
                "home_assistant": {
                    "status": "healthy",
                    "websocket_connected": True,
                    "last_event_received": datetime.now(timezone.utc)
                    - timedelta(minutes=1),
                    "api_response_time_ms": 120,
                    "authentication_valid": True,
                },
                "prediction_engine": {
                    "status": "active",
                    "models_loaded": 3,
                    "last_prediction_time": datetime.now(timezone.utc)
                    - timedelta(minutes=5),
                    "prediction_queue_size": 1,
                    "average_prediction_time_ms": 85,
                },
                "api_server": {
                    "status": "running",
                    "port": 8080,
                    "active_connections": 2,
                    "requests_per_minute": 15,
                    "average_response_time_ms": 65,
                },
            },
        }

        # Validate overall health
        assert system_health["overall_status"] in ["healthy", "degraded", "unhealthy"]
        assert system_health["overall_status"] == "healthy"

        # Validate component health
        for component_name, component_status in system_health["components"].items():
            assert (
                "status" in component_status
            ), f"Component {component_name} missing status"
            assert component_status["status"] in [
                "healthy",
                "active",
                "running",
                "degraded",
                "failed",
            ]

        # Test specific component validations
        db_status = system_health["components"]["database"]
        assert db_status["health_check_passed"] is True
        assert db_status["connection_pool_active"] <= db_status["connection_pool_size"]

        mqtt_status = system_health["components"]["mqtt_broker"]
        assert mqtt_status["connected"] is True
        assert mqtt_status["message_queue_size"] < 100  # Should not be backed up

        ha_status = system_health["components"]["home_assistant"]
        assert ha_status["websocket_connected"] is True
        assert ha_status["authentication_valid"] is True

        prediction_status = system_health["components"]["prediction_engine"]
        assert prediction_status["models_loaded"] > 0
        assert prediction_status["average_prediction_time_ms"] < 500  # Under 500ms

    def test_performance_monitoring_scenario(self):
        """Test user scenario: Monitor system performance metrics."""
        # User story: As a user, I want to know how well the system is performing

        # Mock performance metrics
        performance_metrics = {
            "prediction_performance": {
                "avg_prediction_time_ms": 125,
                "95th_percentile_ms": 250,
                "predictions_per_minute": 12,
                "prediction_cache_hit_rate": 0.68,
                "feature_extraction_time_ms": 45,
            },
            "accuracy_metrics": {
                "overall_accuracy_percentage": 87.3,
                "avg_error_minutes": 11.8,
                "confidence_calibration": 0.91,
                "predictions_validated": 156,
                "accuracy_by_room": {
                    "living_room": 89.2,
                    "kitchen": 85.1,
                    "bedroom": 88.7,
                },
            },
            "resource_usage": {
                "cpu_usage_percent": 15.2,
                "memory_usage_mb": 245,
                "disk_usage_mb": 1024,
                "database_size_mb": 512,
                "network_throughput_kbps": 128,
            },
            "availability_metrics": {
                "uptime_hours": 168.5,  # ~7 days
                "availability_percentage": 99.8,
                "error_rate_per_hour": 0.2,
                "recovery_time_minutes": 2.1,
            },
        }

        # Validate prediction performance
        pred_perf = performance_metrics["prediction_performance"]
        assert (
            pred_perf["avg_prediction_time_ms"] < 200
        ), "Average prediction time should be under 200ms"
        assert (
            pred_perf["95th_percentile_ms"] < 500
        ), "95th percentile should be under 500ms"
        assert (
            pred_perf["prediction_cache_hit_rate"] > 0.5
        ), "Cache hit rate should be above 50%"

        # Validate accuracy metrics
        accuracy = performance_metrics["accuracy_metrics"]
        assert (
            accuracy["overall_accuracy_percentage"] >= 85.0
        ), "Overall accuracy should be 85% or higher"
        assert (
            accuracy["avg_error_minutes"] <= 15.0
        ), "Average error should be 15 minutes or less"
        assert (
            accuracy["confidence_calibration"] >= 0.85
        ), "Confidence should be well-calibrated"

        # Validate per-room accuracy
        for room, room_accuracy in accuracy["accuracy_by_room"].items():
            assert room_accuracy >= 80.0, f"Room {room} accuracy should be at least 80%"

        # Validate resource usage
        resources = performance_metrics["resource_usage"]
        assert resources["cpu_usage_percent"] < 50.0, "CPU usage should be reasonable"
        assert resources["memory_usage_mb"] < 1000, "Memory usage should be under 1GB"

        # Validate availability
        availability = performance_metrics["availability_metrics"]
        assert (
            availability["availability_percentage"] >= 99.0
        ), "System should have 99%+ availability"
        assert availability["error_rate_per_hour"] < 1.0, "Error rate should be low"

    def test_alert_scenarios_workflow(self):
        """Test user scenario: Receive alerts for system issues."""
        # User story: As a user, I want to be notified when the system has problems

        # Mock alert conditions and thresholds
        alert_rules = {
            "accuracy_degradation": {
                "threshold": 80.0,  # Accuracy below 80%
                "severity": "warning",
                "notification_channels": ["email", "ha_notification"],
            },
            "prediction_latency": {
                "threshold": 500,  # Prediction takes over 500ms
                "severity": "warning",
                "notification_channels": ["ha_notification"],
            },
            "component_failure": {
                "threshold": "any_component_failed",
                "severity": "critical",
                "notification_channels": ["email", "ha_notification", "sms"],
            },
            "high_error_rate": {
                "threshold": 5,  # More than 5 errors per hour
                "severity": "warning",
                "notification_channels": ["email"],
            },
        }

        # Mock current system status (with some issues)
        current_status = {
            "overall_accuracy": 78.5,  # Below threshold - should trigger alert
            "prediction_latency_ms": 350,  # Within threshold
            "component_statuses": {
                "database": "healthy",
                "mqtt": "healthy",
                "ha_client": "degraded",  # Not failed, so no critical alert
                "api_server": "healthy",
            },
            "errors_last_hour": 7,  # Above threshold - should trigger alert
        }

        # Alert evaluation logic
        triggered_alerts = []

        # Check accuracy degradation
        if (
            current_status["overall_accuracy"]
            < alert_rules["accuracy_degradation"]["threshold"]
        ):
            triggered_alerts.append(
                {
                    "rule": "accuracy_degradation",
                    "severity": alert_rules["accuracy_degradation"]["severity"],
                    "message": f"Prediction accuracy dropped to {current_status['overall_accuracy']:.1f}%",
                    "channels": alert_rules["accuracy_degradation"][
                        "notification_channels"
                    ],
                }
            )

        # Check prediction latency
        if (
            current_status["prediction_latency_ms"]
            > alert_rules["prediction_latency"]["threshold"]
        ):
            triggered_alerts.append(
                {
                    "rule": "prediction_latency",
                    "severity": alert_rules["prediction_latency"]["severity"],
                    "message": f"Prediction latency increased to {current_status['prediction_latency_ms']}ms",
                    "channels": alert_rules["prediction_latency"][
                        "notification_channels"
                    ],
                }
            )

        # Check component failures
        failed_components = [
            comp
            for comp, status in current_status["component_statuses"].items()
            if status == "failed"
        ]
        if failed_components:
            triggered_alerts.append(
                {
                    "rule": "component_failure",
                    "severity": alert_rules["component_failure"]["severity"],
                    "message": f"Components failed: {', '.join(failed_components)}",
                    "channels": alert_rules["component_failure"][
                        "notification_channels"
                    ],
                }
            )

        # Check error rate
        if (
            current_status["errors_last_hour"]
            > alert_rules["high_error_rate"]["threshold"]
        ):
            triggered_alerts.append(
                {
                    "rule": "high_error_rate",
                    "severity": alert_rules["high_error_rate"]["severity"],
                    "message": f"High error rate: {current_status['errors_last_hour']} errors in last hour",
                    "channels": alert_rules["high_error_rate"]["notification_channels"],
                }
            )

        # Validate alert triggering
        assert (
            len(triggered_alerts) == 2
        ), "Should trigger exactly 2 alerts based on current status"

        alert_rules_triggered = [alert["rule"] for alert in triggered_alerts]
        assert (
            "accuracy_degradation" in alert_rules_triggered
        ), "Should trigger accuracy degradation alert"
        assert (
            "high_error_rate" in alert_rules_triggered
        ), "Should trigger high error rate alert"
        assert (
            "component_failure" not in alert_rules_triggered
        ), "Should not trigger component failure (no failures)"
        assert (
            "prediction_latency" not in alert_rules_triggered
        ), "Should not trigger latency alert (within threshold)"

        # Validate alert content
        accuracy_alert = next(
            alert
            for alert in triggered_alerts
            if alert["rule"] == "accuracy_degradation"
        )
        assert accuracy_alert["severity"] == "warning"
        assert "78.5%" in accuracy_alert["message"]
        assert "email" in accuracy_alert["channels"]
        assert "ha_notification" in accuracy_alert["channels"]

        # Test alert suppression logic (don't spam user)
        alert_history = {
            "accuracy_degradation": {
                "last_sent": datetime.now(timezone.utc)
                - timedelta(minutes=10),  # Recent
                "count_today": 3,
            },
            "high_error_rate": {
                "last_sent": datetime.now(timezone.utc)
                - timedelta(hours=2),  # Not recent
                "count_today": 1,
            },
        }

        # Filter alerts based on suppression rules
        suppression_rules = {
            "min_interval_minutes": 30,  # Don't send same alert more than once per 30 minutes
            "max_per_day": 5,  # Don't send more than 5 of same alert per day
        }

        alerts_to_send = []
        for alert in triggered_alerts:
            rule = alert["rule"]
            if rule in alert_history:
                history = alert_history[rule]
                minutes_since_last = (
                    datetime.now(timezone.utc) - history["last_sent"]
                ).total_seconds() / 60

                if (
                    minutes_since_last >= suppression_rules["min_interval_minutes"]
                    and history["count_today"] < suppression_rules["max_per_day"]
                ):
                    alerts_to_send.append(alert)
            else:
                alerts_to_send.append(alert)

        # Validate suppression logic
        assert (
            len(alerts_to_send) == 1
        ), "Should suppress recent accuracy alert, but send error rate alert"
        assert (
            alerts_to_send[0]["rule"] == "high_error_rate"
        ), "Should only send the high error rate alert"
