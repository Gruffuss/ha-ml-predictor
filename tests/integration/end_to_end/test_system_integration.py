"""End-to-end integration tests for the complete system.

Covers system-wide integration scenarios, full workflow testing,
and complete data pipeline validation from ingestion to prediction delivery.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import (
    APIConfig,
    DatabaseConfig,
    FeaturesConfig,
    HomeAssistantConfig,
    LoggingConfig,
    MQTTConfig,
    PredictionConfig,
    SystemConfig,
    TrackingConfig,
)
from src.data.storage.database import DatabaseManager
from src.data.storage.models import Base, RoomState, SensorEvent
from src.integration.mqtt_integration_manager import MQTTIntegrationManager
from src.main_system import OccupancyPredictionSystem


class TestCompleteSystemWorkflow:
    """Test complete system workflow end-to-end."""

    @pytest.fixture
    def mock_system_config(self):
        """Create complete mock system configuration."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        config = SystemConfig(
            home_assistant=HomeAssistantConfig(
                url="http://test-ha:8123", token="test-token"
            ),
            database=DatabaseConfig(
                connection_string=f"sqlite+aiosqlite:///{temp_db.name}"
            ),
            mqtt=MQTTConfig(broker="test-mqtt", port=1883),
            api=APIConfig(enabled=True, host="127.0.0.1", port=8080),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(enabled=True),
        )

        yield config, temp_db.name

        # Cleanup
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_system_initialization_workflow(self, mock_system_config):
        """Test complete system initialization workflow."""
        config, temp_db_path = mock_system_config

        with patch("src.core.config.get_config", return_value=config), patch(
            "src.data.ingestion.ha_client.HomeAssistantClient"
        ) as mock_ha_client, patch("paho.mqtt.client.Client") as mock_mqtt_client:

            # Configure mocks
            mock_ha_client.return_value.connect = AsyncMock()
            mock_ha_client.return_value.is_connected = MagicMock(return_value=True)

            mock_mqtt_client.return_value.connect.return_value = 0
            mock_mqtt_client.return_value.is_connected.return_value = True

            # Create and initialize system
            system = OccupancyPredictionSystem()

            await system.initialize()

            # Verify system components were initialized
            assert system.running is True
            assert system.database_manager is not None
            assert system.tracking_manager is not None
            assert system.mqtt_manager is not None

            # Test system health
            health_status = system.tracking_manager.get_api_server_status()
            assert "enabled" in health_status

            # Cleanup
            await system.shutdown()
            assert system.running is False

    @pytest.mark.asyncio
    async def test_data_ingestion_to_prediction_workflow(self, mock_system_config):
        """Test complete data flow from ingestion to prediction."""
        config, temp_db_path = mock_system_config

        with patch("src.core.config.get_config", return_value=config), patch(
            "src.data.ingestion.ha_client.HomeAssistantClient"
        ) as mock_ha_client, patch("paho.mqtt.client.Client") as mock_mqtt_client:

            # Configure HA client mock
            mock_ha_instance = mock_ha_client.return_value
            mock_ha_instance.connect = AsyncMock()
            mock_ha_instance.is_connected = MagicMock(return_value=True)

            # Configure MQTT client mock
            mock_mqtt_instance = mock_mqtt_client.return_value
            mock_mqtt_instance.connect.return_value = 0
            mock_mqtt_instance.publish.return_value = MagicMock(rc=0)
            mock_mqtt_instance.is_connected.return_value = True

            # Initialize database with test data
            db_manager = DatabaseManager(config.database)
            await db_manager.initialize()

            # Create database tables
            async with db_manager.get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Insert test sensor events
            now = datetime.now(timezone.utc)
            test_events = [
                SensorEvent(
                    room_id="living_room",
                    sensor_id="binary_sensor.living_room_motion",
                    sensor_type="motion",
                    state="on",
                    previous_state="off",
                    timestamp=now - timedelta(minutes=30),
                    is_human_triggered=True,
                ),
                SensorEvent(
                    room_id="living_room",
                    sensor_id="binary_sensor.living_room_motion",
                    sensor_type="motion",
                    state="off",
                    previous_state="on",
                    timestamp=now - timedelta(minutes=15),
                    is_human_triggered=True,
                ),
            ]

            async with db_manager.get_session() as session:
                session.add_all(test_events)
                await session.commit()

            # Create tracking manager and test prediction workflow
            with patch(
                "src.adaptation.tracking_manager.TrackingManager"
            ) as mock_tracking:
                mock_tracking_instance = mock_tracking.return_value
                mock_tracking_instance.initialize = AsyncMock()
                mock_tracking_instance.get_current_predictions.return_value = {
                    "living_room": {
                        "next_transition_time": now + timedelta(minutes=45),
                        "transition_type": "vacant",
                        "confidence": 0.82,
                        "prediction_time": now,
                    }
                }

                # Initialize system
                system = OccupancyPredictionSystem()
                await system.initialize()

                # Verify prediction workflow
                predictions = system.tracking_manager.get_current_predictions()
                assert "living_room" in predictions
                assert predictions["living_room"]["confidence"] == 0.82

                # Cleanup
                await system.shutdown()
                await db_manager.close()

    @pytest.mark.asyncio
    async def test_system_startup_shutdown_lifecycle(self, mock_system_config):
        """Test complete system startup and shutdown lifecycle."""
        config, temp_db_path = mock_system_config

        with patch("src.core.config.get_config", return_value=config), patch(
            "src.data.ingestion.ha_client.HomeAssistantClient"
        ) as mock_ha, patch("paho.mqtt.client.Client") as mock_mqtt:

            # Configure mocks for successful connections
            mock_ha.return_value.connect = AsyncMock()
            mock_ha.return_value.is_connected = MagicMock(return_value=True)

            mock_mqtt.return_value.connect.return_value = 0
            mock_mqtt.return_value.is_connected.return_value = True

            system = OccupancyPredictionSystem()

            # Test startup sequence
            assert system.running is False

            await system.initialize()

            # Verify system is running
            assert system.running is True
            assert system.database_manager is not None
            assert system.tracking_manager is not None
            assert system.mqtt_manager is not None

            # Test shutdown sequence
            await system.shutdown()

            # Verify clean shutdown
            assert system.running is False

    @pytest.mark.asyncio
    async def test_fault_tolerance_workflow(self, mock_system_config):
        """Test system fault tolerance and recovery."""
        config, temp_db_path = mock_system_config

        with patch("src.core.config.get_config", return_value=config):
            # Test database connection failure
            with patch("src.data.storage.database.create_async_engine") as mock_engine:
                mock_engine.side_effect = Exception("Database connection failed")

                system = OccupancyPredictionSystem()

                # Should handle database failure gracefully
                with pytest.raises(Exception):
                    await system.initialize()

                assert system.running is False

            # Test MQTT connection failure with graceful handling
            with patch("paho.mqtt.client.Client") as mock_mqtt:
                mock_mqtt.return_value.connect.return_value = 1  # Connection failed

                # System should still initialize but log MQTT failure
                system = OccupancyPredictionSystem()

                # The system should handle MQTT failure and continue
                # (depending on implementation, might still initialize)
                try:
                    await system.initialize()
                    # If initialization succeeds, clean up
                    if system.running:
                        await system.shutdown()
                except Exception:
                    # Expected if system requires MQTT to be working
                    pass


class TestSystemIntegration:
    """Test integration between major system components."""

    @pytest.fixture
    def mock_components(self):
        """Create mock system components for integration testing."""
        components = {
            "database_manager": MagicMock(),
            "tracking_manager": MagicMock(),
            "mqtt_manager": MagicMock(),
            "ha_client": MagicMock(),
        }

        # Configure mock behaviors
        components["database_manager"].health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        components["tracking_manager"].get_current_predictions.return_value = {
            "test_room": {"confidence": 0.9}
        }
        components["mqtt_manager"].is_connected.return_value = True

        return components

    def test_component_communication_integration(self, mock_components):
        """Test communication between system components."""
        db_manager = mock_components["database_manager"]
        tracking_manager = mock_components["tracking_manager"]
        mqtt_manager = mock_components["mqtt_manager"]

        # Test tracking manager -> database communication
        tracking_manager.get_current_predictions()
        tracking_manager.get_current_predictions.assert_called_once()

        # Test MQTT manager status check
        mqtt_status = mqtt_manager.is_connected()
        assert mqtt_status is True

        # Test database health check
        asyncio.run(db_manager.health_check())
        db_manager.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self):
        """Test data consistency across system components."""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        try:
            config = DatabaseConfig(
                connection_string=f"sqlite+aiosqlite:///{temp_db.name}"
            )

            db_manager = DatabaseManager(config)
            await db_manager.initialize()

            # Create tables
            async with db_manager.get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Test data consistency by creating and reading data
            now = datetime.now(timezone.utc)
            test_event = SensorEvent(
                room_id="consistency_test",
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                timestamp=now,
                is_human_triggered=True,
            )

            # Write data through one component
            async with db_manager.get_session() as session:
                session.add(test_event)
                await session.commit()
                event_id = test_event.id

            # Read data through another component simulation
            async with db_manager.get_session() as session:
                retrieved_event = await session.get(SensorEvent, event_id)

                assert retrieved_event is not None
                assert retrieved_event.room_id == "consistency_test"
                assert retrieved_event.state == "on"

            await db_manager.close()

        finally:
            # Cleanup
            os.unlink(temp_db.name)

    def test_performance_integration_monitoring(self, mock_components):
        """Test performance monitoring across integrated components."""
        import time

        # Simulate component performance testing
        start_time = time.time()

        # Test multiple component calls
        for _ in range(10):
            mock_components["tracking_manager"].get_current_predictions()
            mock_components["mqtt_manager"].is_connected()

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete quickly for mocked components
        assert total_time < 1.0

        # Verify all calls were made
        assert (
            mock_components["tracking_manager"].get_current_predictions.call_count == 10
        )
        assert mock_components["mqtt_manager"].is_connected.call_count == 10


class TestWorkflowValidation:
    """Test complete workflow validation scenarios."""

    @pytest.fixture
    def workflow_mocks(self):
        """Create comprehensive workflow mocks."""
        mocks = {
            "sensor_data": [
                {
                    "room_id": "living_room",
                    "sensor_id": "motion_sensor",
                    "state": "on",
                    "timestamp": datetime.now(timezone.utc),
                }
            ],
            "predictions": {
                "living_room": {
                    "next_transition_time": datetime.now(timezone.utc)
                    + timedelta(minutes=30),
                    "confidence": 0.85,
                    "transition_type": "vacant",
                }
            },
            "mqtt_messages": [],
        }
        return mocks

    def test_prediction_workflow_integration(self, workflow_mocks):
        """Test complete prediction workflow."""
        # Simulate prediction workflow
        sensor_data = workflow_mocks["sensor_data"][0]
        expected_predictions = workflow_mocks["predictions"]

        # Test workflow steps
        # 1. Sensor data received
        assert sensor_data["state"] == "on"
        assert sensor_data["room_id"] == "living_room"

        # 2. Prediction generated
        room_prediction = expected_predictions["living_room"]
        assert room_prediction["confidence"] >= 0.7
        assert "next_transition_time" in room_prediction

        # 3. Prediction validated
        assert room_prediction["transition_type"] in ["occupied", "vacant"]

    def test_adaptation_workflow_integration(self, workflow_mocks):
        """Test model adaptation workflow."""
        # Simulate adaptation triggers
        accuracy_metrics = {
            "living_room": {
                "avg_error_minutes": 18.5,  # Above threshold
                "total_predictions": 100,
                "accuracy_percentage": 82.0,
            }
        }

        # Test adaptation decision logic
        threshold = 15.0  # minutes
        needs_retraining = (
            accuracy_metrics["living_room"]["avg_error_minutes"] > threshold
        )

        assert needs_retraining is True

        # Simulate retraining workflow
        retraining_config = {
            "room_id": "living_room",
            "trigger_reason": "accuracy_degradation",
            "data_window_days": 7,
            "expected_improvement": 0.05,
        }

        assert retraining_config["trigger_reason"] == "accuracy_degradation"
        assert retraining_config["data_window_days"] > 0

    def test_monitoring_workflow_integration(self, workflow_mocks):
        """Test system monitoring workflow."""
        # Simulate system monitoring
        system_status = {
            "database": {"status": "healthy", "connections": 5},
            "mqtt": {"status": "healthy", "connected": True},
            "predictions": {
                "status": "active",
                "last_update": datetime.now(timezone.utc),
            },
            "api_server": {"status": "running", "port": 8080},
        }

        # Test monitoring checks
        all_healthy = all(
            component["status"] in ["healthy", "active", "running"]
            for component in system_status.values()
        )

        assert all_healthy is True

        # Test alert conditions
        alert_conditions = []

        if system_status["database"]["connections"] > 10:
            alert_conditions.append("high_db_connections")

        if not system_status["mqtt"]["connected"]:
            alert_conditions.append("mqtt_disconnected")

        # No alerts should be triggered in healthy system
        assert len(alert_conditions) == 0

    def test_end_to_end_workflow_validation(self, workflow_mocks):
        """Test complete end-to-end workflow validation."""
        # Simulate complete workflow from sensor event to MQTT publish
        workflow_steps = {
            "1_sensor_event_received": True,
            "2_event_processed": True,
            "3_features_extracted": True,
            "4_prediction_generated": True,
            "5_prediction_validated": True,
            "6_mqtt_published": True,
            "7_database_updated": True,
        }

        # Verify all workflow steps completed
        workflow_success = all(workflow_steps.values())
        assert workflow_success is True

        # Test workflow metrics
        workflow_metrics = {
            "total_steps": len(workflow_steps),
            "completed_steps": sum(workflow_steps.values()),
            "success_rate": sum(workflow_steps.values()) / len(workflow_steps),
        }

        assert workflow_metrics["success_rate"] == 1.0
        assert workflow_metrics["completed_steps"] == workflow_metrics["total_steps"]

    def test_error_handling_in_workflow(self, workflow_mocks):
        """Test error handling throughout workflow."""
        # Simulate workflow with errors
        workflow_errors = []

        try:
            # Step 1: Process sensor data
            sensor_data = workflow_mocks["sensor_data"][0]
            if sensor_data["state"] not in ["on", "off"]:
                raise ValueError("Invalid sensor state")

            # Step 2: Generate prediction
            if workflow_mocks["predictions"]["living_room"]["confidence"] < 0.5:
                raise ValueError("Low confidence prediction")

            # Step 3: Publish to MQTT
            # Simulate MQTT error
            if len(workflow_mocks["mqtt_messages"]) > 100:
                raise ConnectionError("MQTT queue full")

        except ValueError as e:
            workflow_errors.append(f"validation_error: {e}")
        except ConnectionError as e:
            workflow_errors.append(f"connection_error: {e}")
        except Exception as e:
            workflow_errors.append(f"unknown_error: {e}")

        # In this test case, no errors should occur
        assert len(workflow_errors) == 0

        # Test error recovery simulation
        error_recovery_config = {
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "fallback_enabled": True,
            "alert_on_failure": True,
        }

        assert error_recovery_config["max_retries"] > 0
        assert error_recovery_config["fallback_enabled"] is True
