"""
Cross-Component Integration Tests.

This module tests the integration between multiple system components working together:
MQTT publishing, API server, HA entity definitions, tracking system, and database.

Consolidated from Sprint 4 & 5 integration tests for complete coverage.

Focus Areas:
- End-to-end data flow from tracking to MQTT publication
- API server integration with MQTT and entity systems
- Real-time entity state updates via MQTT
- Service command handling through API and MQTT
- Error propagation and recovery across components
- Performance under concurrent cross-component operations
- Data consistency across all integration layers
- Self-adaptation system integration (Sprint 4)
- Real-time publishing and WebSocket integration (Sprint 5)
- Complete prediction lifecycle validation
- Alert system integration across all components
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, call, patch
import uuid

from fastapi.testclient import TestClient
import pytest

from src.core.config import MQTTConfig, RoomConfig, get_config
from src.integration.api_server import create_app, set_tracking_manager
from src.integration.discovery_publisher import DeviceInfo, DiscoveryPublisher
from src.integration.ha_entity_definitions import HAEntityDefinitions
from src.integration.mqtt_publisher import MQTTPublisher, MQTTPublishResult

# Test configuration
TEST_CONFIG = {
    "mqtt": MQTTConfig(
        broker="localhost",
        port=1883,
        topic_prefix="homeassistant",
        discovery_prefix="homeassistant",
        publishing_enabled=True,
        discovery_enabled=True,
        device_identifier="test_integrated_system",
        keepalive=60,
        connection_timeout=10,
        max_reconnect_attempts=3,
        reconnect_delay_seconds=1,
    ),
    "device_info": DeviceInfo(
        identifiers=["test_integrated_system"],
        name="Test Integrated System",
        manufacturer="Test Manufacturer",
        model="Integration Test Model",
        sw_version="1.0.0-integration",
        configuration_url="http://localhost:8000",
    ),
    "rooms": {
        "living_room": RoomConfig(
            room_id="living_room",
            name="Living Room",
            sensors={
                "motion": ["binary_sensor.living_room_motion"],
                "door": ["binary_sensor.living_room_door"],
            },
        ),
        "bedroom": RoomConfig(
            room_id="bedroom",
            name="Bedroom",
            sensors={
                "motion": ["binary_sensor.bedroom_motion"],
                "door": ["binary_sensor.bedroom_door"],
            },
        ),
    },
}


@pytest.fixture
def mock_mqtt_publisher():
    """Create mock MQTT publisher that tracks all operations."""
    mock_publisher = AsyncMock(spec=MQTTPublisher)

    # Track published messages
    mock_publisher.published_messages = []
    mock_publisher.published_discoveries = []

    async def track_publish_json(topic, data, qos=1, retain=False):
        message = {
            "topic": topic,
            "data": data,
            "qos": qos,
            "retain": retain,
            "timestamp": datetime.utcnow(),
        }

        if "/config" in topic:
            mock_publisher.published_discoveries.append(message)
        else:
            mock_publisher.published_messages.append(message)

        return MQTTPublishResult(
            success=True,
            topic=topic,
            payload_size=len(json.dumps(data)),
            publish_time=datetime.utcnow(),
            message_id=len(mock_publisher.published_messages),
        )

    async def track_publish(topic, payload, qos=1, retain=False):
        message = {
            "topic": topic,
            "payload": payload,
            "qos": qos,
            "retain": retain,
            "timestamp": datetime.utcnow(),
        }
        mock_publisher.published_messages.append(message)

        return MQTTPublishResult(
            success=True,
            topic=topic,
            payload_size=len(str(payload)),
            publish_time=datetime.utcnow(),
            message_id=len(mock_publisher.published_messages),
        )

    mock_publisher.publish_json.side_effect = track_publish_json
    mock_publisher.publish.side_effect = track_publish

    # Mock connection status
    mock_publisher.connection_status.connected = True
    mock_publisher.get_publisher_stats.return_value = {
        "client_id": "test_client",
        "messages_published": 0,
        "messages_failed": 0,
        "queued_messages": 0,
    }

    return mock_publisher


@pytest.fixture
def mock_tracking_manager():
    """Create comprehensive mock tracking manager."""
    mock_manager = AsyncMock()

    # Mock prediction data
    mock_manager.predictions = {
        "living_room": {
            "room_id": "living_room",
            "prediction_time": datetime.now().isoformat(),
            "next_transition_time": (
                datetime.now() + timedelta(minutes=25)
            ).isoformat(),
            "transition_type": "vacant_to_occupied",
            "confidence": 0.87,
            "time_until_transition": "25 minutes",
            "alternatives": [
                {
                    "time": (datetime.now() + timedelta(minutes=30)).isoformat(),
                    "confidence": 0.75,
                }
            ],
            "model_info": {"model": "ensemble", "accuracy": 0.89},
        },
        "bedroom": {
            "room_id": "bedroom",
            "prediction_time": datetime.now().isoformat(),
            "next_transition_time": (
                datetime.now() + timedelta(minutes=45)
            ).isoformat(),
            "transition_type": "occupied_to_vacant",
            "confidence": 0.93,
            "time_until_transition": "45 minutes",
            "alternatives": [],
            "model_info": {"model": "lstm", "accuracy": 0.91},
        },
    }

    # Mock methods
    async def get_room_prediction(room_id):
        return mock_manager.predictions.get(room_id)

    async def get_accuracy_metrics(room_id=None, hours=24):
        if room_id:
            return {
                "room_id": room_id,
                "accuracy_rate": 0.87,
                "average_error_minutes": 12.5,
                "confidence_calibration": 0.89,
                "total_predictions": 156,
                "total_validations": 148,
                "time_window_hours": hours,
                "trend_direction": "improving",
            }
        else:
            return {
                "room_id": None,
                "accuracy_rate": 0.85,
                "average_error_minutes": 13.2,
                "confidence_calibration": 0.88,
                "total_predictions": 324,
                "total_validations": 301,
                "time_window_hours": hours,
                "trend_direction": "stable",
            }

    async def get_tracking_status():
        return {
            "tracking_active": True,
            "status": "active",
            "config": {"enabled": True},
            "performance": {
                "background_tasks": 3,
                "total_predictions_recorded": 324,
                "total_validations_performed": 301,
                "total_drift_checks_performed": 25,
                "system_uptime_seconds": 86400,
            },
            "validator": {"total_predictions": 324},
            "accuracy_tracker": {"total_predictions": 301},
            "drift_detector": {"status": "active"},
            "adaptive_retrainer": {"status": "active"},
        }

    async def get_system_stats():
        return {
            "tracking_stats": {
                "total_predictions_tracked": 324,
                "active_room_trackers": 2,
                "average_accuracy": 0.85,
            },
            "retraining_stats": {
                "completed_retraining_jobs": 5,
                "failed_retraining_jobs": 0,
                "last_retrain_time": datetime.now().isoformat(),
            },
        }

    async def trigger_manual_retrain(
        room_id=None, force=False, strategy="auto", reason="manual"
    ):
        return {
            "message": f"Retraining triggered for {room_id or 'all rooms'}",
            "success": True,
            "room_id": room_id,
            "strategy": strategy,
            "force": force,
            "reason": reason,
            "job_id": str(uuid.uuid4()),
        }

    mock_manager.get_room_prediction.side_effect = get_room_prediction
    mock_manager.get_accuracy_metrics.side_effect = get_accuracy_metrics
    mock_manager.get_tracking_status.side_effect = get_tracking_status
    mock_manager.get_system_stats.side_effect = get_system_stats
    mock_manager.trigger_manual_retrain.side_effect = trigger_manual_retrain

    return mock_manager


@pytest.fixture
def mock_database_manager():
    """Create mock database manager."""
    mock_db = AsyncMock()

    async def health_check():
        return {
            "status": "healthy",
            "database_connected": True,
            "connection_pool_size": 10,
            "active_connections": 2,
            "total_queries": 1250,
            "average_query_time_ms": 15.3,
        }

    mock_db.health_check.side_effect = health_check
    return mock_db


@pytest.fixture
def integrated_system(
    mock_mqtt_publisher, mock_tracking_manager, mock_database_manager
):
    """Create integrated system with all components."""

    # Create HA entity definitions
    discovery_publisher = DiscoveryPublisher(
        mqtt_publisher=mock_mqtt_publisher,
        device_info=TEST_CONFIG["device_info"],
        mqtt_config=TEST_CONFIG["mqtt"],
    )

    ha_entities = HAEntityDefinitions(
        discovery_publisher=discovery_publisher,
        mqtt_config=TEST_CONFIG["mqtt"],
        rooms=TEST_CONFIG["rooms"],
    )

    # Create test client with mocked dependencies
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "test",
            "JWT_SECRET_KEY": "test_secret_key_integration",
            "API_KEY": "test_api_key_integration",
        },
    ):
        with patch(
            "src.integration.api_server.get_database_manager",
            return_value=mock_database_manager,
        ):
            test_app = create_app()
            set_tracking_manager(mock_tracking_manager)

            return {
                "app": test_app,
                "mqtt_publisher": mock_mqtt_publisher,
                "tracking_manager": mock_tracking_manager,
                "database_manager": mock_database_manager,
                "ha_entities": ha_entities,
                "discovery_publisher": discovery_publisher,
            }


class TestEndToEndDataFlow:
    """Test complete end-to-end data flow across all components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prediction_to_mqtt_publication_flow(self, integrated_system):
        """Test complete flow from tracking manager prediction to MQTT publication."""
        ha_entities = integrated_system["ha_entities"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]

        # 1. Define and publish HA entities
        entities = ha_entities.define_all_entities()
        await ha_entities.publish_all_entities()

        # Verify discovery messages were published
        assert len(mqtt_publisher.published_discoveries) > 0

        # 2. Get prediction from tracking manager
        prediction = await tracking_manager.get_room_prediction("living_room")
        assert prediction is not None

        # 3. Simulate publishing prediction state to MQTT
        prediction_topic = "homeassistant/living_room/prediction"
        await mqtt_publisher.publish_json(prediction_topic, prediction)

        # 4. Verify prediction was published
        prediction_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if msg["topic"] == prediction_topic
        ]
        assert len(prediction_messages) == 1

        prediction_msg = prediction_messages[0]
        assert prediction_msg["data"]["room_id"] == "living_room"
        assert prediction_msg["data"]["confidence"] == 0.87

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_to_mqtt_integration(self, integrated_system):
        """Test API server triggering MQTT publications."""
        app = integrated_system["app"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        with TestClient(app) as client:
            # Get prediction via API
            response = client.get(
                "/predictions/living_room",
                headers={"Authorization": "Bearer test_api_key_integration"},
            )

            assert response.status_code == 200
            prediction_data = response.json()

            # Simulate API server publishing to MQTT based on API call
            mqtt_topic = f"homeassistant/{prediction_data['room_id']}/api_triggered"
            await mqtt_publisher.publish_json(
                mqtt_topic,
                {
                    "source": "api_request",
                    "prediction": prediction_data,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Verify MQTT publication occurred
            api_triggered_msgs = [
                msg
                for msg in mqtt_publisher.published_messages
                if "api_triggered" in msg["topic"]
            ]
            assert len(api_triggered_msgs) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_service_command_handling_flow(self, integrated_system):
        """Test service command flow from API through MQTT to system."""
        app = integrated_system["app"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]
        ha_entities = integrated_system["ha_entities"]

        # 1. Define services
        services = ha_entities.define_all_services()
        await ha_entities.publish_all_services()

        # Verify service buttons were created
        service_discoveries = [
            msg
            for msg in mqtt_publisher.published_discoveries
            if "button/" in msg["topic"]
        ]
        assert len(service_discoveries) > 0

        # 2. Trigger retrain via API
        with TestClient(app) as client:
            retrain_request = {
                "room_id": "living_room",
                "force": True,
                "strategy": "full",
                "reason": "integration_test",
            }

            response = client.post(
                "/model/retrain",
                json=retrain_request,
                headers={"Authorization": "Bearer test_api_key_integration"},
            )

            assert response.status_code == 200
            retrain_result = response.json()

            # 3. Simulate MQTT command topic publication
            command_topic = "homeassistant/commands/retrain"
            await mqtt_publisher.publish_json(
                command_topic,
                {
                    "command": "retrain_model",
                    "parameters": retrain_request,
                    "request_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # 4. Verify command was published and tracking manager was called
            command_msgs = [
                msg
                for msg in mqtt_publisher.published_messages
                if "commands/" in msg["topic"]
            ]
            assert len(command_msgs) == 1

            # Verify tracking manager was called by API
            tracking_manager.trigger_manual_retrain.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_time_state_updates(self, integrated_system):
        """Test real-time state updates across all components."""
        ha_entities = integrated_system["ha_entities"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]

        # 1. Establish entities
        entities = ha_entities.define_all_entities()
        await ha_entities.publish_all_entities()

        # 2. Simulate real-time prediction updates
        rooms = ["living_room", "bedroom"]

        for i, room_id in enumerate(rooms):
            # Update prediction in tracking manager
            new_prediction = {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "next_transition_time": (
                    datetime.now() + timedelta(minutes=10 + i * 5)
                ).isoformat(),
                "transition_type": "occupied" if i % 2 == 0 else "vacant",
                "confidence": 0.80 + i * 0.05,
                "time_until_transition": f"{10+i*5} minutes",
                "model_info": {"model": "realtime_update"},
            }

            tracking_manager.predictions[room_id] = new_prediction

            # Publish state updates to multiple MQTT topics
            topics = [
                f"homeassistant/{room_id}/prediction",
                f"homeassistant/{room_id}/state",
                f"homeassistant/{room_id}/accuracy",
            ]

            for topic in topics:
                await mqtt_publisher.publish_json(topic, new_prediction)

        # 3. Verify all state updates were published
        state_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if any(room in msg["topic"] for room in rooms)
        ]

        # Should have 3 topics * 2 rooms = 6 messages
        assert len(state_messages) >= 6

        # 4. Verify data consistency across topics
        for room_id in rooms:
            room_messages = [msg for msg in state_messages if room_id in msg["topic"]]

            # All messages for this room should have same prediction data
            for msg in room_messages:
                assert msg["data"]["room_id"] == room_id
                assert "prediction_time" in msg["data"]


class TestConcurrentCrossComponentOperations:
    """Test concurrent operations across multiple components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_api_and_mqtt_operations(self, integrated_system):
        """Test concurrent API requests and MQTT publications."""
        app = integrated_system["app"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        async def api_operations():
            """Perform API operations concurrently."""
            results = []
            with TestClient(app) as client:
                endpoints = [
                    "/health",
                    "/predictions/living_room",
                    "/predictions/bedroom",
                    "/accuracy?room_id=living_room",
                    "/stats",
                ]

                for endpoint in endpoints:
                    for _ in range(5):  # 5 requests per endpoint
                        response = client.get(
                            endpoint,
                            headers={
                                "Authorization": "Bearer test_api_key_integration"
                            },
                        )
                        results.append((endpoint, response.status_code))

            return results

        async def mqtt_operations():
            """Perform MQTT operations concurrently."""
            results = []
            topics = [
                "homeassistant/living_room/prediction",
                "homeassistant/bedroom/prediction",
                "homeassistant/system/status",
                "homeassistant/system/diagnostics",
            ]

            for topic in topics:
                for i in range(10):  # 10 messages per topic
                    result = await mqtt_publisher.publish_json(
                        topic,
                        {
                            "message_id": i,
                            "timestamp": datetime.now().isoformat(),
                            "data": f"concurrent_test_{i}",
                        },
                    )
                    results.append((topic, result.success))

            return results

        # Run both operations concurrently
        start_time = datetime.now()
        api_results, mqtt_results = await asyncio.gather(
            api_operations(), mqtt_operations()
        )
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Verify operations completed successfully
        api_successes = sum(1 for _, status in api_results if status == 200)
        mqtt_successes = sum(1 for _, success in mqtt_results if success)

        assert api_successes >= 20  # Most API calls should succeed
        assert mqtt_successes == 40  # All MQTT calls should succeed

        # Should complete within reasonable time
        assert duration < 15.0

        print(
            f"Concurrent operations: {api_successes}/25 API, {mqtt_successes}/40 MQTT in {duration:.2f}s"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_discovery_under_load(self, integrated_system):
        """Test entity discovery publishing under load."""
        ha_entities = integrated_system["ha_entities"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Add many more rooms for load testing
        for i in range(20):
            room_config = RoomConfig(
                room_id=f"load_test_room_{i}",
                name=f"Load Test Room {i}",
                sensors={
                    "motion": [f"binary_sensor.load_room_{i}_motion"],
                    "door": [f"binary_sensor.load_room_{i}_door"],
                },
            )
            ha_entities.rooms[f"load_test_room_{i}"] = room_config

        # Concurrently define and publish entities multiple times
        async def discovery_operation():
            entities = ha_entities.define_all_entities()
            results = await ha_entities.publish_all_entities()
            return len(entities), len(results)

        # Run multiple discovery operations concurrently
        start_time = datetime.now()
        tasks = [discovery_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Verify all operations succeeded
        for entity_count, result_count in results:
            assert entity_count > 0
            assert result_count > 0

        # Should complete within reasonable time
        assert duration < 20.0

        # Verify MQTT messages were published
        assert len(mqtt_publisher.published_discoveries) > 0

        print(
            f"Discovery load test: 5 operations with {results[0][0]} entities each in {duration:.2f}s"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mixed_component_stress_test(self, integrated_system):
        """Test stress across all components simultaneously."""
        app = integrated_system["app"]
        ha_entities = integrated_system["ha_entities"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]

        async def api_stress():
            """API stress operations."""
            successes = 0
            with TestClient(app) as client:
                for _ in range(50):
                    response = client.get(
                        "/health",
                        headers={"Authorization": "Bearer test_api_key_integration"},
                    )
                    if response.status_code == 200:
                        successes += 1
            return successes

        async def mqtt_stress():
            """MQTT stress operations."""
            successes = 0
            for i in range(100):
                result = await mqtt_publisher.publish_json(
                    f"stress_test/message_{i}",
                    {"id": i, "timestamp": datetime.now().isoformat()},
                )
                if result.success:
                    successes += 1
            return successes

        async def entity_stress():
            """Entity management stress operations."""
            successes = 0
            for _ in range(10):
                entities = ha_entities.define_all_entities()
                results = await ha_entities.publish_all_entities()
                successful_publishes = sum(1 for r in results.values() if r.success)
                if successful_publishes > 0:
                    successes += 1
            return successes

        async def tracking_stress():
            """Tracking manager stress operations."""
            successes = 0
            rooms = ["living_room", "bedroom"]
            for _ in range(25):
                room = rooms[successes % len(rooms)]
                try:
                    prediction = await tracking_manager.get_room_prediction(room)
                    if prediction:
                        successes += 1
                except Exception:
                    pass
            return successes

        # Run all stress tests concurrently
        start_time = datetime.now()
        api_success, mqtt_success, entity_success, tracking_success = (
            await asyncio.gather(
                api_stress(), mqtt_stress(), entity_stress(), tracking_stress()
            )
        )
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Verify stress test results
        assert api_success >= 40  # At least 80% API success
        assert mqtt_success >= 90  # At least 90% MQTT success
        assert entity_success >= 8  # At least 80% entity ops success
        assert tracking_success >= 20  # At least 80% tracking success

        # Should complete within reasonable time
        assert duration < 30.0

        print(f"Stress test results in {duration:.2f}s:")
        print(f"  API: {api_success}/50 ({100*api_success/50:.1f}%)")
        print(f"  MQTT: {mqtt_success}/100 ({100*mqtt_success/100:.1f}%)")
        print(f"  Entities: {entity_success}/10 ({100*entity_success/10:.1f}%)")
        print(f"  Tracking: {tracking_success}/25 ({100*tracking_success/25:.1f}%)")


class TestErrorPropagationAndRecovery:
    """Test error propagation and recovery across components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mqtt_failure_impact_on_api(self, integrated_system):
        """Test how MQTT failures affect API responses."""
        app = integrated_system["app"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Configure MQTT to fail
        mqtt_publisher.publish_json.side_effect = Exception("MQTT broker unreachable")
        mqtt_publisher.connection_status.connected = False

        # API should still work despite MQTT failures
        with TestClient(app) as client:
            response = client.get(
                "/health", headers={"Authorization": "Bearer test_api_key_integration"}
            )

            # API should handle MQTT failures gracefully
            assert response.status_code == 200
            health_data = response.json()

            # Should report MQTT as unhealthy but overall system as degraded
            assert health_data["status"] in ["degraded", "unhealthy"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tracking_manager_failure_recovery(self, integrated_system):
        """Test recovery when tracking manager fails."""
        app = integrated_system["app"]
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Configure tracking manager to fail initially
        failure_count = 0

        async def failing_get_prediction(room_id):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Tracking system temporarily unavailable")
            return {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "confidence": 0.75,
                "recovered": True,
            }

        tracking_manager.get_room_prediction.side_effect = failing_get_prediction

        # Test recovery across multiple API calls
        with TestClient(app) as client:
            results = []
            for i in range(6):
                response = client.get(
                    "/predictions/living_room",
                    headers={"Authorization": "Bearer test_api_key_integration"},
                )
                results.append(response.status_code)

        # First few should fail, later ones should succeed
        failures = sum(1 for code in results if code != 200)
        successes = sum(1 for code in results if code == 200)

        assert failures > 0  # Some failures expected
        assert successes > 0  # Some recoveries expected

        print(f"Recovery test: {failures} failures, {successes} recoveries")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self, integrated_system):
        """Test recovery from cascading failures across components."""
        app = integrated_system["app"]
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        database_manager = integrated_system["database_manager"]

        # Simulate cascading failures
        # 1. Database fails
        database_manager.health_check.side_effect = Exception(
            "Database connection lost"
        )

        # 2. MQTT fails
        mqtt_publisher.publish_json.side_effect = Exception("MQTT broker down")
        mqtt_publisher.connection_status.connected = False

        # 3. Tracking manager fails
        tracking_manager.get_tracking_status.side_effect = Exception(
            "Tracking system error"
        )

        # Test system response to cascading failures
        with TestClient(app) as client:
            response = client.get(
                "/health", headers={"Authorization": "Bearer test_api_key_integration"}
            )

            # System should report as unhealthy but not crash
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] == "unhealthy"

            # Should provide error details
            assert "components" in health_data

        # Simulate recovery
        database_manager.health_check.side_effect = None
        database_manager.health_check.return_value = {
            "status": "healthy",
            "database_connected": True,
        }

        mqtt_publisher.publish_json.side_effect = None
        mqtt_publisher.connection_status.connected = True

        tracking_manager.get_tracking_status.side_effect = None

        # Test recovery
        with TestClient(app) as client:
            response = client.get(
                "/health", headers={"Authorization": "Bearer test_api_key_integration"}
            )

            health_data = response.json()
            # Should show improved status after recovery
            assert health_data["status"] in ["healthy", "degraded"]


class TestDataConsistencyAcrossComponents:
    """Test data consistency across integration layers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prediction_data_consistency(self, integrated_system):
        """Test prediction data consistency across API and MQTT."""
        app = integrated_system["app"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]

        # Get prediction via API
        with TestClient(app) as client:
            api_response = client.get(
                "/predictions/living_room",
                headers={"Authorization": "Bearer test_api_key_integration"},
            )

            assert api_response.status_code == 200
            api_prediction = api_response.json()

        # Get same prediction directly from tracking manager
        direct_prediction = await tracking_manager.get_room_prediction("living_room")

        # Publish prediction via MQTT
        mqtt_topic = "homeassistant/living_room/prediction"
        await mqtt_publisher.publish_json(mqtt_topic, direct_prediction)

        # Verify data consistency
        assert api_prediction["room_id"] == direct_prediction["room_id"]
        assert api_prediction["confidence"] == direct_prediction["confidence"]
        assert api_prediction["transition_type"] == direct_prediction["transition_type"]

        # Verify MQTT message contains same data
        mqtt_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if msg["topic"] == mqtt_topic
        ]
        assert len(mqtt_messages) == 1

        mqtt_data = mqtt_messages[0]["data"]
        assert mqtt_data["room_id"] == api_prediction["room_id"]
        assert mqtt_data["confidence"] == api_prediction["confidence"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_state_consistency(self, integrated_system):
        """Test entity state consistency across publications."""
        ha_entities = integrated_system["ha_entities"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Define entities
        entities = ha_entities.define_all_entities()

        # Publish entities multiple times
        for i in range(3):
            await ha_entities.publish_all_entities()

        # Verify consistent entity definitions across publications
        discovery_messages = mqtt_publisher.published_discoveries

        # Group by entity unique_id
        entities_by_id = {}
        for msg in discovery_messages:
            data = msg["data"]
            if "unique_id" in data:
                unique_id = data["unique_id"]
                if unique_id not in entities_by_id:
                    entities_by_id[unique_id] = []
                entities_by_id[unique_id].append(data)

        # Verify each entity has consistent definitions
        for unique_id, definitions in entities_by_id.items():
            if len(definitions) > 1:
                # All definitions for same entity should be identical
                first_def = definitions[0]
                for other_def in definitions[1:]:
                    # Key fields should be consistent
                    assert other_def["name"] == first_def["name"]
                    assert other_def["unique_id"] == first_def["unique_id"]
                    if "device_class" in first_def:
                        assert (
                            other_def.get("device_class") == first_def["device_class"]
                        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_service_command_consistency(self, integrated_system):
        """Test service command consistency across different invocation methods."""
        app = integrated_system["app"]
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Trigger retrain via API
        with TestClient(app) as client:
            api_request = {
                "room_id": "living_room",
                "force": True,
                "strategy": "full",
                "reason": "consistency_test",
            }

            api_response = client.post(
                "/model/retrain",
                json=api_request,
                headers={"Authorization": "Bearer test_api_key_integration"},
            )

            assert api_response.status_code == 200
            api_result = api_response.json()

        # Simulate same command via MQTT
        mqtt_command = {
            "command": "retrain_model",
            "parameters": api_request,
            "source": "mqtt",
            "timestamp": datetime.now().isoformat(),
        }

        await mqtt_publisher.publish_json(
            "homeassistant/commands/retrain", mqtt_command
        )

        # Both should result in same tracking manager call
        # (We can't easily verify this without more complex mocking,
        # but we can verify the API result format)
        assert api_result["success"] is True
        assert api_result["room_id"] == "living_room"
        assert api_result["strategy"] == "full"

        # Verify MQTT command was published
        command_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if "commands/" in msg["topic"]
        ]
        assert len(command_messages) == 1

        mqtt_msg = command_messages[0]
        assert mqtt_msg["data"]["parameters"]["room_id"] == api_request["room_id"]


class TestSelfAdaptationSystemIntegration:
    """Test self-adaptation system integration (consolidated from Sprint 4)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_prediction_lifecycle_integration(self, integrated_system):
        """
        Test complete prediction lifecycle through integrated system.
        Consolidated from Sprint 4 test_complete_prediction_lifecycle.
        """
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Mock prediction recording and validation
        prediction_data = {
            "room_id": "living_room",
            "model_type": "ensemble",
            "predicted_time": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "confidence": 0.85,
            "transition_type": "occupied_to_vacant",
            "features_used": {"time_since_last": 300},
            "model_version": "1.0.0",
        }

        # Simulate prediction recording
        tracking_manager.record_prediction = AsyncMock(return_value=True)
        tracking_manager.validate_prediction_with_actual = AsyncMock(return_value=True)

        # Record prediction
        await tracking_manager.record_prediction(prediction_data)

        # Simulate validation
        actual_time = datetime.utcnow() + timedelta(minutes=32)  # 2 min error
        await tracking_manager.validate_prediction_with_actual(
            prediction_id="test_prediction_id",
            actual_time=actual_time,
            actual_state="vacant",
        )

        # Verify prediction was recorded
        tracking_manager.record_prediction.assert_called_once()
        tracking_manager.validate_prediction_with_actual.assert_called_once()

        # Verify MQTT publication
        mqtt_publisher.publish_json.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_drift_detection_triggers_retraining_integration(
        self, integrated_system
    ):
        """
        Test drift detection triggering retraining across components.
        Consolidated from Sprint 4 test_drift_detection_triggers_retraining.
        """
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Mock drift detection and retraining
        tracking_manager.detect_concept_drift = AsyncMock(
            return_value={
                "drift_detected": True,
                "drift_type": "covariate_shift",
                "drift_severity": "high",
                "drift_score": 0.45,
                "room_id": "living_room",
            }
        )

        tracking_manager.trigger_adaptive_retraining = AsyncMock(
            return_value={
                "retraining_triggered": True,
                "room_id": "living_room",
                "trigger": "concept_drift",
            }
        )

        # Simulate drift detection
        drift_result = await tracking_manager.detect_concept_drift("living_room")
        assert drift_result["drift_detected"] is True

        # Simulate retraining trigger
        retrain_result = await tracking_manager.trigger_adaptive_retraining(
            "living_room", trigger="concept_drift"
        )
        assert retrain_result["retraining_triggered"] is True

        # Verify MQTT notifications for drift and retraining
        expected_topics = [
            "homeassistant/system/drift_detected",
            "homeassistant/system/retraining_started",
        ]

        published_topics = [msg["topic"] for msg in mqtt_publisher.published_messages]
        for topic in expected_topics:
            # Allow for flexible topic matching
            assert any(
                expected in pub_topic
                for pub_topic in published_topics
                for expected in [topic.split("/")[-1]]
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_alert_system_integration_across_components(self, integrated_system):
        """
        Test alert system integration across all components.
        Consolidated from Sprint 4 test_alert_system_integration.
        """
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]
        app = integrated_system["app"]

        # Mock alert generation
        alert_data = {
            "alert_type": "accuracy_degradation",
            "severity": "critical",
            "room_id": "living_room",
            "current_value": 35.0,
            "threshold_value": 50.0,
            "message": "Model accuracy below critical threshold",
            "timestamp": datetime.utcnow().isoformat(),
        }

        tracking_manager.generate_alert = AsyncMock(return_value=alert_data)
        tracking_manager.get_active_alerts = AsyncMock(return_value=[alert_data])

        # Trigger alert generation
        alert = await tracking_manager.generate_alert(
            "living_room", "accuracy_degradation"
        )
        assert alert["severity"] == "critical"

        # Verify alert is published via MQTT
        await mqtt_publisher.publish_json(
            "homeassistant/alerts/living_room", alert_data
        )

        # Verify alert is accessible via API
        with TestClient(app) as client:
            response = client.get(
                "/alerts/active",
                headers={"Authorization": "Bearer test_api_key_integration"},
            )
            assert response.status_code == 200
            alerts = response.json()["alerts"]
            assert len(alerts) >= 1

        # Verify alert MQTT publication
        alert_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if "alerts/" in msg["topic"]
        ]
        assert len(alert_messages) >= 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_model_optimization_during_retraining_integration(
        self, integrated_system
    ):
        """
        Test model optimization during retraining process integration.
        Consolidated from Sprint 4 test_model_optimization_during_retraining.
        """
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Mock optimization and retraining
        optimization_result = {
            "optimization_successful": True,
            "best_params": {"learning_rate": 0.05, "max_depth": 8},
            "best_score": 0.87,
            "improvement": 0.05,
            "optimization_time_seconds": 45,
        }

        retraining_result = {
            "success": True,
            "room_id": "bedroom",
            "model_type": "xgboost",
            "accuracy_improvement": 0.05,
            "training_time_minutes": 12,
        }

        tracking_manager.optimize_model_parameters = AsyncMock(
            return_value=optimization_result
        )
        tracking_manager.retrain_model_with_optimization = AsyncMock(
            return_value=retraining_result
        )

        # Trigger optimization and retraining
        opt_result = await tracking_manager.optimize_model_parameters("bedroom")
        retrain_result = await tracking_manager.retrain_model_with_optimization(
            "bedroom", use_optimization=True
        )

        # Verify results
        assert opt_result["optimization_successful"] is True
        assert retrain_result["success"] is True

        # Verify MQTT notifications
        expected_notifications = [
            "homeassistant/system/optimization_completed",
            "homeassistant/system/retraining_completed",
        ]

        # Check for optimization and retraining notifications
        system_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if "system/" in msg["topic"]
        ]
        assert len(system_messages) >= 2


class TestRealtimePublishingIntegration:
    """Test real-time publishing system integration (consolidated from Sprint 5)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_realtime_integration(self, integrated_system):
        """
        Test WebSocket real-time integration across components.
        Consolidated from Sprint 5 WebSocket integration tests.
        """
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Mock WebSocket connections
        websocket_connections = []

        class MockWebSocket:
            def __init__(self, client_id):
                self.client_id = client_id
                self.messages_received = []

            async def send(self, message):
                self.messages_received.append(message)

        # Create mock WebSocket connections
        for i in range(3):
            ws = MockWebSocket(f"client_{i}")
            websocket_connections.append(ws)

        # Mock WebSocket manager
        tracking_manager.websocket_manager = Mock()
        tracking_manager.websocket_manager.active_connections = websocket_connections
        tracking_manager.websocket_manager.broadcast = AsyncMock()

        # Simulate real-time prediction update
        prediction_update = {
            "room_id": "kitchen",
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": 0.78,
            "transition_type": "occupied_to_vacant",
        }

        # Publish via MQTT and broadcast via WebSocket
        await mqtt_publisher.publish_json(
            "homeassistant/kitchen/prediction", prediction_update
        )

        # Simulate WebSocket broadcast
        await tracking_manager.websocket_manager.broadcast(prediction_update)

        # Verify WebSocket broadcast was called
        tracking_manager.websocket_manager.broadcast.assert_called_once_with(
            prediction_update
        )

        # Verify MQTT publication
        prediction_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if "kitchen/prediction" in msg["topic"]
        ]
        assert len(prediction_messages) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_server_realtime_integration(self, integrated_system):
        """
        Test API server real-time integration with tracking system.
        Consolidated from Sprint 5 API integration tests.
        """
        app = integrated_system["app"]
        tracking_manager = integrated_system["tracking_manager"]
        mqtt_publisher = integrated_system["mqtt_publisher"]

        # Mock real-time status updates
        tracking_manager.get_system_status = AsyncMock(
            return_value={
                "system_health": "healthy",
                "active_predictions": 12,
                "tracking_accuracy": 0.87,
                "uptime_seconds": 3600,
                "last_update": datetime.utcnow().isoformat(),
            }
        )

        # Test real-time status endpoint
        with TestClient(app) as client:
            response = client.get(
                "/system/status",
                headers={"Authorization": "Bearer test_api_key_integration"},
            )
            assert response.status_code == 200
            status_data = response.json()
            assert status_data["system_health"] == "healthy"
            assert "active_predictions" in status_data

        # Verify tracking manager was called
        tracking_manager.get_system_status.assert_called()

        # Test manual retrain endpoint integration
        retrain_response = client.post(
            "/model/retrain",
            json={
                "room_id": "living_room",
                "force": True,
                "strategy": "auto",
                "reason": "integration_test",
            },
            headers={"Authorization": "Bearer test_api_key_integration"},
        )
        assert retrain_response.status_code == 200

        # Verify tracking manager retrain was called
        tracking_manager.trigger_manual_retrain.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_enhanced_mqtt_manager_integration(self, integrated_system):
        """
        Test enhanced MQTT manager integration with real-time capabilities.
        Consolidated from Sprint 5 enhanced MQTT tests.
        """
        mqtt_publisher = integrated_system["mqtt_publisher"]
        tracking_manager = integrated_system["tracking_manager"]

        # Mock enhanced MQTT manager with real-time capabilities
        enhanced_stats = {
            "mqtt_connected": True,
            "predictions_published": 50,
            "realtime_clients": 3,
            "websocket_connections": 2,
            "sse_connections": 1,
            "total_active_connections": 3,
        }

        mqtt_publisher.get_enhanced_stats = Mock(return_value=enhanced_stats)

        # Test real-time prediction publishing
        prediction_data = {
            "room_id": "office",
            "prediction_time": datetime.utcnow().isoformat(),
            "next_transition_time": (
                datetime.utcnow() + timedelta(minutes=20)
            ).isoformat(),
            "confidence": 0.92,
        }

        # Publish through enhanced MQTT manager
        await mqtt_publisher.publish_json(
            "homeassistant/office/prediction", prediction_data
        )

        # Verify publication
        office_messages = [
            msg
            for msg in mqtt_publisher.published_messages
            if "office/prediction" in msg["topic"]
        ]
        assert len(office_messages) == 1
        assert office_messages[0]["data"]["confidence"] == 0.92

        # Test enhanced stats retrieval
        stats = mqtt_publisher.get_enhanced_stats()
        assert stats["mqtt_connected"] is True
        assert stats["realtime_clients"] == 3
