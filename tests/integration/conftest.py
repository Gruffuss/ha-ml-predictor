"""
Shared test fixtures and configuration for integration tests.

This module provides common fixtures and setup for integration testing,
including database connections, mock services, and test data.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import os
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.config import (
    DatabaseConfig,
    HomeAssistantConfig,
    JWTConfig,
    MQTTConfig,
    RoomConfig,
    SystemConfig,
)
from src.models.base.predictor import PredictionResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_system_config():
    """Create test system configuration."""
    return SystemConfig(
        home_assistant=HomeAssistantConfig(
            url="http://localhost:8123",
            token="test_token_for_ha_integration",
            websocket_timeout=30,
            api_timeout=10,
        ),
        database=DatabaseConfig(
            connection_string="postgresql+asyncpg://test:test@localhost:5432/test_ha_predictor",
            pool_size=5,
            max_overflow=10,
        ),
        mqtt=MQTTConfig(
            broker="localhost",
            port=1883,
            username="test_user",
            password="test_password",
            topic_prefix="occupancy/predictions",
        ),
        jwt=JWTConfig(
            secret_key="test-secret-key-that-is-definitely-long-enough-for-security-testing",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
            issuer="ha-ml-predictor-test",
            audience="ha-ml-predictor-api-test",
            blacklist_enabled=True,
        ),
        rooms={
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={
                    "motion": ["binary_sensor.living_room_motion"],
                    "door": ["binary_sensor.living_room_door"],
                    "temperature": ["sensor.living_room_temperature"],
                },
            ),
            "kitchen": RoomConfig(
                room_id="kitchen",
                name="Kitchen",
                sensors={
                    "motion": ["binary_sensor.kitchen_motion"],
                    "door": ["binary_sensor.kitchen_door"],
                    "light": ["sensor.kitchen_light"],
                },
            ),
            "bedroom": RoomConfig(
                room_id="bedroom",
                name="Main Bedroom",
                sensors={
                    "motion": ["binary_sensor.bedroom_motion"],
                    "door": ["binary_sensor.bedroom_door"],
                    "temperature": ["sensor.bedroom_temperature"],
                },
            ),
        },
    )


@pytest.fixture
def test_room_configs():
    """Create test room configurations."""
    return {
        "living_room": RoomConfig(
            room_id="living_room",
            name="Living Room",
            sensors={
                "motion": ["binary_sensor.living_room_motion"],
                "door": ["binary_sensor.living_room_door"],
            },
        ),
        "kitchen": RoomConfig(
            room_id="kitchen",
            name="Kitchen",
            sensors={
                "motion": ["binary_sensor.kitchen_motion"],
                "door": ["binary_sensor.kitchen_door"],
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
    }


@pytest.fixture
def test_mqtt_config():
    """Create test MQTT configuration."""
    return MQTTConfig(
        broker="localhost",
        port=1883,
        username="test_mqtt_user",
        password="test_mqtt_password",
        topic_prefix="occupancy/predictions",
    )


@pytest.fixture
def test_jwt_config():
    """Create test JWT configuration."""
    return JWTConfig(
        secret_key="test-secret-key-that-is-definitely-long-enough-for-security-testing",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
        issuer="ha-ml-predictor-test",
        audience="ha-ml-predictor-api-test",
        blacklist_enabled=True,
    )


@pytest.fixture
def sample_prediction_result():
    """Create a sample prediction result for testing."""
    return PredictionResult(
        predicted_time=datetime.now(timezone.utc) + timedelta(minutes=15),
        transition_type="occupied",
        confidence_score=0.85,
        model_type="ensemble",
        model_version="1.0.0",
        features_used=[
            "time_since_last_occupancy",
            "hour_of_day",
            "day_of_week",
            "recent_transitions",
        ],
        alternatives=[
            (datetime.now(timezone.utc) + timedelta(minutes=10), 0.72),
            (datetime.now(timezone.utc) + timedelta(minutes=20), 0.68),
        ],
        prediction_metadata={
            "feature_count": 15,
            "model_confidence": 0.91,
            "data_quality_score": 0.88,
            "prediction_latency_ms": 45,
        },
    )


@pytest.fixture
def sample_prediction_results():
    """Create multiple sample prediction results for testing."""
    base_time = datetime.now(timezone.utc)

    return [
        PredictionResult(
            predicted_time=base_time + timedelta(minutes=5),
            transition_type="vacant",
            confidence_score=0.92,
            model_type="lstm",
            model_version="1.0.0",
        ),
        PredictionResult(
            predicted_time=base_time + timedelta(minutes=15),
            transition_type="occupied",
            confidence_score=0.78,
            model_type="xgboost",
            model_version="1.0.0",
        ),
        PredictionResult(
            predicted_time=base_time + timedelta(hours=2),
            transition_type="vacant",
            confidence_score=0.65,
            model_type="hmm",
            model_version="1.0.0",
        ),
    ]


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.send = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_sse_queue():
    """Create mock SSE queue."""
    return asyncio.Queue()


@pytest.fixture
def mock_prediction_publisher():
    """Create mock prediction publisher."""
    from src.integration.mqtt_publisher import MQTTPublishResult

    publisher = Mock()
    publisher.publish_prediction = AsyncMock(
        return_value=MQTTPublishResult(success=True, error_message=None)
    )
    return publisher


@pytest.fixture
def mock_ha_client():
    """Create mock Home Assistant client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.get_entity_history = AsyncMock(return_value=[])
    client.subscribe_to_events = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and cleanup."""
    # Set test environment
    os.environ["TESTING"] = "1"
    os.environ["DISABLE_BACKGROUND_TASKS"] = "1"

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "DISABLE_BACKGROUND_TASKS" in os.environ:
        del os.environ["DISABLE_BACKGROUND_TASKS"]


@pytest.fixture
def mock_database():
    """Create mock database connection."""
    db = AsyncMock()
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    db.execute = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.health_check = AsyncMock(return_value={"status": "healthy"})
    return db


@pytest.fixture
def sample_sensor_events():
    """Create sample sensor events for testing."""
    base_time = datetime.now(timezone.utc)

    return [
        {
            "entity_id": "binary_sensor.living_room_motion",
            "state": "on",
            "last_changed": base_time - timedelta(minutes=5),
            "attributes": {"device_class": "motion"},
        },
        {
            "entity_id": "binary_sensor.living_room_motion",
            "state": "off",
            "last_changed": base_time - timedelta(minutes=2),
            "attributes": {"device_class": "motion"},
        },
        {
            "entity_id": "binary_sensor.kitchen_motion",
            "state": "on",
            "last_changed": base_time - timedelta(minutes=1),
            "attributes": {"device_class": "motion"},
        },
    ]


@pytest.fixture
def test_user_data():
    """Create test user data for authentication tests."""
    return {
        "admin": {
            "user_id": "admin",
            "username": "admin",
            "email": "admin@test.local",
            "password": "admin123!",
            "permissions": ["read", "write", "admin", "model_retrain"],
            "roles": ["admin"],
            "is_admin": True,
            "is_active": True,
        },
        "operator": {
            "user_id": "operator",
            "username": "operator",
            "email": "operator@test.local",
            "password": "operator123!",
            "permissions": ["read", "write", "prediction_view"],
            "roles": ["operator"],
            "is_admin": False,
            "is_active": True,
        },
        "viewer": {
            "user_id": "viewer",
            "username": "viewer",
            "email": "viewer@test.local",
            "password": "viewer123!",
            "permissions": ["read", "prediction_view"],
            "roles": ["viewer"],
            "is_admin": False,
            "is_active": True,
        },
    }


@pytest.fixture
def sample_realtime_events():
    """Create sample real-time events for testing."""
    from src.integration.realtime_publisher import RealtimePredictionEvent

    base_time = datetime.now(timezone.utc)

    return [
        RealtimePredictionEvent(
            event_id="event-1",
            event_type="prediction",
            timestamp=base_time,
            room_id="living_room",
            data={
                "predicted_time": (base_time + timedelta(minutes=10)).isoformat(),
                "confidence": 0.85,
                "transition_type": "occupied",
            },
        ),
        RealtimePredictionEvent(
            event_id="event-2",
            event_type="system_status",
            timestamp=base_time + timedelta(seconds=30),
            room_id=None,
            data={
                "status": "healthy",
                "uptime": 3600,
                "active_models": 3,
            },
        ),
        RealtimePredictionEvent(
            event_id="event-3",
            event_type="alert",
            timestamp=base_time + timedelta(minutes=1),
            room_id="kitchen",
            data={
                "alert_type": "accuracy_degradation",
                "message": "Model accuracy below threshold",
                "severity": "medium",
            },
        ),
    ]


@pytest.fixture
async def async_mock_context():
    """Create async context for testing async operations."""

    class AsyncContextManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return AsyncContextManager()


@pytest.fixture
def performance_metrics():
    """Create sample performance metrics."""
    return {
        "prediction_latency_p95": 150.5,
        "prediction_latency_p99": 250.3,
        "accuracy_last_24h": 0.87,
        "model_confidence_avg": 0.82,
        "total_predictions": 1547,
        "successful_predictions": 1503,
        "failed_predictions": 44,
        "error_rate": 0.028,
        "uptime_seconds": 86400,
        "memory_usage_mb": 512.3,
        "cpu_usage_percent": 15.7,
    }


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "auth: marks tests as authentication tests")
    config.addinivalue_line(
        "markers", "realtime: marks tests as real-time publishing tests"
    )
    config.addinivalue_line("markers", "jwt: marks tests as JWT-related tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark auth tests
        if "auth" in str(item.fspath):
            item.add_marker(pytest.mark.auth)

        # Mark real-time tests
        if "realtime" in str(item.fspath):
            item.add_marker(pytest.mark.realtime)

        # Mark JWT tests
        if "jwt" in str(item.fspath):
            item.add_marker(pytest.mark.jwt)

        # Mark async tests that might be slow
        if getattr(item, "function", None) and asyncio.iscoroutinefunction(
            item.function
        ):
            if not item.get_closest_marker("slow"):
                # Add slow marker to complex async tests
                if any(
                    keyword in item.name.lower()
                    for keyword in ["full", "integration", "end_to_end", "complete"]
                ):
                    item.add_marker(pytest.mark.slow)
