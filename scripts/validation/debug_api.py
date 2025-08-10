#!/usr/bin/env python3
"""Quick debug script to test API creation."""

import asyncio
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


def test_api_creation():
    """Test if we can create the API app."""

    # Mock the config
    mock_config = Mock()
    mock_config.api = Mock()
    mock_config.api.enabled = True
    mock_config.api.host = "0.0.0.0"
    mock_config.api.port = 8000
    mock_config.api.debug = False
    mock_config.api.enable_cors = True
    mock_config.api.cors_origins = ["*"]
    mock_config.api.api_key_enabled = False
    mock_config.api.api_key = None
    mock_config.api.rate_limit_enabled = False
    mock_config.api.requests_per_minute = 60
    mock_config.api.burst_limit = 100
    mock_config.api.request_timeout_seconds = 30
    mock_config.api.max_request_size_mb = 10
    mock_config.api.include_docs = True
    mock_config.api.docs_url = "/docs"
    mock_config.api.redoc_url = "/redoc"

    mock_config.rooms = {"living_room": Mock(room_id="living_room", name="Living Room")}

    # Mock dependencies
    with patch("src.core.config.get_config", return_value=mock_config):
        with patch(
            "src.integration.api_server.get_tracking_manager"
        ) as mock_get_tracking:
            with patch(
                "src.integration.api_server.get_database_manager"
            ) as mock_get_db:
                with patch(
                    "src.integration.api_server.get_mqtt_manager"
                ) as mock_get_mqtt:

                    # Mock the tracking manager
                    mock_tracking = Mock()
                    mock_tracking.get_room_prediction = Mock(
                        return_value={"room_id": "living_room", "confidence": 0.85}
                    )
                    mock_get_tracking.return_value = mock_tracking

                    # Mock DB and MQTT managers
                    mock_get_db.return_value = Mock()
                    mock_get_mqtt.return_value = Mock()

                    # Import and create app
                    from src.integration.api_server import create_app

                    app = create_app()
                    print(f"App created: {app}")
                    print(f"Routes: {[route.path for route in app.routes]}")

                    # Test with client but avoid lifespan
                    from starlette.testclient import TestClient as StarletteTestClient

                    client = StarletteTestClient(app)

                    response = client.get("/")
                    print(f"Root response: {response.status_code} - {response.text}")

                    health_response = client.get("/health")
                    print(
                        f"Health response: {health_response.status_code} - {health_response.text}"
                    )

                    client.close()


if __name__ == "__main__":
    test_api_creation()
