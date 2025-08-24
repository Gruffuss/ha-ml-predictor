"""Pytest configuration for the HA ML Predictor project.

This conftest.py file sets up the test environment BEFORE any modules are imported,
ensuring that configuration dependencies (like JWT_SECRET_KEY) are available.
"""

import os
import sys
from pathlib import Path

# Set test environment variables BEFORE any imports
os.environ["ENVIRONMENT"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "true"

# Set JWT configuration for tests to avoid configuration errors
os.environ["JWT_ENABLED"] = "true"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_comprehensive_security_validation_testing_minimum_32_characters_required_for_hmac_sha256_algorithm"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_EXPIRATION_MINUTES"] = "30"

# Set database configuration for tests
os.environ.setdefault("DB_CONNECTION_STRING", "postgresql://test_user:test_pass@localhost:5432/test_db")

# Set other test configurations
os.environ.setdefault("API_KEY_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("BACKGROUND_TASKS_ENABLED", "false")

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import pytest and fixtures after environment is set up
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to ensure test isolation."""
    # Import after environment is set
    from src.core import config
    
    # Reset config singleton
    if hasattr(config, '_config_instance'):
        config._config_instance = None
    
    yield
    
    # Clean up after test
    if hasattr(config, '_config_instance'):
        config._config_instance = None


@pytest.fixture
def mock_config():
    """Provide a mock configuration for tests."""
    from src.core.config import SystemConfig, APIConfig, DatabaseConfig, MQTTConfig, JWTConfig
    
    mock_cfg = MagicMock(spec=SystemConfig)
    
    # Set up API config
    mock_cfg.api = MagicMock(spec=APIConfig)
    mock_cfg.api.enabled = True
    mock_cfg.api.host = "127.0.0.1"
    mock_cfg.api.port = 8000
    mock_cfg.api.api_key_enabled = False
    mock_cfg.api.rate_limit_enabled = False
    mock_cfg.api.background_tasks_enabled = False
    
    # Set up JWT config
    mock_cfg.jwt = MagicMock(spec=JWTConfig)
    mock_cfg.jwt.enabled = True
    mock_cfg.jwt.secret_key = "test_jwt_secret_key_for_comprehensive_security_validation_testing_minimum_32_characters_required_for_hmac_sha256_algorithm"
    mock_cfg.jwt.algorithm = "HS256"
    mock_cfg.jwt.expiration_minutes = 30
    
    # Set up database config
    mock_cfg.database = MagicMock(spec=DatabaseConfig)
    mock_cfg.database.connection_string = "postgresql://test_user:test_pass@localhost:5432/test_db"
    
    # Set up MQTT config
    mock_cfg.mqtt = MagicMock(spec=MQTTConfig)
    mock_cfg.mqtt.enabled = False
    
    # Set up rooms
    mock_cfg.rooms = {
        "living_room": MagicMock(name="Living Room"),
        "bedroom": MagicMock(name="Bedroom"),
        "kitchen": MagicMock(name="Kitchen"),
    }
    
    return mock_cfg


@pytest.fixture
def mock_tracking_manager():
    """Provide a mock tracking manager for tests."""
    from src.adaptation.tracking_manager import TrackingManager
    
    mock_manager = AsyncMock(spec=TrackingManager)
    mock_manager.get_prediction = AsyncMock()
    mock_manager.validate_prediction = AsyncMock()
    mock_manager.get_accuracy_metrics = AsyncMock()
    mock_manager.trigger_retrain = AsyncMock()
    mock_manager.refresh_models = AsyncMock()
    mock_manager.get_model_info = AsyncMock()
    mock_manager.cleanup = AsyncMock()
    
    return mock_manager


@pytest.fixture
def mock_database_manager():
    """Provide a mock database manager for tests."""
    from src.data.storage.database import DatabaseManager
    
    mock_db = AsyncMock(spec=DatabaseManager)
    mock_db.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_db.get_session = AsyncMock()
    mock_db.cleanup = AsyncMock()
    
    return mock_db


@pytest.fixture
def mock_health_monitor():
    """Provide a mock health monitor for tests."""
    from src.utils.health_monitor import HealthMonitor, HealthStatus
    
    mock_monitor = MagicMock(spec=HealthMonitor)
    mock_monitor.get_health_status = MagicMock(return_value=HealthStatus.HEALTHY)
    mock_monitor.check_component_health = AsyncMock(return_value=True)
    mock_monitor.record_metric = MagicMock()
    
    return mock_monitor