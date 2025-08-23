"""
Integration Test Configuration and Fixtures.

This module provides configuration and shared fixtures for all integration tests,
ensuring consistent test environment setup and teardown across the test suite.
"""

import asyncio
import logging
import os
from typing import Any, Dict

import pytest

# Configure test logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress verbose logs from external libraries during testing
logging.getLogger("docker").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line(
        "markers", "container: mark test as requiring Docker containers"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running (> 30 seconds)")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Mark all tests in integration directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if (
            "performance" in item.name
            or "load" in item.name
            or "throughput" in item.name
        ):
            item.add_marker(pytest.mark.performance)

        # Mark container tests
        if "container" in item.name or "broker" in item.name:
            item.add_marker(pytest.mark.container)

        # Mark slow tests
        if "sustained" in item.name or "long_running" in item.name:
            item.add_marker(pytest.mark.slow)


# Remove deprecated event_loop fixture - pytest-asyncio handles this automatically
# @pytest.fixture(scope="session")
# def event_loop():
#     """Create an instance of the default event loop for the test session."""
#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     yield loop
#     loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "INFO",
        "JWT_SECRET_KEY": "test_secret_key_for_integration_testing",
        "API_KEY": "test_api_key_for_integration_testing",
        "PYTEST_RUNNING": "true",
    }

    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def integration_test_config():
    """Provide configuration for integration tests."""
    return {
        "mqtt": {
            "test_broker_host": "localhost",
            "test_broker_port": 1883,
            "connection_timeout": 10,
            "message_timeout": 5,
        },
        "database": {
            "test_host": "localhost",
            "test_port": 5432,
            "test_database": "test_ha_ml",
            "test_user": "test_user",
            "test_password": "test_password",
        },
        "api": {
            "test_host": "localhost",
            "test_port": 8000,
            "test_api_key": "test_api_key_for_integration_testing",
            "request_timeout": 30,
        },
        "performance": {
            "high_load_message_count": 1000,
            "concurrent_clients": 10,
            "sustained_duration_seconds": 60,
            "memory_limit_mb": 500,
            "cpu_limit_percent": 80,
        },
        "container": {
            "startup_timeout": 60,
            "cleanup_timeout": 30,
            "network_name_prefix": "ha_ml_test",
        },
    }


@pytest.fixture
async def cleanup_tasks():
    """Provide cleanup task management for tests."""
    cleanup_functions = []

    def add_cleanup(func, *args, **kwargs):
        """Add a cleanup function to be called at test end."""
        cleanup_functions.append((func, args, kwargs))

    yield add_cleanup

    # Execute all cleanup functions
    for func, args, kwargs in reversed(cleanup_functions):
        try:
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Cleanup function failed: {e}")


@pytest.fixture
def test_data_generator():
    """Provide test data generation utilities."""
    from datetime import datetime, timedelta
    import random

    class TestDataGenerator:
        @staticmethod
        def generate_room_config(room_id: str = None, num_sensors: int = 3):
            """Generate realistic room configuration."""
            if room_id is None:
                room_id = f"test_room_{random.randint(1000, 9999)}"

            sensors = {}
            sensor_types = ["motion", "door", "temperature", "humidity", "light"]

            for i in range(num_sensors):
                sensor_type = sensor_types[i % len(sensor_types)]
                if sensor_type not in sensors:
                    sensors[sensor_type] = []
                sensors[sensor_type].append(f"sensor.{room_id}_{sensor_type}_{i}")

            return {
                "room_id": room_id,
                "name": room_id.replace("_", " ").title(),
                "sensors": sensors,
            }

        @staticmethod
        def generate_prediction_data(room_id: str = "test_room"):
            """Generate realistic prediction data."""
            return {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "next_transition_time": (
                    datetime.now() + timedelta(minutes=random.randint(5, 120))
                ).isoformat(),
                "transition_type": random.choice(
                    ["occupied", "vacant", "occupied_to_vacant", "vacant_to_occupied"]
                ),
                "confidence": random.uniform(0.7, 0.95),
                "time_until_transition": f"{random.randint(5, 120)} minutes",
                "alternatives": [],
                "model_info": {
                    "model": random.choice(["lstm", "xgboost", "ensemble"]),
                    "accuracy": random.uniform(0.8, 0.95),
                },
            }

        @staticmethod
        def generate_sensor_events(room_id: str = "test_room", count: int = 100):
            """Generate realistic sensor event sequence."""
            events = []
            current_time = datetime.now() - timedelta(hours=24)

            for i in range(count):
                event = {
                    "room_id": room_id,
                    "sensor_id": f"sensor.{room_id}_motion",
                    "sensor_type": "motion",
                    "state": random.choice(["on", "off"]),
                    "timestamp": current_time.isoformat(),
                    "attributes": {
                        "device_class": "motion",
                        "battery_level": random.randint(80, 100),
                    },
                }
                events.append(event)
                current_time += timedelta(minutes=random.randint(1, 30))

            return events

        @staticmethod
        def generate_mqtt_messages(topic_prefix: str = "test", count: int = 50):
            """Generate realistic MQTT messages."""
            messages = []

            for i in range(count):
                message = {
                    "topic": f"{topic_prefix}/room_{i % 5}/data",
                    "payload": {
                        "id": i,
                        "timestamp": datetime.now().isoformat(),
                        "sensor_data": {
                            "temperature": random.uniform(18, 25),
                            "humidity": random.uniform(30, 70),
                            "motion": random.choice([True, False]),
                        },
                    },
                }
                messages.append(message)

            return messages

    return TestDataGenerator()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.metrics = {}

        def start_monitoring(self):
            """Start performance monitoring."""
            self.start_time = time.time()
            self.metrics = {
                "initial_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
                "initial_cpu": self.process.cpu_percent(),
                "memory_samples": [],
                "cpu_samples": [],
            }

        def sample_metrics(self):
            """Sample current performance metrics."""
            if self.start_time is None:
                return

            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            current_cpu = self.process.cpu_percent()

            self.metrics["memory_samples"].append(current_memory)
            self.metrics["cpu_samples"].append(current_cpu)

        def get_report(self):
            """Get performance report."""
            if not self.metrics:
                return {}

            end_time = time.time()
            duration = end_time - self.start_time if self.start_time else 0

            memory_samples = self.metrics["memory_samples"]
            cpu_samples = self.metrics["cpu_samples"]

            report = {
                "duration_seconds": duration,
                "initial_memory_mb": self.metrics["initial_memory"],
                "final_memory_mb": memory_samples[-1] if memory_samples else 0,
                "peak_memory_mb": max(memory_samples) if memory_samples else 0,
                "avg_memory_mb": (
                    sum(memory_samples) / len(memory_samples) if memory_samples else 0
                ),
                "avg_cpu_percent": (
                    sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
                ),
                "peak_cpu_percent": max(cpu_samples) if cpu_samples else 0,
                "memory_increase_mb": (
                    (memory_samples[-1] - self.metrics["initial_memory"])
                    if memory_samples
                    else 0
                ),
            }

            return report

    return PerformanceMonitor()


@pytest.fixture
def network_condition_simulator():
    """Provide network condition simulation utilities."""
    import asyncio
    import random

    class NetworkConditionSimulator:
        def __init__(self):
            self.latency_ms = 0
            self.packet_loss_rate = 0.0
            self.bandwidth_limit = None
            self.partition_active = False

        async def simulate_latency(self, latency_ms: int):
            """Simulate network latency."""
            self.latency_ms = latency_ms
            if latency_ms > 0:
                await asyncio.sleep(latency_ms / 1000)

        def simulate_packet_loss(self, loss_rate: float):
            """Simulate packet loss."""
            self.packet_loss_rate = loss_rate

        def should_drop_packet(self) -> bool:
            """Determine if packet should be dropped."""
            return random.random() < self.packet_loss_rate

        async def simulate_network_partition(self, duration_seconds: int):
            """Simulate network partition."""
            self.partition_active = True
            await asyncio.sleep(duration_seconds)
            self.partition_active = False

        def is_partitioned(self) -> bool:
            """Check if network is currently partitioned."""
            return self.partition_active

    return NetworkConditionSimulator()


# Custom pytest markers for test categorization (fixed marker definitions)
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.container = pytest.mark.container
pytest.mark.slow = pytest.mark.slow
pytest.mark.network = pytest.mark.network


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip container tests if Docker is not available
    if item.get_closest_marker("container"):
        try:
            import docker

            docker.from_env().ping()
        except Exception:
            pytest.skip("Docker not available for container tests")

    # Skip performance tests in CI if requested
    if item.get_closest_marker("performance") and os.environ.get(
        "SKIP_PERFORMANCE_TESTS"
    ):
        pytest.skip("Performance tests skipped in CI environment")

    # Skip slow tests if requested
    if item.get_closest_marker("slow") and os.environ.get("SKIP_SLOW_TESTS"):
        pytest.skip("Slow tests skipped")


def pytest_runtest_teardown(item):
    """Tear down individual test runs."""
    # Force garbage collection after each test to help with memory management
    import gc

    gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    # Log test session summary
    if hasattr(session.config, "_testmon"):
        # If testmon is available, log coverage information
        logging.info("Integration test session completed")

    # Final cleanup
    import gc

    gc.collect()


# Test timeout configuration - Cross-platform implementation
@pytest.fixture(autouse=True)
def test_timeout():
    """Apply reasonable timeouts to all integration tests (cross-platform)."""
    import threading
    import time

    import platform

    timeout_occurred = False
    timer_thread = None

    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        logging.error("Test timeout occurred after 5 minutes")

    # Use cross-platform timer approach instead of Unix signals
    if platform.system() != "Windows":
        # On Unix systems, we can still use signals if available
        try:
            import signal

            def signal_handler(signum, frame):
                raise TimeoutError("Test timed out after 5 minutes")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(300)  # 5 minutes
        except (AttributeError, ImportError):
            # Fallback to threading timer
            timer_thread = threading.Timer(300.0, timeout_handler)
            timer_thread.daemon = True
            timer_thread.start()
    else:
        # Windows - use threading timer
        timer_thread = threading.Timer(300.0, timeout_handler)
        timer_thread.daemon = True
        timer_thread.start()

    yield

    # Clean up timeout
    if platform.system() != "Windows":
        try:
            import signal

            signal.alarm(0)
        except (AttributeError, ImportError):
            if timer_thread and timer_thread.is_alive():
                timer_thread.cancel()
    else:
        if timer_thread and timer_thread.is_alive():
            timer_thread.cancel()

    # Check if timeout occurred
    if timeout_occurred:
        pytest.fail("Test exceeded 5 minute timeout")
