#!/usr/bin/env python3
"""
Test script for API deployment and configuration.

This script tests the complete FastAPI deployment with proper environment
configuration and validates that all endpoints are accessible.
"""

import asyncio
import os
import sys
from threading import Thread
import time

import requests
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Set test environment before importing anything else
def setup_test_environment():
    """Configure test environment variables for deployment testing."""

    # Core JWT Configuration
    os.environ["JWT_SECRET_KEY"] = (
        "test_jwt_secret_key_for_deployment_testing_at_least_32_characters_long"
    )
    os.environ["JWT_ENABLED"] = "false"  # Disable JWT for deployment testing

    # API Configuration
    os.environ["API_ENABLED"] = "true"
    os.environ["API_HOST"] = "127.0.0.1"
    os.environ["API_PORT"] = "8002"
    os.environ["API_DEBUG"] = "true"
    os.environ["API_INCLUDE_DOCS"] = "true"
    os.environ["API_ACCESS_LOG"] = "false"
    os.environ["API_LOG_REQUESTS"] = "false"
    os.environ["API_LOG_RESPONSES"] = "false"
    os.environ["API_BACKGROUND_TASKS_ENABLED"] = "false"

    # API Key Configuration (disabled for testing)
    os.environ["API_KEY_ENABLED"] = "false"
    os.environ["API_RATE_LIMIT_ENABLED"] = "false"

    # Test database configuration - use PostgreSQL format but non-existent server for testing
    # This tests configuration loading without actual database connection
    os.environ["DATABASE_URL"] = (
        "postgresql://test_user:test_pass@localhost:5433/test_deployment_db"
    )

    # Test mode flags
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "true"

    # Security configuration
    os.environ["CORS_ENABLED"] = "true"
    os.environ["CORS_ALLOW_ORIGINS"] = "*"

    print("OK Test environment configured for deployment testing")


def run_server_background(host: str, port: int):
    """Run FastAPI server in background thread."""
    try:
        from src.integration.api_server import app

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Configure and run server
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="error",  # Reduce noise during testing
            access_log=False,
        )
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    except Exception as e:
        print(f"ERROR Server failed to start: {e}")


def test_endpoint(
    url: str, name: str, expected_status: int = 200, timeout: int = 5
) -> bool:
    """Test a single API endpoint."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == expected_status:
            print(f"OK {name}: {response.status_code}")
            return True
        else:
            print(
                f"ERROR {name}: Expected {expected_status}, got {response.status_code}"
            )
            if response.status_code >= 400:
                print(f"   Error response: {response.text[:200]}...")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR {name}: Connection failed - {e}")
        return False


def main():
    """Test API deployment with proper configuration."""
    print("=" * 60)
    print("FastAPI Deployment Test")
    print("=" * 60)

    # Setup test environment
    setup_test_environment()

    # Import configuration to validate it loads correctly
    try:
        from src.core.config import get_config

        config = get_config()
        print("OK Configuration loaded successfully")
        print(f"   API enabled: {config.api.enabled}")
        print(f"   JWT enabled: {config.api.jwt.enabled}")
        print(f"   Rate limiting: {config.api.rate_limit_enabled}")
        print(f"   Background tasks: {config.api.background_tasks_enabled}")
    except Exception as e:
        print(f"ERROR Configuration loading failed: {e}")
        return False

    # Test FastAPI app creation
    try:
        from src.integration.api_server import app

        print("OK FastAPI app created successfully")
        print(f"   Routes registered: {len(app.routes)}")
    except Exception as e:
        print(f"ERROR FastAPI app creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Start server in background
    host = "127.0.0.1"
    port = 8002
    print(f"\nStarting test server on {host}:{port}...")

    server_thread = Thread(target=run_server_background, args=(host, port), daemon=True)
    server_thread.start()

    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)

    # Test endpoints
    base_url = f"http://{host}:{port}"
    tests = [
        (f"{base_url}/", "Root endpoint"),
        (f"{base_url}/health", "Health endpoint"),
        (f"{base_url}/predictions/living_kitchen", "Predictions endpoint"),
        (f"{base_url}/accuracy", "Accuracy endpoint"),
        (f"{base_url}/stats", "Stats endpoint"),
        (f"{base_url}/docs", "API documentation"),
    ]

    print("\nTesting API endpoints...")
    success_count = 0

    for url, name in tests:
        if test_endpoint(url, name):
            success_count += 1

    # Test results
    total_tests = len(tests)
    print("\nTest Results:")
    print(f"   Successful: {success_count}/{total_tests}")
    print(f"   Failed: {total_tests - success_count}/{total_tests}")

    if success_count == total_tests:
        print("\nAll deployment tests passed!")
        print("   The API server is properly configured and running.")
        return True
    else:
        print("\nSome tests failed. Check configuration and dependencies.")
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("DEPLOYMENT TEST SUCCESSFUL")
        print("FastAPI server is ready for production deployment.")
    else:
        print("DEPLOYMENT TEST FAILED")
        print("Fix the issues above before deploying.")
    print("=" * 60)

    sys.exit(0 if success else 1)
