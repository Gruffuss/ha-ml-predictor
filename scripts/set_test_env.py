#!/usr/bin/env python3
"""
Set environment variables for testing JWT authentication system.

This script sets up the required environment variables for testing
the JWT authentication system with proper security configurations.
"""

import os
import sys


def set_test_environment():
    """Set comprehensive environment variables for testing."""

    # Core JWT Configuration
    os.environ["JWT_SECRET_KEY"] = (
        "test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long"
    )
    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"
    os.environ["JWT_REFRESH_TOKEN_EXPIRE_DAYS"] = "30"
    os.environ["JWT_ISSUER"] = "ha-ml-predictor-test"
    os.environ["JWT_AUDIENCE"] = "ha-ml-predictor-api-test"
    os.environ["JWT_REQUIRE_HTTPS"] = "false"
    os.environ["JWT_BLACKLIST_ENABLED"] = "true"

    # API Authentication Configuration
    os.environ["API_KEY_ENABLED"] = "true"
    os.environ["API_KEY"] = "test_api_key_for_security_validation_testing"
    os.environ["API_RATE_LIMIT_ENABLED"] = "true"
    os.environ["API_RATE_LIMIT_PER_MINUTE"] = "60"
    os.environ["API_RATE_LIMIT_BURST"] = "10"

    # Test database configuration
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test_db"

    # Test API configuration
    os.environ["API_ENABLED"] = "true"
    os.environ["API_HOST"] = "0.0.0.0"
    os.environ["API_PORT"] = "8001"
    os.environ["API_DEBUG"] = "true"
    os.environ["API_INCLUDE_DOCS"] = "true"
    os.environ["API_ACCESS_LOG"] = "false"  # Reduce noise in tests
    os.environ["API_LOG_REQUESTS"] = "false"

    # JWT Configuration for testing
    os.environ["JWT_ENABLED"] = "false"  # Disable JWT in test mode

    # Disable background tasks for testing
    os.environ["API_BACKGROUND_TASKS_ENABLED"] = "false"
    os.environ["HEALTH_CHECK_INTERVAL_SECONDS"] = "300"

    # Test mode flags
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "true"
    os.environ["CI"] = os.environ.get("CI", "false")

    # Security configuration
    os.environ["SECURITY_HEADERS_ENABLED"] = "true"
    os.environ["CORS_ENABLED"] = "true"
    os.environ["CORS_ALLOW_ORIGINS"] = "*"

    print("Test environment variables set successfully:")
    print(f"  JWT_SECRET_KEY: {'*' * len(os.environ['JWT_SECRET_KEY'])}")
    print(f"  JWT_ALGORITHM: {os.environ['JWT_ALGORITHM']}")
    print(f"  JWT_ISSUER: {os.environ['JWT_ISSUER']}")
    print(f"  JWT_AUDIENCE: {os.environ['JWT_AUDIENCE']}")
    print(f"  JWT_REQUIRE_HTTPS: {os.environ['JWT_REQUIRE_HTTPS']}")
    print(f"  API_KEY_ENABLED: {os.environ['API_KEY_ENABLED']}")
    print(f"  API_KEY: {'*' * len(os.environ['API_KEY'])}")
    print(f"  DATABASE_URL: {os.environ['DATABASE_URL']}")
    print(f"  ENVIRONMENT: {os.environ['ENVIRONMENT']}")
    print(f"  DEBUG: {os.environ['DEBUG']}")
    print(f"  CI: {os.environ['CI']}")

    # Validate JWT configuration
    jwt_secret = os.environ["JWT_SECRET_KEY"]
    if len(jwt_secret) < 32:
        print(
            f"WARNING: JWT secret key is only {len(jwt_secret)} characters, should be at least 32"
        )
    else:
        print(f"  JWT Secret Key Length: {len(jwt_secret)} characters [OK]")

    print("\n[SUCCESS] All environment variables configured for security testing")


if __name__ == "__main__":
    set_test_environment()
