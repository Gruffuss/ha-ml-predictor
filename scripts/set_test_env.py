#!/usr/bin/env python3
"""
Set environment variables for testing JWT authentication system.

This script sets up the required environment variables for testing
the JWT authentication system with proper security configurations.
"""

import os
import sys


def set_test_environment():
    """Set environment variables for testing."""

    # JWT secret key (must be at least 32 characters)
    os.environ["JWT_SECRET_KEY"] = (
        "test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long"
    )

    # Test database configuration
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test_db"

    # Test mode flag
    os.environ["ENVIRONMENT"] = "test"

    # Disable HTTPS requirement for testing
    os.environ["JWT_REQUIRE_HTTPS"] = "false"

    # Enable debug mode for testing
    os.environ["DEBUG"] = "true"

    print("Test environment variables set successfully:")
    print(f"  JWT_SECRET_KEY: {'*' * len(os.environ['JWT_SECRET_KEY'])}")
    print(f"  DATABASE_URL: {os.environ['DATABASE_URL']}")
    print(f"  ENVIRONMENT: {os.environ['ENVIRONMENT']}")
    print(f"  JWT_REQUIRE_HTTPS: {os.environ['JWT_REQUIRE_HTTPS']}")
    print(f"  DEBUG: {os.environ['DEBUG']}")


if __name__ == "__main__":
    set_test_environment()
