"""
Comprehensive Security Validation Testing for Sprint 6 Task 6 Integration Test Coverage.

This module provides security testing scenarios to validate authentication, authorization,
input validation, rate limiting, and protection against common security vulnerabilities.

Test Coverage:
- Authentication bypass and token validation testing
- Authorization and access control testing
- Input validation and SQL injection protection
- Rate limiting and abuse prevention
- API security boundary testing
- Cross-site scripting (XSS) protection validation
- Security header validation and HTTPS enforcement
- Sensitive data exposure prevention
"""

# CRITICAL: Set test environment variables BEFORE any imports
# This prevents JWT configuration errors during module imports
import os
os.environ["ENVIRONMENT"] = "test"
os.environ["CI"] = "true"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"
os.environ["JWT_REFRESH_TOKEN_EXPIRE_DAYS"] = "30"
os.environ["JWT_ISSUER"] = "ha-ml-predictor-test"
os.environ["JWT_AUDIENCE"] = "ha-ml-predictor-api-test"
os.environ["JWT_REQUIRE_HTTPS"] = "false"
os.environ["API_KEY_ENABLED"] = "true"
os.environ["API_KEY"] = "test_api_key_for_security_validation_testing"
os.environ["DEBUG"] = "true"

import asyncio
import base64
from datetime import datetime, timedelta
import hashlib
import logging
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx

# JWT functionality mocked - no actual JWT library needed for testing
import pytest
import pytest_asyncio
import secrets

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import get_config
from src.core.exceptions import (
    APIAuthenticationError,
    APIAuthorizationError,
    APIRateLimitError,
    APISecurityError,
    ErrorSeverity,
)
from src.integration.api_server import APIServer, create_app

logger = logging.getLogger(__name__)


@pytest.fixture
async def security_test_config():
    """Configuration for security testing scenarios."""
    return {
        "test_api_key": "test_api_key_12345",
        "invalid_api_key": "invalid_key_67890",
        "rate_limit_per_minute": 60,
        "rate_limit_burst": 10,
        "jwt_secret": "test_jwt_secret_key_for_testing_only",
        "sql_injection_payloads": [
            "'; DROP TABLE sensor_events; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO admin (user) VALUES ('hacker'); --",
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//'",
        ],
    }


@pytest.fixture
async def real_jwt_manager():
    """Real JWT manager for security testing."""
    from src.core.config import JWTConfig
    from src.integration.auth.jwt_manager import JWTManager

    # Create test JWT configuration
    jwt_config = JWTConfig(
        enabled=True,
        secret_key="test_jwt_secret_key_for_testing_security_validation_at_least_32_chars",
        algorithm="HS256",
        access_token_expire_minutes=60,
        refresh_token_expire_days=30,
        issuer="ha-ml-predictor-test",
        audience="ha-ml-predictor-api-test",
        blacklist_enabled=True,
    )

    return JWTManager(jwt_config)


class TestAuthenticationSecurity:
    """Test authentication security and bypass attempts."""

    def _create_expired_token(self, jwt_manager, user_id: str) -> str:
        """Create an expired JWT token for testing."""
        # Temporarily modify the expiration time to create an expired token
        original_expire_minutes = jwt_manager.config.access_token_expire_minutes
        jwt_manager.config.access_token_expire_minutes = -1  # Expired immediately

        try:
            token = jwt_manager.generate_access_token(user_id, ["read", "write"])
            return token
        finally:
            # Restore original expiration time
            jwt_manager.config.access_token_expire_minutes = original_expire_minutes

    async def test_authentication_bypass_attempts(
        self, security_test_config, real_jwt_manager
    ):
        """Test various authentication bypass attempts."""
        jwt_manager = real_jwt_manager

        # Create test app with authentication
        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = [
                "living_room",
                "bedroom",
            ]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Test cases for authentication bypass
            bypass_attempts = [
                # No authentication header
                {"headers": {}, "expected_status": 401},
                # Empty authorization header
                {"headers": {"Authorization": ""}, "expected_status": 401},
                # Invalid bearer token format
                {
                    "headers": {"Authorization": "InvalidFormat token123"},
                    "expected_status": 401,
                },
                # Malformed JWT token
                {
                    "headers": {"Authorization": "Bearer invalid.jwt.token"},
                    "expected_status": 401,
                },
                # Expired token (create one manually by modifying JWT manager config temporarily)
                {
                    "headers": {
                        "Authorization": f'Bearer {self._create_expired_token(jwt_manager, "test_user")}'
                    },
                    "expected_status": 401,
                },
                # SQL injection in token
                {
                    "headers": {"Authorization": "Bearer '; DROP TABLE users; --"},
                    "expected_status": 401,
                },
                # XSS in authorization header
                {
                    "headers": {
                        "Authorization": 'Bearer <script>alert("xss")</script>'
                    },
                    "expected_status": 401,
                },
            ]

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                for i, attempt in enumerate(bypass_attempts):
                    response = await client.get(
                        "/api/rooms", headers=attempt["headers"], timeout=5.0
                    )

                    assert response.status_code == attempt["expected_status"], (
                        f"Bypass attempt {i} failed: expected {attempt['expected_status']}, "
                        f"got {response.status_code}"
                    )

                    # Ensure no sensitive data in error response
                    response_text = response.text.lower()
                    sensitive_data = [
                        "password",
                        "secret",
                        "key",
                        "token",
                        "jwt",
                    ]
                    for sensitive in sensitive_data:
                        assert (
                            sensitive not in response_text
                        ), f"Sensitive data '{sensitive}' exposed in error response"

            logger.info(
                f"Authentication bypass test completed: {len(bypass_attempts)} attempts blocked"
            )

    async def test_token_validation_and_expiration(
        self, security_test_config, real_jwt_manager
    ):
        """Test proper token validation and expiration handling."""
        jwt_manager = real_jwt_manager

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Create valid token
            valid_token = jwt_manager.generate_access_token(
                "test_user", ["read", "write"]
            )

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                # Test valid token access
                response = await client.get(
                    "/api/health",
                    headers={"Authorization": f"Bearer {valid_token}"},
                    timeout=5.0,
                )
                assert response.status_code == 200, "Valid token should allow access"

                # Test token revocation
                jwt_manager.revoke_token(valid_token)

                # Simulate token validation with revoked token
                try:
                    jwt_manager.validate_token(valid_token)
                    assert False, "Revoked token should not validate"
                except APIAuthenticationError as e:
                    assert "revoked" in str(e).lower()

                # Test expired token
                expired_token = self._create_expired_token(jwt_manager, "test_user")

                try:
                    jwt_manager.validate_token(expired_token)
                    assert False, "Expired token should not validate"
                except APIAuthenticationError as e:
                    assert "expired" in str(e).lower()

            logger.info("Token validation and expiration test completed successfully")

    async def test_rate_limiting_security(self, security_test_config):
        """Test rate limiting to prevent abuse and DoS attacks."""
        rate_limit = security_test_config["rate_limit_per_minute"]

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Track request timestamps for rate limiting simulation
            request_times = []
            rate_limited_count = 0
            successful_requests = 0

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                # Attempt requests beyond rate limit
                for i in range(rate_limit + 20):  # Exceed rate limit
                    try:
                        start_time = time.time()
                        response = await client.get(
                            "/api/health",
                            timeout=2.0,  # Short timeout for rate limit testing
                        )
                        request_times.append(start_time)

                        if response.status_code == 429:  # Too Many Requests
                            rate_limited_count += 1
                        elif 200 <= response.status_code < 300:
                            successful_requests += 1

                        # Small delay between requests
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            rate_limited_count += 1
                        else:
                            logger.warning(
                                f"Request {i} failed with unexpected error: {e}"
                            )

                # Analyze rate limiting effectiveness
                total_requests = successful_requests + rate_limited_count
                rate_limit_effectiveness = rate_limited_count / max(total_requests, 1)

                # Validate rate limiting
                assert rate_limited_count > 0, "Rate limiting should have triggered"
                assert (
                    rate_limit_effectiveness > 0.1
                ), f"Rate limiting not effective enough: {rate_limit_effectiveness}"
                assert (
                    successful_requests <= rate_limit * 1.1
                ), f"Too many successful requests: {successful_requests}"

            logger.info(
                f"Rate limiting test: {rate_limited_count} requests rate limited, "
                f"{successful_requests} successful"
            )


class TestInputValidationSecurity:
    """Test input validation and injection attack prevention."""

    async def test_sql_injection_prevention(self, security_test_config):
        """Test protection against SQL injection attacks."""
        sql_payloads = security_test_config["sql_injection_payloads"]

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_room_metrics.return_value = {"accuracy": 0.85}
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                sql_injection_blocked = 0

                for i, payload in enumerate(sql_payloads):
                    # Test SQL injection in URL parameters
                    try:
                        # URL-encode the payload
                        encoded_payload = quote(payload, safe="")

                        response = await client.get(
                            f"/api/rooms/{encoded_payload}/metrics",
                            timeout=5.0,
                        )

                        # Check response for security
                        assert response.status_code in [
                            400,
                            404,
                            422,
                        ], f"SQL injection payload {i} not properly rejected: {response.status_code}"

                        # Ensure no database error information leaked
                        response_text = response.text.lower()
                        db_error_indicators = [
                            "syntax error",
                            "sql",
                            "database",
                            "table",
                            "column",
                            "postgresql",
                            "timescaledb",
                            "constraint",
                        ]

                        for indicator in db_error_indicators:
                            assert (
                                indicator not in response_text
                            ), f"Database error information leaked: '{indicator}' in response"

                        sql_injection_blocked += 1

                    except Exception as e:
                        logger.warning(f"SQL injection test {i} failed: {e}")

                # Test SQL injection in JSON body
                for payload in sql_payloads[:3]:  # Test subset for body injection
                    try:
                        malicious_data = {
                            "room_id": payload,
                            "action": "update_settings",
                        }

                        response = await client.post(
                            "/api/rooms/living_room/settings",
                            json=malicious_data,
                            timeout=5.0,
                        )

                        # Should be rejected with proper error code
                        assert response.status_code in [
                            400,
                            422,
                            404,
                        ], "JSON SQL injection not properly rejected"

                        sql_injection_blocked += 1

                    except Exception as e:
                        logger.warning(f"JSON SQL injection test failed: {e}")

                # Validate SQL injection prevention
                assert sql_injection_blocked >= len(
                    sql_payloads
                ), f"Only {sql_injection_blocked}/{len(sql_payloads)} SQL injections blocked"

            logger.info(
                f"SQL injection prevention test: {sql_injection_blocked} attacks blocked"
            )

    async def test_xss_prevention(self, security_test_config):
        """Test protection against cross-site scripting (XSS) attacks."""
        xss_payloads = security_test_config["xss_payloads"]

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                xss_attacks_blocked = 0

                for i, payload in enumerate(xss_payloads):
                    # Test XSS in URL parameters
                    try:
                        encoded_payload = quote(payload, safe="")

                        response = await client.get(
                            f"/api/search?query={encoded_payload}", timeout=5.0
                        )

                        # Check that response doesn't contain unescaped script content
                        response_text = response.text

                        # Scripts should be escaped or removed
                        dangerous_patterns = [
                            "<script",
                            "javascript:",
                            "onerror=",
                            "onload=",
                            "alert(",
                            "eval(",
                            "document.cookie",
                        ]

                        for pattern in dangerous_patterns:
                            assert (
                                pattern.lower() not in response_text.lower()
                            ), f"XSS payload {i} not properly sanitized: '{pattern}' found in response"

                        xss_attacks_blocked += 1

                    except Exception as e:
                        logger.warning(f"XSS test {i} failed: {e}")

                # Test XSS in JSON body
                for payload in xss_payloads[:2]:  # Test subset
                    try:
                        malicious_data = {
                            "name": payload,
                            "description": f"Room with {payload}",
                        }

                        response = await client.post(
                            "/api/rooms", json=malicious_data, timeout=5.0
                        )

                        if response.status_code < 500:  # Don't count server errors
                            response_text = response.text

                            # Verify XSS content is sanitized
                            assert "<script" not in response_text.lower()
                            assert "javascript:" not in response_text.lower()

                            xss_attacks_blocked += 1

                    except Exception as e:
                        logger.warning(f"JSON XSS test failed: {e}")

                # Validate XSS prevention
                assert xss_attacks_blocked >= len(
                    xss_payloads
                ), f"Only {xss_attacks_blocked}/{len(xss_payloads)} XSS attacks blocked"

            logger.info(f"XSS prevention test: {xss_attacks_blocked} attacks blocked")

    async def test_input_size_and_format_validation(self, security_test_config):
        """Test input size limits and format validation."""

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                validation_tests_passed = 0

                # Test oversized input
                large_payload = "x" * 100000  # 100KB payload

                try:
                    response = await client.post(
                        "/api/rooms",
                        json={"name": large_payload, "description": "test"},
                        timeout=10.0,
                    )

                    # Should reject oversized input
                    assert response.status_code in [
                        400,
                        413,
                        422,
                    ], f"Oversized input not rejected: {response.status_code}"

                    validation_tests_passed += 1

                except Exception as e:
                    if "413" in str(e) or "request entity too large" in str(e).lower():
                        validation_tests_passed += 1
                    else:
                        logger.warning(f"Oversized input test failed: {e}")

                # Test malformed JSON
                try:
                    response = await client.post(
                        "/api/rooms",
                        data='{"name": "test", invalid json}',
                        headers={"Content-Type": "application/json"},
                        timeout=5.0,
                    )

                    assert response.status_code in [
                        400,
                        422,
                    ], f"Malformed JSON not rejected: {response.status_code}"

                    validation_tests_passed += 1

                except Exception as e:
                    logger.warning(f"Malformed JSON test failed: {e}")

                # Test invalid data types
                invalid_data_tests = [
                    {"room_id": 123},  # Should be string
                    {"timestamp": "not-a-date"},  # Invalid date format
                    {"confidence": "high"},  # Should be numeric
                    {"settings": "not-an-object"},  # Should be object
                ]

                for invalid_data in invalid_data_tests:
                    try:
                        response = await client.post(
                            "/api/rooms/living_room/update",
                            json=invalid_data,
                            timeout=5.0,
                        )

                        # Should reject invalid data types
                        assert response.status_code in [
                            400,
                            422,
                        ], f"Invalid data type not rejected: {invalid_data}"

                        validation_tests_passed += 1

                    except Exception as e:
                        logger.warning(f"Data type validation test failed: {e}")

                # Validate input validation effectiveness
                expected_tests = (
                    1 + 1 + len(invalid_data_tests)
                )  # oversized + malformed + data types
                assert (
                    validation_tests_passed >= expected_tests * 0.8
                ), f"Input validation not comprehensive enough: {validation_tests_passed}/{expected_tests}"

            logger.info(
                f"Input validation test: {validation_tests_passed} validation checks passed"
            )


class TestAPISecurityBoundaries:
    """Test API security boundaries and access controls."""

    async def test_unauthorized_endpoint_access(
        self, security_test_config, real_jwt_manager
    ):
        """Test access controls on protected endpoints."""
        jwt_manager = real_jwt_manager

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Define protected endpoints and required permissions
            protected_endpoints = [
                {
                    "method": "GET",
                    "path": "/api/admin/status",
                    "required_perm": "admin",
                },
                {
                    "method": "POST",
                    "path": "/api/rooms",
                    "required_perm": "write",
                },
                {
                    "method": "DELETE",
                    "path": "/api/rooms/living_room",
                    "required_perm": "admin",
                },
                {
                    "method": "PUT",
                    "path": "/api/system/config",
                    "required_perm": "admin",
                },
                {
                    "method": "POST",
                    "path": "/api/model/retrain",
                    "required_perm": "admin",
                },
            ]

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                unauthorized_access_blocked = 0

                for endpoint in protected_endpoints:
                    # Test unauthorized access (no token)
                    try:
                        if endpoint["method"] == "GET":
                            response = await client.get(endpoint["path"], timeout=5.0)
                        elif endpoint["method"] == "POST":
                            response = await client.post(
                                endpoint["path"], json={}, timeout=5.0
                            )
                        elif endpoint["method"] == "PUT":
                            response = await client.put(
                                endpoint["path"], json={}, timeout=5.0
                            )
                        elif endpoint["method"] == "DELETE":
                            response = await client.delete(
                                endpoint["path"], timeout=5.0
                            )

                        assert (
                            response.status_code == 401
                        ), f"Unauthorized access not blocked for {endpoint['path']}"

                        unauthorized_access_blocked += 1

                    except Exception as e:
                        logger.warning(
                            f"Unauthorized access test failed for {endpoint['path']}: {e}"
                        )

                # Test insufficient permissions
                read_only_token = jwt_manager.generate_access_token(
                    "read_user", ["read"]
                )

                for endpoint in protected_endpoints:
                    if endpoint["required_perm"] in ["write", "admin"]:
                        try:
                            headers = {"Authorization": f"Bearer {read_only_token}"}

                            if endpoint["method"] == "GET":
                                response = await client.get(
                                    endpoint["path"],
                                    headers=headers,
                                    timeout=5.0,
                                )
                            elif endpoint["method"] == "POST":
                                response = await client.post(
                                    endpoint["path"],
                                    json={},
                                    headers=headers,
                                    timeout=5.0,
                                )
                            elif endpoint["method"] == "PUT":
                                response = await client.put(
                                    endpoint["path"],
                                    json={},
                                    headers=headers,
                                    timeout=5.0,
                                )
                            elif endpoint["method"] == "DELETE":
                                response = await client.delete(
                                    endpoint["path"],
                                    headers=headers,
                                    timeout=5.0,
                                )

                            assert response.status_code in [
                                403,
                                404,
                            ], f"Insufficient permission not blocked for {endpoint['path']}"

                            unauthorized_access_blocked += 1

                        except Exception as e:
                            logger.warning(
                                f"Insufficient permission test failed for {endpoint['path']}: {e}"
                            )

                # Validate access control effectiveness
                expected_blocks = (
                    len(protected_endpoints) * 2
                )  # unauthorized + insufficient perms
                assert (
                    unauthorized_access_blocked >= expected_blocks * 0.7
                ), f"Access controls not effective: {unauthorized_access_blocked}/{expected_blocks}"

            logger.info(
                f"API security boundaries test: {unauthorized_access_blocked} access attempts blocked"
            )

    async def test_sensitive_data_exposure_prevention(self, security_test_config):
        """Test prevention of sensitive data exposure in API responses."""

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_system_status.return_value = {
                "status": "healthy",
                "database_url": "postgresql://user:password@localhost/db",  # Should be filtered
                "api_key": "secret_api_key_12345",  # Should be filtered
                "internal_token": "internal_secret_token",  # Should be filtered
            }
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            sensitive_data_patterns = [
                "password",
                "secret",
                "key",
                "token",
                "credential",
                "postgresql://",
                "mysql://",
                "mongodb://",
                "aws_access_key",
                "private_key",
                "jwt_secret",
            ]

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                # Test endpoints that might expose sensitive data
                test_endpoints = [
                    "/api/health",
                    "/api/system/info",
                    "/api/config/status",
                ]

                data_exposure_prevented = 0

                for endpoint in test_endpoints:
                    try:
                        response = await client.get(endpoint, timeout=5.0)

                        if 200 <= response.status_code < 300:
                            response_text = response.text.lower()

                            # Check for sensitive data exposure
                            exposed_data = []
                            for pattern in sensitive_data_patterns:
                                if pattern in response_text:
                                    # Additional check - make sure it's not just the field name
                                    if (
                                        f'"{pattern}":' not in response_text
                                        and pattern not in ["key", "token"]
                                    ):
                                        exposed_data.append(pattern)

                            assert (
                                len(exposed_data) == 0
                            ), f"Sensitive data exposed in {endpoint}: {exposed_data}"

                            data_exposure_prevented += 1

                    except Exception as e:
                        logger.warning(
                            f"Sensitive data test failed for {endpoint}: {e}"
                        )

                # Test error responses don't expose sensitive info
                try:
                    response = await client.get("/api/nonexistent", timeout=5.0)

                    if response.status_code >= 400:
                        error_text = response.text.lower()

                        # Error responses should not contain sensitive info
                        error_sensitive_patterns = [
                            "traceback",
                            "stack trace",
                            "/home/",
                            "/opt/",
                            "database connection",
                            "sql error",
                        ]

                        for pattern in error_sensitive_patterns:
                            assert (
                                pattern not in error_text
                            ), f"Sensitive error info exposed: {pattern}"

                        data_exposure_prevented += 1

                except Exception as e:
                    logger.warning(f"Error response test failed: {e}")

                # Validate data exposure prevention
                assert data_exposure_prevented >= len(
                    test_endpoints
                ), f"Sensitive data exposure prevention insufficient: {data_exposure_prevented}"

            logger.info(
                f"Sensitive data exposure prevention test: {data_exposure_prevented} tests passed"
            )


class TestSecurityHeadersAndHTTPS:
    """Test security headers and HTTPS enforcement."""

    async def test_security_headers_validation(self, security_test_config):
        """Test presence and configuration of security headers."""

        with patch("src.integration.api_server.get_tracking_manager") as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = ["living_room"]
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Expected security headers
            required_headers = {
                "X-Content-Type-Options": "nosnif",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Content-Security-Policy": "default-src",  # Should contain CSP
                "Strict-Transport-Security": "max-age",  # Should contain HSTS
            }

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                response = await client.get("/api/health", timeout=5.0)

                security_headers_present = 0

                for header, expected_value in required_headers.items():
                    if header in response.headers:
                        header_value = response.headers[header]

                        if isinstance(expected_value, list):
                            # Multiple valid values
                            header_valid = any(
                                val in header_value for val in expected_value
                            )
                        else:
                            # Single expected substring
                            header_valid = (
                                expected_value.lower() in header_value.lower()
                            )

                        if header_valid:
                            security_headers_present += 1
                            logger.info(
                                f"Security header '{header}' properly configured: {header_value}"
                            )
                        else:
                            logger.warning(
                                f"Security header '{header}' misconfigured: {header_value}"
                            )
                    else:
                        logger.warning(f"Security header '{header}' missing")

                # Check for information disclosure headers that should be removed
                info_disclosure_headers = [
                    "Server",
                    "X-Powered-By",
                    "X-AspNet-Version",
                    "X-AspNetMvc-Version",
                    "X-Rack-Cache",
                ]

                headers_hidden = 0
                for header in info_disclosure_headers:
                    if header not in response.headers:
                        headers_hidden += 1
                    else:
                        logger.warning(
                            f"Information disclosure header present: {header}"
                        )

                # Validate security headers
                header_score = security_headers_present / len(required_headers)
                assert (
                    header_score >= 0.6
                ), f"Insufficient security headers: {security_headers_present}/{len(required_headers)}"

                # Validate information hiding
                hiding_score = headers_hidden / len(info_disclosure_headers)
                assert (
                    hiding_score >= 0.8
                ), f"Too many information disclosure headers: {headers_hidden}/{len(info_disclosure_headers)}"

            logger.info(
                f"Security headers test: {security_headers_present}/{len(required_headers)} present, "
                f"{headers_hidden}/{len(info_disclosure_headers)} hidden"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
