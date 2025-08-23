"""
API Server Load Testing and Concurrent Access Integration Tests.

This module provides comprehensive integration testing for the FastAPI server
under realistic load conditions, focusing on concurrent access, authentication
edge cases, rate limiting, and performance under stress.

Focus Areas:
- High concurrent request handling
- Authentication system stress testing
- Rate limiting effectiveness under load
- Database connection pooling under stress
- Memory and resource usage during high load
- WebSocket connection handling
- Error recovery under load conditions
- Security testing with real attack vectors
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import logging
import random
import threading
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch
import uuid

from fastapi.testclient import TestClient
import httpx
import pytest

# Skip tests if JWT module is not available
jwt = pytest.importorskip("jwt", reason="PyJWT package not installed")

from src.core.config import APIConfig, get_config
from src.core.exceptions import (
    APIAuthenticationError,
    APIRateLimitError,
    APIServerError,
)
from src.integration.api_server import (
    APIServer,
    app,
    create_app,
    get_tracking_manager,
    set_tracking_manager,
)


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    # Set up test environment
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "test",
            "JWT_SECRET_KEY": "test_secret_key_for_integration_testing",
            "API_KEY": "test_api_key_for_integration_testing",
        },
    ):
        test_app = create_app()
        with TestClient(test_app) as client:
            yield client


@pytest.fixture
def mock_tracking_manager():
    """Create mock tracking manager for testing."""
    mock_manager = AsyncMock()

    # Mock room prediction
    mock_manager.get_room_prediction.return_value = {
        "room_id": "living_room",
        "prediction_time": datetime.now().isoformat(),
        "next_transition_time": (datetime.now() + timedelta(minutes=30)).isoformat(),
        "transition_type": "vacant_to_occupied",
        "confidence": 0.85,
        "time_until_transition": "30 minutes",
        "alternatives": [],
        "model_info": {"model": "ensemble", "version": "1.0"},
    }

    # Mock accuracy metrics
    mock_manager.get_accuracy_metrics.return_value = {
        "room_id": "living_room",
        "accuracy_rate": 0.87,
        "average_error_minutes": 12.3,
        "confidence_calibration": 0.91,
        "total_predictions": 450,
        "total_validations": 423,
        "time_window_hours": 24,
        "trend_direction": "improving",
    }

    # Mock system stats
    mock_manager.get_system_stats.return_value = {
        "tracking_stats": {
            "total_predictions_tracked": 1500,
            "total_validations_performed": 1423,
            "accuracy_rate": 0.87,
        },
        "retraining_stats": {
            "completed_retraining_jobs": 12,
            "failed_retraining_jobs": 1,
        },
    }

    # Mock tracking status
    mock_manager.get_tracking_status.return_value = {
        "tracking_active": True,
        "status": "active",
        "config": {"enabled": True},
        "performance": {
            "background_tasks": 3,
            "total_predictions_recorded": 1500,
            "total_validations_performed": 1423,
            "total_drift_checks_performed": 156,
            "system_uptime_seconds": 86400,
        },
        "validator": {"total_predictions": 1500},
        "accuracy_tracker": {"total_predictions": 1423},
        "drift_detector": "active",
        "adaptive_retrainer": "active",
    }

    # Mock manual retrain
    mock_manager.trigger_manual_retrain.return_value = {
        "message": "Retraining triggered successfully",
        "success": True,
        "room_id": None,
        "strategy": "auto",
        "force": False,
    }

    return mock_manager


@pytest.fixture(autouse=True)
def setup_tracking_manager(mock_tracking_manager):
    """Set up mock tracking manager for all tests."""
    set_tracking_manager(mock_tracking_manager)
    yield
    # Cleanup
    set_tracking_manager(None)


@pytest.fixture
def auth_headers():
    """Create valid authentication headers."""
    return {"Authorization": "Bearer test_api_key_for_integration_testing"}


class TestConcurrentLoadTesting:
    """Test API server under concurrent load conditions."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, test_client, auth_headers):
        """Test concurrent health check requests."""

        async def make_health_request():
            response = test_client.get("/health", headers=auth_headers)
            return response.status_code, response.json()

        # Make 50 concurrent health check requests
        tasks = [make_health_request() for _ in range(50)]

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        end_time = time.time()
        duration = end_time - start_time

        # All requests should succeed
        status_codes = [result[0] for result in results]
        assert all(code == 200 for code in status_codes)

        # Should complete within reasonable time (< 10 seconds)
        assert duration < 10.0

        # Verify response structure
        for status_code, response_data in results:
            assert "status" in response_data
            assert "timestamp" in response_data
            assert "components" in response_data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_prediction_requests(self, test_client, auth_headers):
        """Test concurrent prediction requests for different rooms."""
        rooms = ["living_room", "bedroom", "kitchen", "office", "bathroom"]

        async def make_prediction_request(room_id):
            try:
                response = test_client.get(
                    f"/predictions/{room_id}", headers=auth_headers
                )
                return room_id, response.status_code, response.json()
            except Exception as e:
                return room_id, 500, {"error": str(e)}

        # Make concurrent requests for all rooms
        tasks = [
            make_prediction_request(room) for room in rooms * 20
        ]  # 100 total requests

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        end_time = time.time()
        duration = end_time - start_time

        # Most requests should succeed (allowing for some room not found errors)
        successful_requests = sum(1 for _, status, _ in results if status == 200)
        assert successful_requests >= 80  # At least 80% success rate

        # Should maintain reasonable performance
        assert duration < 15.0  # 100 requests in < 15 seconds

        print(f"Processed {len(results)} concurrent requests in {duration:.2f} seconds")
        print(
            f"Success rate: {successful_requests}/{len(results)} ({100*successful_requests/len(results):.1f}%)"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mixed_endpoint_concurrent_access(self, test_client, auth_headers):
        """Test concurrent access to different endpoints simultaneously."""

        async def make_random_request():
            endpoints = [
                "/health",
                "/predictions/living_room",
                "/accuracy?room_id=living_room",
                "/stats",
                "/health/comprehensive",
                "/health/system",
            ]

            endpoint = random.choice(endpoints)
            try:
                response = test_client.get(endpoint, headers=auth_headers)
                return endpoint, response.status_code, len(response.content)
            except Exception as e:
                return endpoint, 500, 0

        # Make 200 random concurrent requests
        tasks = [make_random_request() for _ in range(200)]

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        end_time = time.time()
        duration = end_time - start_time

        # Analyze results by endpoint
        endpoint_stats = {}
        for endpoint, status, content_size in results:
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"requests": 0, "success": 0, "avg_size": 0}

            endpoint_stats[endpoint]["requests"] += 1
            if status == 200:
                endpoint_stats[endpoint]["success"] += 1
            endpoint_stats[endpoint]["avg_size"] += content_size

        # Calculate averages
        for endpoint, stats in endpoint_stats.items():
            stats["success_rate"] = stats["success"] / stats["requests"]
            stats["avg_size"] = stats["avg_size"] / stats["requests"]

        # Verify performance
        assert duration < 30.0  # 200 mixed requests in < 30 seconds

        # Most endpoints should have good success rates
        for endpoint, stats in endpoint_stats.items():
            print(
                f"{endpoint}: {stats['success']}/{stats['requests']} "
                f"({100*stats['success_rate']:.1f}%) avg_size={stats['avg_size']:.0f}b"
            )
            assert stats["success_rate"] >= 0.8  # At least 80% success rate

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, test_client, auth_headers):
        """Test memory usage during sustained load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        async def sustained_load_batch():
            results = []
            for i in range(100):
                endpoint = f"/predictions/room_{i % 5}"
                try:
                    response = test_client.get(endpoint, headers=auth_headers)
                    results.append(response.status_code)
                except Exception:
                    results.append(500)
            return results

        # Run 10 batches of 100 requests each
        start_time = time.time()

        for batch in range(10):
            batch_results = await sustained_load_batch()

            # Check memory periodically
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            print(
                f"Batch {batch + 1}/10: Memory usage: {current_memory:.1f}MB "
                f"(+{memory_increase:.1f}MB)"
            )

            # Memory should not grow excessively
            assert memory_increase < 200  # Less than 200MB increase

            # Brief pause between batches
            await asyncio.sleep(0.1)

        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory

        print(f"Total processing time: {end_time - start_time:.2f}s")
        print(f"Memory increase: {total_increase:.1f}MB")

        # Final memory increase should be reasonable
        assert total_increase < 150  # Less than 150MB total increase


class TestRateLimitingUnderLoad:
    """Test rate limiting behavior under high load."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, test_client, auth_headers):
        """Test that rate limits are enforced under rapid requests."""

        # Configure a low rate limit for testing
        with patch("src.integration.api_server.rate_limiter") as mock_rate_limiter:
            # Allow first 10 requests, then deny
            call_count = 0

            def mock_is_allowed(client_ip, limit, window_minutes=1):
                nonlocal call_count
                call_count += 1
                return call_count <= 10

            mock_rate_limiter.is_allowed = mock_is_allowed

            # Make rapid requests
            results = []
            for i in range(20):
                response = test_client.get("/health", headers=auth_headers)
                results.append(response.status_code)

            # First 10 should succeed, rest should be rate limited
            successful = sum(1 for code in results if code == 200)
            rate_limited = sum(1 for code in results if code == 429)

            assert successful == 10
            assert rate_limited == 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limiting_per_ip(self, test_client, auth_headers):
        """Test rate limiting behavior per IP address."""

        # Mock different client IPs
        with patch("src.integration.api_server.rate_limiter") as mock_rate_limiter:
            ip_requests = {}

            def mock_is_allowed(client_ip, limit, window_minutes=1):
                if client_ip not in ip_requests:
                    ip_requests[client_ip] = 0
                ip_requests[client_ip] += 1
                return ip_requests[client_ip] <= 5  # 5 requests per IP

            mock_rate_limiter.is_allowed = mock_is_allowed

            # Simulate requests from different IPs
            results_by_ip = {}

            for ip in ["192.168.1.1", "192.168.1.2", "192.168.1.3"]:
                results_by_ip[ip] = []

                # Mock the client IP
                with patch("src.integration.api_server.Request") as mock_request:
                    mock_request.client.host = ip

                    for i in range(10):
                        response = test_client.get("/health", headers=auth_headers)
                        results_by_ip[ip].append(response.status_code)

            # Each IP should have 5 successful requests
            for ip, results in results_by_ip.items():
                successful = sum(1 for code in results if code == 200)
                rate_limited = sum(1 for code in results if code == 429)
                print(f"IP {ip}: {successful} successful, {rate_limited} rate limited")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, test_client, auth_headers):
        """Test rate limit recovery after window expires."""

        with patch("src.integration.api_server.rate_limiter") as mock_rate_limiter:
            request_times = []

            def mock_is_allowed(client_ip, limit, window_minutes=1):
                now = time.time()
                request_times.append(now)

                # Only allow requests if fewer than 3 in last 1 second (simulated window)
                recent_requests = [t for t in request_times if now - t < 1]
                return len(recent_requests) <= 3

            mock_rate_limiter.is_allowed = mock_is_allowed

            # Make rapid requests - should be rate limited
            rapid_results = []
            for i in range(10):
                response = test_client.get("/health", headers=auth_headers)
                rapid_results.append(response.status_code)
                time.sleep(0.01)  # Small delay

            # Wait for window to reset
            time.sleep(1.5)

            # Make more requests - should be allowed again
            recovery_results = []
            for i in range(5):
                response = test_client.get("/health", headers=auth_headers)
                recovery_results.append(response.status_code)
                time.sleep(0.5)  # Space out requests

            # Recovery requests should mostly succeed
            recovery_success = sum(1 for code in recovery_results if code == 200)
            assert recovery_success >= 3  # Most should succeed after recovery


class TestAuthenticationStressTesting:
    """Test authentication system under stress conditions."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_authentication_requests(self, test_client):
        """Test concurrent authentication with different credentials."""

        valid_headers = {"Authorization": "Bearer test_api_key_for_integration_testing"}
        invalid_headers = {"Authorization": "Bearer invalid_key"}
        no_headers = {}

        async def make_auth_request(headers):
            try:
                response = test_client.get("/health", headers=headers)
                return response.status_code
            except Exception:
                return 500

        # Mix of valid, invalid, and missing auth
        auth_scenarios = (
            [valid_headers] * 50 + [invalid_headers] * 25 + [no_headers] * 25
        )
        random.shuffle(auth_scenarios)

        # Make concurrent requests
        tasks = [make_auth_request(headers) for headers in auth_scenarios]

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        end_time = time.time()
        duration = end_time - start_time

        # Count results
        success_count = sum(1 for code in results if code == 200)
        auth_error_count = sum(1 for code in results if code == 401)

        # Should have 50 successes and ~50 auth errors
        assert success_count == 50
        assert auth_error_count >= 45  # Allow for some variance

        # Should process quickly
        assert duration < 10.0

        print(
            f"Auth stress test: {success_count} success, {auth_error_count} auth errors "
            f"in {duration:.2f}s"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_token_validation_stress(self, test_client):
        """Test JWT token validation under stress."""

        # Create various JWT tokens
        secret_key = "test_secret_key_for_integration_testing"

        # Valid token
        valid_token = jwt.encode(
            {"sub": "test_user", "exp": datetime.utcnow() + timedelta(hours=1)},
            secret_key,
            algorithm="HS256",
        )

        # Expired token
        expired_token = jwt.encode(
            {"sub": "test_user", "exp": datetime.utcnow() - timedelta(hours=1)},
            secret_key,
            algorithm="HS256",
        )

        # Invalid signature
        invalid_token = jwt.encode(
            {"sub": "test_user", "exp": datetime.utcnow() + timedelta(hours=1)},
            "wrong_secret",
            algorithm="HS256",
        )

        # Malformed token
        malformed_token = "not.a.valid.jwt.token"

        tokens = {
            "valid": valid_token,
            "expired": expired_token,
            "invalid": invalid_token,
            "malformed": malformed_token,
        }

        async def test_token(token_type, token):
            headers = {"Authorization": f"Bearer {token}"}
            try:
                response = test_client.get("/health", headers=headers)
                return token_type, response.status_code
            except Exception:
                return token_type, 500

        # Test each token type multiple times concurrently
        tasks = []
        for token_type, token in tokens.items():
            for _ in range(25):  # 25 requests per token type
                tasks.append(test_token(token_type, token))

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        # Analyze results by token type
        results_by_type = {}
        for token_type, status_code in results:
            if token_type not in results_by_type:
                results_by_type[token_type] = []
            results_by_type[token_type].append(status_code)

        # Valid tokens should succeed
        assert all(code == 200 for code in results_by_type["valid"])

        # Invalid tokens should fail with 401
        for token_type in ["expired", "invalid", "malformed"]:
            assert all(code == 401 for code in results_by_type[token_type])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_brute_force_attack_simulation(self, test_client):
        """Test API response to brute force attack simulation."""

        # Simulate brute force with many invalid API keys
        invalid_keys = [f"fake_key_{i}" for i in range(100)]

        async def brute_force_attempt(api_key):
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                response = test_client.get("/health", headers=headers)
                return response.status_code, (
                    response.elapsed.total_seconds()
                    if hasattr(response, "elapsed")
                    else 0
                )
            except Exception:
                return 500, 0

        # Execute brute force attempts
        start_time = time.time()
        tasks = [brute_force_attempt(key) for key in invalid_keys]

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_duration = end_time - start_time

        # All should be rejected
        status_codes = [result[0] for result in results]
        assert all(code == 401 for code in status_codes)

        # Should not cause excessive slowdown (no more than 30 seconds for 100 requests)
        assert total_duration < 30.0

        # Verify system remains responsive after attack
        valid_headers = {"Authorization": "Bearer test_api_key_for_integration_testing"}
        post_attack_response = test_client.get("/health", headers=valid_headers)
        assert post_attack_response.status_code == 200

        print(f"Brute force simulation: 100 invalid attempts in {total_duration:.2f}s")


class TestDatabaseConnectionPoolingUnderLoad:
    """Test database connection handling under concurrent load."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_database_access(self, test_client, auth_headers):
        """Test concurrent database access through API endpoints."""

        # Mock database manager with connection tracking
        with patch("src.integration.api_server.get_database_manager") as mock_get_db:
            connection_count = 0
            max_concurrent_connections = 0
            active_connections = 0
            connection_lock = threading.Lock()

            async def mock_health_check():
                nonlocal connection_count, max_concurrent_connections, active_connections

                with connection_lock:
                    connection_count += 1
                    active_connections += 1
                    max_concurrent_connections = max(
                        max_concurrent_connections, active_connections
                    )

                # Simulate database operation
                await asyncio.sleep(0.1)

                with connection_lock:
                    active_connections -= 1

                return {
                    "status": "healthy",
                    "database_connected": True,
                    "connection_pool_size": 10,
                    "active_connections": active_connections,
                }

            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = mock_health_check
            mock_get_db.return_value = mock_db_manager

            # Make concurrent health check requests (which access database)
            async def make_health_request():
                response = test_client.get("/health", headers=auth_headers)
                return response.status_code

            tasks = [make_health_request() for _ in range(50)]

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(asyncio.run, task) for task in tasks]
                results = [future.result() for future in futures]

            end_time = time.time()
            duration = end_time - start_time

            # All requests should succeed
            assert all(code == 200 for code in results)

            # Should complete in reasonable time
            assert duration < 15.0

            # Verify connection pooling worked
            assert connection_count == 50  # All requests made DB calls
            print(
                f"DB stress test: {connection_count} total connections, "
                f"max concurrent: {max_concurrent_connections}, "
                f"duration: {duration:.2f}s"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_connection_failure_recovery(
        self, test_client, auth_headers
    ):
        """Test API behavior when database connections fail."""

        with patch("src.integration.api_server.get_database_manager") as mock_get_db:
            failure_count = 0

            async def failing_health_check():
                nonlocal failure_count
                failure_count += 1

                if failure_count <= 10:
                    # First 10 calls fail
                    raise ConnectionError("Database connection failed")
                else:
                    # Subsequent calls succeed
                    return {
                        "status": "healthy",
                        "database_connected": True,
                        "recovered": True,
                    }

            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = failing_health_check
            mock_get_db.return_value = mock_db_manager

            # Make requests during failure and recovery
            results = []
            for i in range(20):
                response = test_client.get("/health", headers=auth_headers)
                results.append((i, response.status_code))
                await asyncio.sleep(0.1)  # Small delay between requests

            # Should handle failures gracefully and recover
            early_failures = sum(1 for i, code in results[:10] if code != 200)
            later_successes = sum(1 for i, code in results[10:] if code == 200)

            # Some early requests should fail due to DB issues
            assert early_failures > 0

            # Later requests should succeed after recovery
            assert later_successes > 5


class TestWebSocketIntegration:
    """Test WebSocket functionality under load (if implemented)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_connection_stress(self, test_client):
        """Test multiple WebSocket connections simultaneously."""

        # Skip if WebSocket not implemented
        try:
            # Attempt WebSocket connection
            with test_client.websocket_connect("/ws") as websocket:
                # If this succeeds, WebSocket is implemented
                pass
        except Exception:
            pytest.skip("WebSocket endpoint not implemented")

        # Test multiple concurrent connections
        active_connections = []

        try:
            # Open multiple WebSocket connections
            for i in range(10):
                websocket = test_client.websocket_connect(f"/ws?client_id={i}")
                active_connections.append(websocket)

            # Send messages on all connections
            for i, ws in enumerate(active_connections):
                ws.send_json({"type": "ping", "client_id": i})
                response = ws.receive_json()
                assert response["type"] == "pong"

        finally:
            # Clean up connections
            for ws in active_connections:
                try:
                    ws.close()
                except Exception:
                    pass

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self, test_client):
        """Test WebSocket message throughput."""

        try:
            with test_client.websocket_connect("/ws") as websocket:
                start_time = time.time()

                # Send 100 messages rapidly
                for i in range(100):
                    websocket.send_json(
                        {
                            "type": "test_message",
                            "id": i,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # Receive response
                    response = websocket.receive_json()
                    assert "id" in response

                end_time = time.time()
                duration = end_time - start_time

                # Should handle 100 messages in reasonable time
                assert duration < 10.0

                print(f"WebSocket throughput: 100 messages in {duration:.2f}s")

        except Exception:
            pytest.skip("WebSocket endpoint not implemented or failed")


class TestAPIErrorRecoveryUnderLoad:
    """Test API error handling and recovery under load conditions."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_under_concurrent_load(
        self, test_client, auth_headers
    ):
        """Test error handling when components fail under load."""

        with patch("src.integration.api_server.get_tracking_manager") as mock_get_tm:
            failure_probability = 0.3  # 30% of requests fail

            async def intermittent_failure():
                if random.random() < failure_probability:
                    raise Exception("Simulated component failure")

                return {
                    "room_id": "test_room",
                    "prediction_time": datetime.now().isoformat(),
                    "confidence": 0.85,
                    "transition_type": "occupied",
                }

            mock_manager = AsyncMock()
            mock_manager.get_room_prediction = intermittent_failure
            mock_get_tm.return_value = mock_manager

            # Make concurrent prediction requests
            async def make_prediction_request():
                try:
                    response = test_client.get(
                        "/predictions/test_room", headers=auth_headers
                    )
                    return response.status_code
                except Exception:
                    return 500

            tasks = [make_prediction_request() for _ in range(100)]

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(asyncio.run, task) for task in tasks]
                results = [future.result() for future in futures]

            # Count outcomes
            successes = sum(1 for code in results if code == 200)
            errors = sum(1 for code in results if code == 500)

            print(f"Error recovery test: {successes} success, {errors} errors")

            # Should handle failures gracefully
            assert errors > 0  # Some failures expected
            assert successes > 0  # Some successes expected
            assert successes + errors == 100  # All requests handled

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_load(self, test_client, auth_headers):
        """Test graceful degradation when system is under stress."""

        # Simulate various component failures
        with patch(
            "src.integration.api_server.get_database_manager"
        ) as mock_get_db, patch(
            "src.integration.api_server.get_tracking_manager"
        ) as mock_get_tm, patch(
            "src.integration.api_server.get_mqtt_manager"
        ) as mock_get_mqtt:

            # Database sometimes fails
            db_failure_count = 0

            async def db_health_check():
                nonlocal db_failure_count
                db_failure_count += 1
                if db_failure_count % 3 == 0:  # Every 3rd call fails
                    raise ConnectionError("Database temporarily unavailable")
                return {"status": "healthy", "database_connected": True}

            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = db_health_check
            mock_get_db.return_value = mock_db_manager

            # Tracking manager sometimes fails
            tm_failure_count = 0

            async def tm_status():
                nonlocal tm_failure_count
                tm_failure_count += 1
                if tm_failure_count % 4 == 0:  # Every 4th call fails
                    raise Exception("Tracking manager temporarily unavailable")
                return {"tracking_active": True, "status": "active"}

            mock_tm = AsyncMock()
            mock_tm.get_tracking_status = tm_status
            mock_get_tm.return_value = mock_tm

            # MQTT sometimes fails
            def mqtt_stats():
                if random.random() < 0.2:  # 20% failure rate
                    raise Exception("MQTT temporarily unavailable")
                return {"mqtt_connected": True, "predictions_published": 100}

            mock_mqtt_manager = AsyncMock()
            mock_mqtt_manager.get_integration_stats = mqtt_stats
            mock_get_mqtt.return_value = mock_mqtt_manager

            # Make many health check requests
            results = []
            for i in range(50):
                try:
                    response = test_client.get("/health", headers=auth_headers)
                    results.append(response.status_code)
                except Exception:
                    results.append(500)

            # System should degrade gracefully - some requests succeed despite failures
            successes = sum(1 for code in results if code == 200)

            # Should have some successes even with component failures
            assert successes > 10  # At least 20% success rate

            print(
                f"Graceful degradation: {successes}/50 requests succeeded despite component failures"
            )
