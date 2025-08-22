"""
Comprehensive unit tests for authentication dependencies module.

This test suite validates FastAPI dependencies for JWT authentication,
permission checking, user context injection, and security controls.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
import pytest

from src.core.exceptions import APIAuthenticationError
from src.integration.auth.auth_models import AuthUser
from src.integration.auth.dependencies import (
    _jwt_manager,
    get_current_user,
    get_jwt_manager,
    get_optional_user,
    get_request_context,
    require_admin,
    require_permission,
    require_permissions,
    require_role,
    security_scheme,
    validate_api_key,
)
from src.integration.auth.jwt_manager import JWTManager


class TestGetJWTManager:
    """Test JWT manager dependency functionality."""

    def teardown_method(self):
        """Reset global JWT manager instance after each test."""
        global _jwt_manager
        _jwt_manager = None

    @patch("src.integration.auth.dependencies.get_config")
    @patch("src.integration.auth.dependencies.JWTManager")
    def test_get_jwt_manager_creates_instance(self, mock_jwt_class, mock_get_config):
        """Test JWT manager instance creation."""
        # Mock configuration
        mock_config = Mock()
        mock_config.api.jwt.enabled = True
        mock_config.api.jwt = Mock()  # JWT config object
        mock_get_config.return_value = mock_config

        # Mock JWTManager class
        mock_jwt_instance = Mock()
        mock_jwt_class.return_value = mock_jwt_instance

        # Get JWT manager
        result = get_jwt_manager()

        # Verify
        assert result == mock_jwt_instance
        mock_jwt_class.assert_called_once_with(mock_config.api.jwt)

    @patch("src.integration.auth.dependencies.get_config")
    def test_get_jwt_manager_jwt_disabled(self, mock_get_config):
        """Test JWT manager when JWT is disabled."""
        mock_config = Mock()
        mock_config.api.jwt.enabled = False
        mock_get_config.return_value = mock_config

        with pytest.raises(HTTPException) as excinfo:
            get_jwt_manager()

        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "JWT authentication is not enabled" in excinfo.value.detail

    def test_get_jwt_manager_singleton_behavior(self):
        """Test that JWT manager is a singleton."""
        with patch(
            "src.integration.auth.dependencies.get_config"
        ) as mock_get_config, patch(
            "src.integration.auth.dependencies.JWTManager"
        ) as mock_jwt_class:

            mock_config = Mock()
            mock_config.api.jwt.enabled = True
            mock_config.api.jwt = Mock()
            mock_get_config.return_value = mock_config

            mock_jwt_instance = Mock()
            mock_jwt_class.return_value = mock_jwt_instance

            # Call twice
            result1 = get_jwt_manager()
            result2 = get_jwt_manager()

            # Should return same instance
            assert result1 == result2
            mock_jwt_class.assert_called_once()  # Only called once


class TestGetCurrentUser:
    """Test get_current_user dependency functionality."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock(spec=Request)
        request.state = Mock()
        return request

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager."""
        manager = Mock(spec=JWTManager)
        manager.validate_token.return_value = {
            "sub": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "permissions": ["read", "write"],
            "roles": ["user"],
            "is_admin": False,
        }
        return manager

    @pytest.mark.asyncio
    async def test_get_current_user_from_request_state(self, mock_request):
        """Test getting user from request state (already authenticated by middleware)."""
        # Set user in request state
        expected_user = AuthUser(user_id="test", username="testuser")
        mock_request.state.user = expected_user

        result = await get_current_user(mock_request, None)
        assert result == expected_user

    @pytest.mark.asyncio
    async def test_get_current_user_missing_credentials(self, mock_request):
        """Test get_current_user with missing credentials."""
        mock_request.state.user = None

        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(mock_request, None)

        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authentication credentials" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_token(
        self, mock_request, mock_jwt_manager
    ):
        """Test get_current_user with valid JWT token."""
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="valid_token"
        )

        with patch(
            "src.integration.auth.dependencies.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            result = await get_current_user(mock_request, credentials)

            assert isinstance(result, AuthUser)
            assert result.user_id == "user_123"
            assert result.username == "testuser"
            assert result.email == "test@example.com"
            assert result.permissions == ["read", "write"]
            assert result.roles == ["user"]
            assert result.is_admin is False

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_request, mock_jwt_manager):
        """Test get_current_user with invalid JWT token."""
        mock_request.state.user = None
        mock_jwt_manager.validate_token.side_effect = APIAuthenticationError(
            "Invalid token"
        )

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_token"
        )

        with patch(
            "src.integration.auth.dependencies.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await get_current_user(mock_request, credentials)

            assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid token" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_manager_error(self, mock_request):
        """Test get_current_user with JWT manager error."""
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        with patch(
            "src.integration.auth.dependencies.get_jwt_manager",
            side_effect=Exception("JWT error"),
        ):
            with pytest.raises(HTTPException) as excinfo:
                await get_current_user(mock_request, credentials)

            assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Authentication service error" in excinfo.value.detail


class TestGetOptionalUser:
    """Test get_optional_user dependency functionality."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock(spec=Request)
        request.state = Mock()
        return request

    @pytest.mark.asyncio
    async def test_get_optional_user_success(self, mock_request):
        """Test get_optional_user with valid authentication."""
        expected_user = AuthUser(user_id="test", username="testuser")
        mock_request.state.user = expected_user

        result = await get_optional_user(mock_request, None)
        assert result == expected_user

    @pytest.mark.asyncio
    async def test_get_optional_user_failure_returns_none(self, mock_request):
        """Test get_optional_user returns None on authentication failure."""
        mock_request.state.user = None

        # Should return None instead of raising exception
        result = await get_optional_user(mock_request, None)
        assert result is None


class TestRequirePermission:
    """Test require_permission dependency factory functionality."""

    @pytest.fixture
    def mock_user_with_read(self):
        """Mock user with read permission."""
        return AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read", "write"],
            is_admin=False,
        )

    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user."""
        return AuthUser(
            user_id="admin", username="adminuser", permissions=["read"], is_admin=True
        )

    @pytest.mark.asyncio
    async def test_require_permission_success(self, mock_user_with_read):
        """Test require_permission with user having required permission."""
        permission_checker = require_permission("read")

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_read,
        ):
            result = await permission_checker()
            assert result == mock_user_with_read

    @pytest.mark.asyncio
    async def test_require_permission_admin_bypass(self, mock_admin_user):
        """Test require_permission with admin user bypassing specific permission."""
        permission_checker = require_permission("system_config")

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_admin_user,
        ):
            result = await permission_checker()
            assert result == mock_admin_user

    @pytest.mark.asyncio
    async def test_require_permission_failure(self, mock_user_with_read):
        """Test require_permission with user lacking required permission."""
        permission_checker = require_permission("admin")

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_read,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await permission_checker()

            assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Insufficient permissions: 'admin' required" in excinfo.value.detail


class TestRequireRole:
    """Test require_role dependency factory functionality."""

    @pytest.fixture
    def mock_user_with_role(self):
        """Mock user with specific role."""
        return AuthUser(user_id="test", username="testuser", roles=["user", "operator"])

    @pytest.mark.asyncio
    async def test_require_role_success(self, mock_user_with_role):
        """Test require_role with user having required role."""
        role_checker = require_role("operator")

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_role,
        ):
            result = await role_checker()
            assert result == mock_user_with_role

    @pytest.mark.asyncio
    async def test_require_role_failure(self, mock_user_with_role):
        """Test require_role with user lacking required role."""
        role_checker = require_role("admin")

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_role,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await role_checker()

            assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Insufficient role: 'admin' required" in excinfo.value.detail


class TestRequireAdmin:
    """Test require_admin dependency functionality."""

    @pytest.fixture
    def mock_regular_user(self):
        """Mock regular (non-admin) user."""
        return AuthUser(user_id="test", username="testuser", is_admin=False)

    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user."""
        return AuthUser(user_id="admin", username="adminuser", is_admin=True)

    @pytest.mark.asyncio
    async def test_require_admin_success(self, mock_admin_user):
        """Test require_admin with admin user."""
        admin_checker = require_admin()

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_admin_user,
        ):
            result = await admin_checker()
            assert result == mock_admin_user

    @pytest.mark.asyncio
    async def test_require_admin_failure(self, mock_regular_user):
        """Test require_admin with regular user."""
        admin_checker = require_admin()

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_regular_user,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await admin_checker()

            assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Administrator privileges required" in excinfo.value.detail


class TestRequirePermissions:
    """Test require_permissions dependency factory functionality."""

    @pytest.fixture
    def mock_user_with_permissions(self):
        """Mock user with specific permissions."""
        return AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read", "write", "prediction_view"],
            is_admin=False,
        )

    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user."""
        return AuthUser(
            user_id="admin", username="adminuser", permissions=["read"], is_admin=True
        )

    @pytest.mark.asyncio
    async def test_require_permissions_all_success(self, mock_user_with_permissions):
        """Test require_permissions with user having all required permissions."""
        permissions_checker = require_permissions(["read", "write"], require_all=True)

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_permissions,
        ):
            result = await permissions_checker()
            assert result == mock_user_with_permissions

    @pytest.mark.asyncio
    async def test_require_permissions_all_failure(self, mock_user_with_permissions):
        """Test require_permissions with user missing some required permissions."""
        permissions_checker = require_permissions(["read", "admin"], require_all=True)

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_permissions,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await permissions_checker()

            assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Missing required permissions" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_require_permissions_any_success(self, mock_user_with_permissions):
        """Test require_permissions with user having at least one required permission."""
        permissions_checker = require_permissions(["read", "admin"], require_all=False)

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_permissions,
        ):
            result = await permissions_checker()
            assert result == mock_user_with_permissions

    @pytest.mark.asyncio
    async def test_require_permissions_any_failure(self, mock_user_with_permissions):
        """Test require_permissions with user having none of the required permissions."""
        permissions_checker = require_permissions(
            ["admin", "system_config"], require_all=False
        )

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_user_with_permissions,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await permissions_checker()

            assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
            assert "One of these permissions required" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_require_permissions_admin_bypass(self, mock_admin_user):
        """Test require_permissions with admin user bypassing permission checks."""
        permissions_checker = require_permissions(["system_config"], require_all=True)

        with patch(
            "src.integration.auth.dependencies.get_current_user",
            return_value=mock_admin_user,
        ):
            result = await permissions_checker()
            assert result == mock_admin_user


class TestValidateAPIKey:
    """Test validate_api_key functionality."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": "test-api-key"}
        request.query_params = {"api_key": "query-api-key"}
        return request

    @pytest.mark.asyncio
    async def test_validate_api_key_disabled(self):
        """Test API key validation when disabled."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = False
            mock_get_config.return_value = mock_config

            result = await validate_api_key()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_from_parameter(self):
        """Test API key validation from function parameter."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "correct-api-key"
            mock_get_config.return_value = mock_config

            result = await validate_api_key("correct-api-key", None)
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_from_header(self, mock_request):
        """Test API key validation from header."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "test-api-key"
            mock_get_config.return_value = mock_config

            result = await validate_api_key(None, mock_request)
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_from_query_debug_mode(self, mock_request):
        """Test API key validation from query parameter in debug mode."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "query-api-key"
            mock_config.api.debug = True
            mock_get_config.return_value = mock_config

            # Remove header to force query parameter usage
            mock_request.headers = {}

            result = await validate_api_key(None, mock_request)
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self):
        """Test API key validation with invalid key."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "correct-api-key"
            mock_get_config.return_value = mock_config

            result = await validate_api_key("wrong-api-key", None)
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_api_key_missing(self, mock_request):
        """Test API key validation with missing key."""
        with patch("src.integration.auth.dependencies.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "test-api-key"
            mock_get_config.return_value = mock_config

            # Remove all API key sources
            mock_request.headers = {}
            mock_request.query_params = {}

            result = await validate_api_key(None, mock_request)
            assert result is False


class TestGetRequestContext:
    """Test get_request_context functionality."""

    @pytest.fixture
    def mock_request_with_user(self):
        """Mock request with authenticated user."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "req_123"
        request.state.start_time = datetime.now(timezone.utc)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/predictions"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {"User-Agent": "TestClient/1.0"}

        # Add authenticated user
        user = AuthUser(
            user_id="user_123",
            username="testuser",
            permissions=["read", "write"],
            is_admin=False,
        )
        request.state.user = user

        return request

    @pytest.fixture
    def mock_request_no_user(self):
        """Mock request without authenticated user."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "req_456"
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/auth/login"
        request.client = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {"User-Agent": "Browser/1.0"}
        request.state.__dict__ = {}  # Empty state

        return request

    def test_get_request_context_with_user(self, mock_request_with_user):
        """Test request context generation with authenticated user."""
        context = get_request_context(mock_request_with_user)

        assert context["request_id"] == "req_123"
        assert context["method"] == "GET"
        assert context["path"] == "/api/predictions"
        assert context["client_ip"] == "127.0.0.1"
        assert context["user_agent"] == "TestClient/1.0"
        assert context["user_id"] == "user_123"
        assert context["user_permissions"] == ["read", "write"]
        assert context["is_admin"] is False

    def test_get_request_context_without_user(self, mock_request_no_user):
        """Test request context generation without authenticated user."""
        context = get_request_context(mock_request_no_user)

        assert context["request_id"] == "req_456"
        assert context["method"] == "POST"
        assert context["path"] == "/auth/login"
        assert context["client_ip"] == "192.168.1.100"
        assert context["user_agent"] == "Browser/1.0"
        assert "user_id" not in context
        assert "user_permissions" not in context
        assert "is_admin" not in context

    def test_get_request_context_missing_client(self):
        """Test request context with missing client information."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = None
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/health"
        request.client = None  # No client info
        request.headers = {}
        request.state.__dict__ = {}

        context = get_request_context(request)

        assert context["client_ip"] == "unknown"
        assert context["user_agent"] is None
        assert context["request_id"] is None

    def test_get_request_context_partial_state(self):
        """Test request context with partially populated state."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "partial_req"
        request.method = "PUT"
        request.url = Mock()
        request.url.path = "/api/update"
        request.client = Mock()
        request.client.host = "10.0.0.1"
        request.headers = {"User-Agent": "PartialClient/2.0"}

        # State has user but no start_time
        user = AuthUser(user_id="partial_user", username="partial")
        request.state.user = user
        request.state.__dict__ = {"request_id": "partial_req", "user": user}

        context = get_request_context(request)

        assert context["user_id"] == "partial_user"
        assert context["timestamp"] is None  # No start_time in state


class TestSecurityScheme:
    """Test security scheme configuration."""

    def test_security_scheme_configuration(self):
        """Test that security scheme is properly configured."""
        from fastapi.security import HTTPBearer

        assert isinstance(security_scheme, HTTPBearer)
        assert security_scheme.auto_error is False  # Should not auto-raise errors


class TestDependenciesIntegration:
    """Test integration scenarios between dependencies."""

    @pytest.fixture
    def mock_app_with_dependencies(self):
        """Mock FastAPI app with authentication dependencies."""
        from fastapi import Depends, FastAPI

        app = FastAPI()

        @app.get("/test-read")
        async def test_read_endpoint(
            user: AuthUser = Depends(require_permission("read")),
        ):
            return {"user": user.username, "endpoint": "read"}

        @app.get("/test-admin")
        async def test_admin_endpoint(user: AuthUser = Depends(require_admin())):
            return {"user": user.username, "endpoint": "admin"}

        @app.get("/test-multiple-perms")
        async def test_multiple_perms(
            user: AuthUser = Depends(require_permissions(["read", "write"]))
        ):
            return {"user": user.username, "endpoint": "multi-perms"}

        return app

    @pytest.mark.asyncio
    async def test_dependency_chain_success(self):
        """Test successful dependency chain execution."""
        mock_user = AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read", "write"],
            roles=["user"],
        )

        # Test permission dependency chain
        permission_checker = require_permission("read")

        with patch(
            "src.integration.auth.dependencies.get_current_user", return_value=mock_user
        ):
            result = await permission_checker()
            assert result == mock_user

    @pytest.mark.asyncio
    async def test_dependency_chain_multiple_requirements(self):
        """Test dependency chain with multiple permission requirements."""
        mock_user = AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read", "write", "admin"],
            roles=["admin"],
            is_admin=True,
        )

        # Test multiple dependencies
        permission_checker = require_permissions(["read", "write"], require_all=True)
        role_checker = require_role("admin")
        admin_checker = require_admin()

        with patch(
            "src.integration.auth.dependencies.get_current_user", return_value=mock_user
        ):
            # All should succeed with admin user
            assert await permission_checker() == mock_user
            assert await role_checker() == mock_user
            assert await admin_checker() == mock_user

    @pytest.mark.asyncio
    async def test_dependency_failure_propagation(self):
        """Test that dependency failures propagate correctly."""
        mock_user = AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read"],  # Limited permissions
            roles=["user"],
            is_admin=False,
        )

        # Test cascading failures
        admin_checker = require_admin()
        high_perm_checker = require_permission("system_config")

        with patch(
            "src.integration.auth.dependencies.get_current_user", return_value=mock_user
        ):
            # Both should fail
            with pytest.raises(HTTPException):
                await admin_checker()

            with pytest.raises(HTTPException):
                await high_perm_checker()


class TestDependenciesErrorHandling:
    """Test error handling in dependencies."""

    @pytest.mark.asyncio
    async def test_jwt_manager_initialization_error(self):
        """Test JWT manager initialization error handling."""
        with patch(
            "src.integration.auth.dependencies.get_config",
            side_effect=Exception("Config error"),
        ):
            with pytest.raises(Exception, match="Config error"):
                get_jwt_manager()

    @pytest.mark.asyncio
    async def test_get_current_user_with_malformed_token(self):
        """Test get_current_user with malformed token data."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="malformed_token"
        )

        mock_jwt_manager = Mock()
        mock_jwt_manager.validate_token.return_value = {
            # Missing required 'sub' field
            "username": "testuser",
        }

        with patch(
            "src.integration.auth.dependencies.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await get_current_user(mock_request, credentials)

            assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_permission_checker_with_none_user(self):
        """Test permission checker when get_current_user returns unexpected data."""
        permission_checker = require_permission("read")

        with patch(
            "src.integration.auth.dependencies.get_current_user", return_value=None
        ):
            with pytest.raises(AttributeError):
                await permission_checker()


class TestDependenciesPerformance:
    """Test performance aspects of dependencies."""

    @pytest.mark.asyncio
    async def test_jwt_manager_caching(self):
        """Test that JWT manager instance is cached properly."""
        call_count = 0

        def mock_jwt_constructor(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Mock()

        with patch(
            "src.integration.auth.dependencies.get_config"
        ) as mock_get_config, patch(
            "src.integration.auth.dependencies.JWTManager",
            side_effect=mock_jwt_constructor,
        ):

            mock_config = Mock()
            mock_config.api.jwt.enabled = True
            mock_config.api.jwt = Mock()
            mock_get_config.return_value = mock_config

            # Multiple calls should use cached instance
            get_jwt_manager()
            get_jwt_manager()
            get_jwt_manager()

            assert call_count == 1  # Should only be constructed once

    @pytest.mark.asyncio
    async def test_permission_validation_efficiency(self):
        """Test that permission validation is efficient."""
        user = AuthUser(
            user_id="test",
            username="testuser",
            permissions=["read"] * 100,  # Large permission set
            is_admin=False,
        )

        permission_checker = require_permission("read")

        with patch(
            "src.integration.auth.dependencies.get_current_user", return_value=user
        ):
            # Should complete quickly even with large permission set
            result = await permission_checker()
            assert result == user
