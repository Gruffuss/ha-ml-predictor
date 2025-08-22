"""
Comprehensive unit tests for authentication dependencies.

This test suite validates FastAPI dependency functions for JWT authentication,
permission checking, user context injection, and API key validation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
import pytest

from src.core.exceptions import APIAuthenticationError
from src.integration.auth.auth_models import AuthUser
from src.integration.auth.dependencies import (
    get_current_user,
    get_jwt_manager,
    get_optional_user,
    get_request_context,
    require_admin,
    require_permission,
    require_permissions,
    require_role,
    validate_api_key,
)


class TestJWTManagerDependency:
    """Test JWT manager dependency function."""

    @patch("src.integration.auth.dependencies.get_config")
    @patch("src.integration.auth.dependencies._jwt_manager", None)
    def test_get_jwt_manager_initialization(self, mock_get_config):
        """Test JWT manager initialization when not exists."""
        mock_config = Mock()
        mock_config.api.jwt.enabled = True
        mock_get_config.return_value = mock_config

        with patch("src.integration.auth.dependencies.JWTManager") as mock_jwt_class:
            mock_jwt_instance = Mock()
            mock_jwt_class.return_value = mock_jwt_instance

            result = get_jwt_manager()

            assert result == mock_jwt_instance
            mock_jwt_class.assert_called_once_with(mock_config.api.jwt)

    @patch("src.integration.auth.dependencies.get_config")
    def test_get_jwt_manager_disabled(self, mock_get_config):
        """Test JWT manager when JWT is disabled."""
        mock_config = Mock()
        mock_config.api.jwt.enabled = False
        mock_get_config.return_value = mock_config

        with patch("src.integration.auth.dependencies._jwt_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                get_jwt_manager()

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "JWT authentication is not enabled" in str(exc_info.value.detail)

    @patch("src.integration.auth.dependencies._jwt_manager")
    def test_get_jwt_manager_cached(self, mock_cached_manager):
        """Test JWT manager returns cached instance."""
        result = get_jwt_manager()
        assert result == mock_cached_manager


class TestGetCurrentUser:
    """Test get_current_user dependency function."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock(spec=Request)
        request.state = Mock()
        return request

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager."""
        manager = Mock()
        manager.validate_token.return_value = {
            "sub": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "permissions": ["read", "write"],
            "roles": ["user"],
            "is_admin": False,
        }
        return manager

    @pytest.mark.asyncio
    async def test_get_current_user_from_request_state(self, mock_request):
        """Test get_current_user when user is already in request state."""
        # Setup user in request state
        test_user = AuthUser(user_id="test", username="testuser")
        mock_request.state.user = test_user

        result = await get_current_user(mock_request, None)

        assert result == test_user

    @pytest.mark.asyncio
    async def test_get_current_user_missing_credentials(self, mock_request):
        """Test get_current_user with missing credentials."""
        mock_request.state.user = None

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authentication credentials" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_jwt_manager")
    async def test_get_current_user_valid_token(
        self, mock_get_jwt, mock_request, mock_jwt_manager
    ):
        """Test get_current_user with valid JWT token."""
        mock_request.state.user = None
        mock_get_jwt.return_value = mock_jwt_manager

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="valid_token_123"
        )

        result = await get_current_user(mock_request, credentials)

        assert isinstance(result, AuthUser)
        assert result.user_id == "test_user_123"
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        assert result.permissions == ["read", "write"]
        assert result.roles == ["user"]
        assert result.is_admin is False
        assert result.is_active is True

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_jwt_manager")
    async def test_get_current_user_invalid_token(
        self, mock_get_jwt, mock_request, mock_jwt_manager
    ):
        """Test get_current_user with invalid JWT token."""
        mock_request.state.user = None
        mock_get_jwt.return_value = mock_jwt_manager
        mock_jwt_manager.validate_token.side_effect = APIAuthenticationError(
            "Invalid token"
        )

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_token"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_jwt_manager")
    async def test_get_current_user_service_error(self, mock_get_jwt, mock_request):
        """Test get_current_user with authentication service error."""
        mock_request.state.user = None
        mock_get_jwt.side_effect = Exception("Service unavailable")

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, credentials)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Authentication service error" in str(exc_info.value.detail)


class TestGetOptionalUser:
    """Test get_optional_user dependency function."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock(spec=Request)
        request.state = Mock()
        return request

    @pytest.mark.asyncio
    async def test_get_optional_user_success(self, mock_request):
        """Test get_optional_user with successful authentication."""
        test_user = AuthUser(user_id="test", username="testuser")
        mock_request.state.user = test_user

        result = await get_optional_user(mock_request, None)

        assert result == test_user

    @pytest.mark.asyncio
    async def test_get_optional_user_auth_failure(self, mock_request):
        """Test get_optional_user with authentication failure."""
        mock_request.state.user = None

        # This should return None instead of raising exception
        result = await get_optional_user(mock_request, None)

        assert result is None


class TestPermissionDependencies:
    """Test permission-based dependency functions."""

    @pytest.fixture
    def user_with_permissions(self):
        """User with specific permissions."""
        return AuthUser(
            user_id="test_user",
            username="testuser",
            permissions=["read", "write", "prediction_view"],
            roles=["user"],
            is_admin=False,
        )

    @pytest.fixture
    def admin_user(self):
        """Admin user with admin privileges."""
        return AuthUser(
            user_id="admin_user",
            username="admin",
            permissions=["read"],
            roles=["admin"],
            is_admin=True,
        )

    @pytest.mark.asyncio
    async def test_require_permission_success(self, user_with_permissions):
        """Test require_permission with user having required permission."""
        permission_checker = require_permission("read")

        result = await permission_checker(user_with_permissions)

        assert result == user_with_permissions

    @pytest.mark.asyncio
    async def test_require_permission_failure(self, user_with_permissions):
        """Test require_permission with user missing required permission."""
        permission_checker = require_permission("admin")

        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(user_with_permissions)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions: 'admin' required" in str(
            exc_info.value.detail
        )

    @pytest.mark.asyncio
    async def test_require_permission_admin_override(self, admin_user):
        """Test require_permission with admin user (should have all permissions)."""
        permission_checker = require_permission("system_config")

        result = await permission_checker(admin_user)

        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_role_success(self, user_with_permissions):
        """Test require_role with user having required role."""
        role_checker = require_role("user")

        result = await role_checker(user_with_permissions)

        assert result == user_with_permissions

    @pytest.mark.asyncio
    async def test_require_role_failure(self, user_with_permissions):
        """Test require_role with user missing required role."""
        role_checker = require_role("admin")

        with pytest.raises(HTTPException) as exc_info:
            await role_checker(user_with_permissions)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient role: 'admin' required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_admin_success(self, admin_user):
        """Test require_admin with admin user."""
        admin_checker = require_admin()

        result = await admin_checker(admin_user)

        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_admin_failure(self, user_with_permissions):
        """Test require_admin with non-admin user."""
        admin_checker = require_admin()

        with pytest.raises(HTTPException) as exc_info:
            await admin_checker(user_with_permissions)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Administrator privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_permissions_all_success(self, user_with_permissions):
        """Test require_permissions with user having all required permissions."""
        permissions_checker = require_permissions(["read", "write"], require_all=True)

        result = await permissions_checker(user_with_permissions)

        assert result == user_with_permissions

    @pytest.mark.asyncio
    async def test_require_permissions_all_failure(self, user_with_permissions):
        """Test require_permissions with user missing some required permissions."""
        permissions_checker = require_permissions(["read", "admin"], require_all=True)

        with pytest.raises(HTTPException) as exc_info:
            await permissions_checker(user_with_permissions)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Missing required permissions" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_permissions_any_success(self, user_with_permissions):
        """Test require_permissions with user having at least one required permission."""
        permissions_checker = require_permissions(["read", "admin"], require_all=False)

        result = await permissions_checker(user_with_permissions)

        assert result == user_with_permissions

    @pytest.mark.asyncio
    async def test_require_permissions_any_failure(self, user_with_permissions):
        """Test require_permissions with user having none of the required permissions."""
        permissions_checker = require_permissions(
            ["admin", "system_config"], require_all=False
        )

        with pytest.raises(HTTPException) as exc_info:
            await permissions_checker(user_with_permissions)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "One of these permissions required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_permissions_admin_override_all(self, admin_user):
        """Test require_permissions with admin user (require_all=True)."""
        permissions_checker = require_permissions(
            ["system_config", "model_retrain"], require_all=True
        )

        result = await permissions_checker(admin_user)

        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_permissions_admin_override_any(self, admin_user):
        """Test require_permissions with admin user (require_all=False)."""
        permissions_checker = require_permissions(
            ["system_config", "model_retrain"], require_all=False
        )

        result = await permissions_checker(admin_user)

        assert result == admin_user


class TestAPIKeyValidation:
    """Test API key validation functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.api.api_key_enabled = True
        config.api.api_key = "valid_api_key_123"
        config.api.debug = False
        return config

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": "valid_api_key_123"}
        request.query_params = {}
        return request

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_disabled(self, mock_get_config):
        """Test API key validation when disabled."""
        mock_config = Mock()
        mock_config.api.api_key_enabled = False
        mock_get_config.return_value = mock_config

        result = await validate_api_key()

        assert result is True

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_provided_valid(self, mock_get_config, mock_config):
        """Test API key validation with valid key provided directly."""
        mock_get_config.return_value = mock_config

        result = await validate_api_key(api_key="valid_api_key_123")

        assert result is True

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_provided_invalid(
        self, mock_get_config, mock_config
    ):
        """Test API key validation with invalid key provided directly."""
        mock_get_config.return_value = mock_config

        result = await validate_api_key(api_key="invalid_key")

        assert result is False

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_from_header(
        self, mock_get_config, mock_config, mock_request
    ):
        """Test API key validation from request header."""
        mock_get_config.return_value = mock_config

        result = await validate_api_key(request=mock_request)

        assert result is True

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_from_query_debug(
        self, mock_get_config, mock_config
    ):
        """Test API key validation from query parameter in debug mode."""
        mock_config.api.debug = True
        mock_get_config.return_value = mock_config

        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {"api_key": "valid_api_key_123"}

        result = await validate_api_key(request=mock_request)

        assert result is True

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_from_query_production(
        self, mock_get_config, mock_config
    ):
        """Test API key validation from query parameter in production (should be ignored)."""
        mock_config.api.debug = False
        mock_get_config.return_value = mock_config

        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {"api_key": "valid_api_key_123"}

        result = await validate_api_key(request=mock_request)

        assert result is False

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_missing(self, mock_get_config, mock_config):
        """Test API key validation with missing key."""
        mock_get_config.return_value = mock_config

        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}

        result = await validate_api_key(request=mock_request)

        assert result is False

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_validate_api_key_header_invalid(self, mock_get_config, mock_config):
        """Test API key validation with invalid key in header."""
        mock_get_config.return_value = mock_config

        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-API-Key": "invalid_key"}
        mock_request.query_params = {}

        result = await validate_api_key(request=mock_request)

        assert result is False


class TestGetRequestContext:
    """Test get_request_context utility function."""

    def test_get_request_context_minimal(self):
        """Test get_request_context with minimal request data."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"User-Agent": "TestClient/1.0"}
        mock_request.state = Mock()
        mock_request.state.__dict__ = {}

        # Mock missing attributes
        mock_request.state.request_id = None
        mock_request.state.user = None

        context = get_request_context(mock_request)

        assert context["method"] == "GET"
        assert context["path"] == "/api/test"
        assert context["client_ip"] == "127.0.0.1"
        assert context["user_agent"] == "TestClient/1.0"
        assert context["request_id"] is None
        assert context["timestamp"] is None

    def test_get_request_context_with_user(self):
        """Test get_request_context with authenticated user."""
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/predictions"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {"User-Agent": "APIClient/2.0"}

        # Setup request state
        mock_request.state = Mock()
        mock_request.state.request_id = "req_12345"
        mock_request.state.__dict__ = {"start_time": "2024-01-01T12:00:00Z"}

        # Setup user
        test_user = AuthUser(
            user_id="test_user",
            username="testuser",
            permissions=["read", "write"],
            is_admin=False,
        )
        mock_request.state.user = test_user

        context = get_request_context(mock_request)

        assert context["method"] == "POST"
        assert context["path"] == "/api/predictions"
        assert context["client_ip"] == "192.168.1.1"
        assert context["user_agent"] == "APIClient/2.0"
        assert context["request_id"] == "req_12345"
        assert context["timestamp"] == "2024-01-01T12:00:00Z"
        assert context["user_id"] == "test_user"
        assert context["user_permissions"] == ["read", "write"]
        assert context["is_admin"] is False

    def test_get_request_context_no_client(self):
        """Test get_request_context when client info is not available."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/health"
        mock_request.client = None
        mock_request.headers = {}
        mock_request.state = Mock()
        mock_request.state.__dict__ = {}
        mock_request.state.request_id = None
        mock_request.state.user = None

        context = get_request_context(mock_request)

        assert context["method"] == "GET"
        assert context["path"] == "/api/health"
        assert context["client_ip"] == "unknown"
        assert context["user_agent"] is None


class TestDependencyIntegration:
    """Test integration between different dependency functions."""

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager for integration tests."""
        manager = Mock()
        manager.validate_token.return_value = {
            "sub": "integration_user",
            "username": "integrationuser",
            "permissions": ["read", "write", "admin"],
            "roles": ["admin"],
            "is_admin": True,
        }
        return manager

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_jwt_manager")
    async def test_permission_chain_with_jwt(self, mock_get_jwt, mock_jwt_manager):
        """Test permission checking with JWT authentication chain."""
        mock_get_jwt.return_value = mock_jwt_manager

        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()
        mock_request.state.user = None

        # Mock credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="valid_token"
        )

        # Get user through JWT
        user = await get_current_user(mock_request, credentials)

        # Test various permission checks with the same user
        admin_checker = require_admin()
        admin_result = await admin_checker(user)
        assert admin_result == user

        permission_checker = require_permission("system_config")
        permission_result = await permission_checker(user)
        assert permission_result == user

        role_checker = require_role("admin")
        role_result = await role_checker(user)
        assert role_result == user

    @pytest.mark.asyncio
    async def test_optional_user_integration(self):
        """Test optional user dependency integration."""
        # Test with no credentials
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()
        mock_request.state.user = None

        optional_user = await get_optional_user(mock_request, None)
        assert optional_user is None

        # Test with user in state
        test_user = AuthUser(user_id="test", username="test")
        mock_request.state.user = test_user

        optional_user = await get_optional_user(mock_request, None)
        assert optional_user == test_user

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_config")
    async def test_api_key_and_jwt_interaction(self, mock_get_config):
        """Test interaction between API key validation and JWT authentication."""
        mock_config = Mock()
        mock_config.api.api_key_enabled = True
        mock_config.api.api_key = "test_key"
        mock_get_config.return_value = mock_config

        # Test API key validation
        api_key_valid = await validate_api_key(api_key="test_key")
        assert api_key_valid is True

        # Test that different auth methods can coexist
        api_key_invalid = await validate_api_key(api_key="wrong_key")
        assert api_key_invalid is False


class TestErrorHandling:
    """Test error handling in dependencies."""

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_manager_exception(self):
        """Test get_current_user when JWT manager raises exception."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        with patch("src.integration.auth.dependencies.get_jwt_manager") as mock_get_jwt:
            mock_get_jwt.side_effect = Exception("JWT service error")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_request, credentials)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    @patch("src.integration.auth.dependencies.get_jwt_manager")
    async def test_get_current_user_token_validation_exception(self, mock_get_jwt):
        """Test get_current_user when token validation raises unexpected exception."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        mock_jwt_manager = Mock()
        mock_jwt_manager.validate_token.side_effect = Exception("Unexpected error")
        mock_get_jwt.return_value = mock_jwt_manager

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, credentials)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Authentication service error" in str(exc_info.value.detail)

    def test_permission_dependencies_with_invalid_user(self):
        """Test permission dependencies with None user (should not happen in practice)."""
        permission_checker = require_permission("read")

        # This would typically be caught by FastAPI's dependency injection
        # but we test the error condition anyway
        with pytest.raises(AttributeError):
            import asyncio

            asyncio.run(permission_checker(None))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_get_current_user_empty_token_payload(self):
        """Test get_current_user with empty token payload."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        with patch("src.integration.auth.dependencies.get_jwt_manager") as mock_get_jwt:
            mock_jwt_manager = Mock()
            mock_jwt_manager.validate_token.return_value = {}  # Empty payload
            mock_get_jwt.return_value = mock_jwt_manager

            result = await get_current_user(mock_request, credentials)

            # Should create user with minimal data
            assert isinstance(result, AuthUser)
            assert result.user_id == ""  # Empty sub
            assert result.username == ""  # Empty sub as fallback
            assert result.permissions == []
            assert result.roles == []

    @pytest.mark.asyncio
    async def test_require_permissions_empty_list(self):
        """Test require_permissions with empty permission list."""
        user = AuthUser(user_id="test", username="test", permissions=["read"])

        # Empty required permissions should always pass
        permissions_checker = require_permissions([], require_all=True)
        result = await permissions_checker(user)
        assert result == user

        permissions_checker = require_permissions([], require_all=False)
        result = await permissions_checker(user)
        assert result == user

    def test_get_request_context_missing_attributes(self):
        """Test get_request_context with missing request attributes."""
        mock_request = Mock(spec=Request)

        # Remove some attributes to test robustness
        del mock_request.client
        mock_request.headers = {}
        mock_request.state = Mock()
        mock_request.state.__dict__ = {}

        # Should handle missing attributes gracefully
        context = get_request_context(mock_request)
        assert context["client_ip"] == "unknown"
        assert context["user_agent"] is None
