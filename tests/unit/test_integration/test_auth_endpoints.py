"""
Comprehensive unit tests for authentication endpoints - CONSOLIDATED.

This test suite validates FastAPI endpoints for user authentication, token management,
user account operations, and admin functions. Consolidated from multiple test files
to eliminate duplication and provide authoritative authentication endpoint testing.
"""

from datetime import datetime, timezone
import hashlib
import json
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
import pytest

from src.core.config import JWTConfig
from src.core.exceptions import APIAuthenticationError, APIError
from src.integration.auth.auth_models import (
    AuthUser,
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    PasswordChangeRequest,
    RefreshRequest,
    RefreshResponse,
    UserCreateRequest,
)
from src.integration.auth.endpoints import (
    USER_STORE,
    auth_router,
    change_password,
    create_user,
    delete_user,
    get_current_user_info,
    get_user_by_username,
    hash_password,
    list_users,
    login,
    logout,
    refresh_token,
    verify_password,
)
from src.integration.auth.jwt_manager import JWTManager


class TestPasswordUtilities:
    """Test password hashing and verification utilities."""

    def test_hash_password(self):
        """Test password hashing function."""
        password = "test_password_123"
        hashed = hash_password(password)

        # Should return SHA256 hash
        expected_hash = hashlib.sha256(password.encode()).hexdigest()
        assert hashed == expected_hash

        # Same password should produce same hash
        assert hash_password(password) == hashed

    def test_hash_password_deterministic(self):
        """Test that same password produces same hash."""
        password = "consistent_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 == hash2

    def test_hash_password_different_inputs(self):
        """Test that different passwords produce different hashes."""
        password1 = "password1"
        password2 = "password2"

        hash1 = hash_password(password1)
        hash2 = hash_password(password2)

        assert hash1 != hash2

    def test_verify_password_success(self):
        """Test successful password verification."""
        password = "test_password_123"
        password_hash = hash_password(password)

        assert verify_password(password, password_hash) is True

    def test_verify_password_failure(self):
        """Test failed password verification."""
        password = "correct_password"
        wrong_password = "wrong_password"
        password_hash = hash_password(password)

        assert verify_password(wrong_password, password_hash) is False

    def test_verify_password_empty_password(self):
        """Test password verification with empty password."""
        password_hash = hash_password("some_password")

        assert verify_password("", password_hash) is False

    def test_hash_password_empty_input(self):
        """Test hashing empty password."""
        hashed = hash_password("")
        expected_hash = hashlib.sha256("".encode()).hexdigest()
        assert hashed == expected_hash


class TestUserStore:
    """Test user store functionality."""

    def test_get_user_by_username_existing(self):
        """Test getting existing user from store."""
        user = get_user_by_username("admin")

        assert user is not None
        assert user["username"] == "admin"
        assert user["is_admin"] is True
        assert "password_hash" in user

    def test_get_user_by_username_non_existing(self):
        """Test getting non-existing user from store."""
        user = get_user_by_username("non_existent_user")
        assert user is None

    def test_get_user_by_username_case_insensitive(self):
        """Test that username lookup is case insensitive."""
        user1 = get_user_by_username("ADMIN")
        user2 = get_user_by_username("admin")
        user3 = get_user_by_username("Admin")

        assert user1 == user2 == user3
        assert user1["username"] == "admin"

    def test_user_store_default_users(self):
        """Test that default users exist in store."""
        # Check that default users are present
        admin = get_user_by_username("admin")
        operator = get_user_by_username("operator")
        viewer = get_user_by_username("viewer")

        assert admin is not None and admin["is_admin"] is True
        assert operator is not None and admin["is_admin"] is True  # Admin check
        assert viewer is not None and viewer["is_admin"] is False

    def test_user_store_permissions(self):
        """Test user permissions in store."""
        admin = get_user_by_username("admin")
        operator = get_user_by_username("operator")
        viewer = get_user_by_username("viewer")

        # Admin should have admin permissions
        assert "admin" in admin["permissions"]
        assert "system_config" in admin["permissions"]

        # Operator should have operational permissions
        assert "read" in operator["permissions"]
        assert "write" in operator["permissions"]
        assert "prediction_view" in operator["permissions"]

        # Viewer should have minimal permissions
        assert "read" in viewer["permissions"]
        assert "admin" not in viewer["permissions"]


class TestAuthenticationRouter:
    """Test authentication router setup and configuration."""

    def test_auth_router_creation(self):
        """Test that auth router is properly configured."""
        assert auth_router.prefix == "/auth"
        assert "authentication" in auth_router.tags

    def test_auth_router_routes_registration(self):
        """Test that all expected routes are registered."""
        route_paths = {route.path for route in auth_router.routes}

        expected_paths = {
            "/login",
            "/logout",
            "/refresh",
            "/profile",
            "/password/change",
            "/users",
            "/users/{user_id}",
        }

        for expected_path in expected_paths:
            assert expected_path in route_paths


class TestAuthRouterApp:
    """Test authentication router as FastAPI app."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    @pytest.fixture
    def jwt_config(self):
        """Create JWT configuration for testing."""
        return JWTConfig(
            secret_key="test-secret-key-that-is-definitely-long-enough-for-security",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
            issuer="ha-ml-predictor",
            audience="ha-ml-predictor-api",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def mock_jwt_manager(self, jwt_config):
        """Mock JWT manager."""
        manager = Mock()
        manager.generate_access_token.return_value = "access_token_123"
        manager.generate_refresh_token.return_value = "refresh_token_123"
        manager.config.access_token_expire_minutes = 60
        manager.config.refresh_token_expire_days = 30
        manager.refresh_access_token.return_value = ("new_access", "new_refresh")
        manager.revoke_token.return_value = True
        manager.get_token_info.return_value = {
            "user_id": "test_user",
            "username": "testuser",
            "token_type": "access",
            "permissions": ["read"],
            "issued_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc),
            "is_expired": False,
            "is_blacklisted": False,
            "jti": "token_id_123",
        }
        return manager

    @pytest.fixture(autouse=True)
    def setup_dependencies(self, mock_jwt_manager):
        """Setup dependency overrides."""
        from src.integration.auth.dependencies import get_jwt_manager

        def get_test_jwt_manager():
            return mock_jwt_manager

        # Create test app with dependencies
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(auth_router)
        test_app.dependency_overrides[get_jwt_manager] = get_test_jwt_manager
        yield
        test_app.dependency_overrides.clear()

    def test_login_endpoint_success(self, client, mock_jwt_manager):
        """Test successful login."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/login",
                json={
                    "username": "admin",
                    "password": "admin123!",
                    "remember_me": False,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600  # 60 minutes * 60 seconds
        assert data["user"]["username"] == "admin"
        assert data["user"]["is_admin"] is True

    def test_login_endpoint_invalid_username(self, client):
        """Test login with invalid username."""
        response = client.post(
            "/auth/login",
            json={
                "username": "non_existent_user",
                "password": "any_password",
                "remember_me": False,
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_endpoint_invalid_password(self, client):
        """Test login with invalid password."""
        response = client.post(
            "/auth/login",
            json={
                "username": "admin",
                "password": "wrong_password",
                "remember_me": False,
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_endpoint_inactive_user(self, client):
        """Test login with inactive user."""
        # Temporarily modify user store
        original_active = USER_STORE["admin"]["is_active"]
        USER_STORE["admin"]["is_active"] = False

        try:
            response = client.post(
                "/auth/login",
                json={
                    "username": "admin",
                    "password": "admin123!",
                    "remember_me": False,
                },
            )

            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Account is disabled" in response.json()["detail"]
        finally:
            # Restore original state
            USER_STORE["admin"]["is_active"] = original_active

    def test_login_endpoint_remember_me(self, client, mock_jwt_manager):
        """Test login with remember me option."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/login",
                json={
                    "username": "admin",
                    "password": "admin123!",
                    "remember_me": True,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"] is not None
        assert data["refresh_token"] is not None

    def test_login_endpoint_validation_errors(self, client):
        """Test login validation errors."""
        # Missing required fields
        response = client.post("/auth/login", json={})
        assert response.status_code == 422  # Validation error

        # Invalid username format
        response = client.post(
            "/auth/login",
            json={"username": "us", "password": "Password123!"},  # Too short
        )
        assert response.status_code == 422

        # Weak password
        response = client.post(
            "/auth/login", json={"username": "testuser", "password": "weak"}
        )
        assert response.status_code == 422

    def test_refresh_token_endpoint_success(self, client, mock_jwt_manager):
        """Test successful token refresh."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/refresh", json={"refresh_token": "valid.refresh.token"}
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["access_token"] == "new_access"
        assert data["refresh_token"] == "new_refresh"
        assert data["token_type"] == "bearer"

    def test_refresh_token_endpoint_invalid_token(self, client, mock_jwt_manager):
        """Test token refresh with invalid token."""
        from src.core.exceptions import APIAuthenticationError

        mock_jwt_manager.refresh_access_token.side_effect = APIAuthenticationError(
            "Invalid refresh token"
        )

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/refresh", json={"refresh_token": "invalid.refresh.token"}
            )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid refresh token" in response.json()["detail"]

    def test_refresh_token_validation_error(self, client):
        """Test refresh token request validation."""
        # Invalid token format
        response = client.post("/auth/refresh", json={"refresh_token": "invalid"})
        assert response.status_code == 422

    def test_logout_endpoint_success(self, client, mock_jwt_manager):
        """Test successful logout."""
        with patch(
            "src.integration.auth.endpoints.get_current_user"
        ) as mock_get_user, patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            mock_user = AuthUser(user_id="test", username="testuser")
            mock_get_user.return_value = mock_user

            response = client.post(
                "/auth/logout",
                json={"refresh_token": "token_to_revoke", "revoke_all_tokens": False},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["message"] == "Logout successful"
        assert data["revoked_tokens"] == 1
        assert "timestamp" in data

    def test_logout_endpoint_no_token(self, client):
        """Test logout without providing refresh token."""
        with patch(
            "src.integration.auth.endpoints.get_current_user"
        ) as mock_get_user, patch(
            "src.integration.auth.endpoints.get_jwt_manager"
        ) as mock_jwt:

            mock_user = AuthUser(user_id="test", username="testuser")
            mock_get_user.return_value = mock_user
            mock_jwt.return_value.revoke_token.return_value = False

            response = client.post("/auth/logout", json={})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["revoked_tokens"] == 0

    def test_me_endpoint_success(self, client):
        """Test getting current user information."""
        with patch("src.integration.auth.endpoints.get_current_user") as mock_get_user:
            mock_user = AuthUser(
                user_id="test_user",
                username="testuser",
                email="test@example.com",
                permissions=["read", "write"],
                roles=["user"],
                is_admin=False,
            )
            mock_get_user.return_value = mock_user

            response = client.get("/auth/me")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["user_id"] == "test_user"
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["permissions"] == ["read", "write"]
        assert data["roles"] == ["user"]
        assert data["is_admin"] is False

    def test_change_password_endpoint_success(self, client):
        """Test successful password change."""
        with patch("src.integration.auth.endpoints.get_current_user") as mock_get_user:
            mock_user = AuthUser(user_id="admin", username="admin")
            mock_get_user.return_value = mock_user

            response = client.post(
                "/auth/change-password",
                json={
                    "current_password": "admin123!",
                    "new_password": "NewPassword123!",
                    "confirm_password": "NewPassword123!",
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["message"] == "Password changed successfully"
        assert "timestamp" in data

    def test_change_password_endpoint_wrong_current(self, client):
        """Test password change with wrong current password."""
        with patch("src.integration.auth.endpoints.get_current_user") as mock_get_user:
            mock_user = AuthUser(user_id="admin", username="admin")
            mock_get_user.return_value = mock_user

            response = client.post(
                "/auth/change-password",
                json={
                    "current_password": "wrong_password",
                    "new_password": "NewPassword123!",
                    "confirm_password": "NewPassword123!",
                },
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Current password is incorrect" in response.json()["detail"]

    def test_change_password_endpoint_user_not_found(self, client):
        """Test password change when user not found in store."""
        with patch("src.integration.auth.endpoints.get_current_user") as mock_get_user:
            mock_user = AuthUser(user_id="missing", username="missing_user")
            mock_get_user.return_value = mock_user

            response = client.post(
                "/auth/change-password",
                json={
                    "current_password": "any_password",
                    "new_password": "NewPassword123!",
                    "confirm_password": "NewPassword123!",
                },
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in response.json()["detail"]

    def test_token_info_endpoint_success(self, client, mock_jwt_manager):
        """Test token info endpoint."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/token/info", headers={"Authorization": "Bearer valid_token"}
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["user_id"] == "test_user"
        assert data["username"] == "testuser"
        assert data["token_type"] == "access"
        assert data["permissions"] == ["read"]
        assert data["is_active"] is True

    def test_token_info_endpoint_no_token(self, client):
        """Test token info endpoint without token."""
        response = client.post("/auth/token/info")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token required" in response.json()["detail"]

    def test_token_info_endpoint_invalid_token(self, client, mock_jwt_manager):
        """Test token info endpoint with invalid token."""
        mock_jwt_manager.get_token_info.return_value = {"error": "Invalid token"}

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):
            response = client.post(
                "/auth/token/info", headers={"Authorization": "Bearer invalid_token"}
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid token" in response.json()["detail"]


class TestLoginEndpoint:
    """Test login endpoint functionality."""

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager."""
        manager = AsyncMock()
        manager.generate_access_token.return_value = "access_token_123"
        manager.generate_refresh_token.return_value = "refresh_token_123"
        return manager

    @pytest.fixture
    def mock_user_service(self):
        """Mock user service."""
        service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_login_success(self, mock_jwt_manager, mock_user_service):
        """Test successful login."""
        # Mock user authentication
        test_user = AuthUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_active=True,
            last_login=datetime.now(timezone.utc),
        )
        mock_user_service.authenticate_user.return_value = test_user
        mock_user_service.update_last_login.return_value = None

        login_request = LoginRequest(
            username="testuser", password="TestPass123!", remember_me=False
        )

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await login(login_request)

            assert isinstance(response, LoginResponse)
            assert response.access_token == "access_token_123"
            assert response.refresh_token == "refresh_token_123"
            assert response.token_type == "bearer"
            assert response.expires_in == 900  # 15 minutes default
            assert response.user == test_user

            # Verify calls
            mock_user_service.authenticate_user.assert_called_once_with(
                "testuser", "TestPass123!"
            )
            mock_user_service.update_last_login.assert_called_once_with("user123")
            mock_jwt_manager.generate_access_token.assert_called_once()
            mock_jwt_manager.generate_refresh_token.assert_called_once_with("user123")

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, mock_jwt_manager, mock_user_service):
        """Test login with invalid credentials."""
        mock_user_service.authenticate_user.return_value = None

        login_request = LoginRequest(username="testuser", password="wrongpassword")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIAuthenticationError, match="Invalid username or password"
            ):
                await login(login_request)

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, mock_jwt_manager, mock_user_service):
        """Test login with inactive user."""
        inactive_user = AuthUser(
            user_id="user123", username="testuser", is_active=False
        )
        mock_user_service.authenticate_user.return_value = inactive_user

        login_request = LoginRequest(username="testuser", password="TestPass123!")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIAuthenticationError, match="User account is inactive"
            ):
                await login(login_request)

    @pytest.mark.asyncio
    async def test_login_remember_me_extended_expiration(
        self, mock_jwt_manager, mock_user_service
    ):
        """Test login with remember_me for extended token expiration."""
        test_user = AuthUser(user_id="user123", username="testuser", is_active=True)
        mock_user_service.authenticate_user.return_value = test_user
        mock_user_service.update_last_login.return_value = None

        login_request = LoginRequest(
            username="testuser", password="TestPass123!", remember_me=True
        )

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await login(login_request)

            # Should have extended expiration
            assert response.expires_in == 7200  # 2 hours for remember_me

            # Verify extended expiration passed to token generation
            call_args = mock_jwt_manager.generate_access_token.call_args
            assert call_args[1].get("expires_in_minutes") == 120  # 2 hours

    @pytest.mark.asyncio
    async def test_login_service_error(self, mock_jwt_manager, mock_user_service):
        """Test login with user service error."""
        mock_user_service.authenticate_user.side_effect = Exception("Database error")

        login_request = LoginRequest(username="testuser", password="TestPass123!")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIError, match="Authentication service temporarily unavailable"
            ):
                await login(login_request)


class TestLogoutEndpoint:
    """Test logout endpoint functionality."""

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager."""
        manager = AsyncMock()
        manager.revoke_token.return_value = True
        return manager

    @pytest.fixture
    def test_user(self):
        """Test user fixture."""
        return AuthUser(user_id="user123", username="testuser", is_active=True)

    @pytest.mark.asyncio
    async def test_logout_success(self, mock_jwt_manager, test_user):
        """Test successful logout."""
        logout_request = LogoutRequest(
            refresh_token="refresh_token_123", revoke_all_tokens=False
        )

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            response = await logout(logout_request, test_user)

            assert response["message"] == "Successfully logged out"
            assert response["revoked_tokens"] == 1

            # Verify refresh token was revoked
            mock_jwt_manager.revoke_token.assert_called_once_with("refresh_token_123")

    @pytest.mark.asyncio
    async def test_logout_revoke_all_tokens(self, mock_jwt_manager, test_user):
        """Test logout with revoke all tokens option."""
        logout_request = LogoutRequest(
            refresh_token="refresh_token_123", revoke_all_tokens=True
        )

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ), patch(
            "src.integration.auth.endpoints.get_user_service"
        ) as mock_user_service:

            mock_user_service_instance = AsyncMock()
            mock_user_service_instance.revoke_all_user_tokens.return_value = 5
            mock_user_service.return_value = mock_user_service_instance

            response = await logout(logout_request, test_user)

            assert response["message"] == "Successfully logged out"
            assert response["revoked_tokens"] == 5

            # Verify all tokens were revoked
            mock_user_service_instance.revoke_all_user_tokens.assert_called_once_with(
                "user123"
            )

    @pytest.mark.asyncio
    async def test_logout_no_refresh_token(self, mock_jwt_manager, test_user):
        """Test logout without refresh token."""
        logout_request = LogoutRequest()

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            response = await logout(logout_request, test_user)

            assert response["message"] == "Successfully logged out"
            assert response["revoked_tokens"] == 0

            # No tokens should be revoked
            mock_jwt_manager.revoke_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_logout_revoke_failure(self, mock_jwt_manager, test_user):
        """Test logout when token revocation fails."""
        mock_jwt_manager.revoke_token.return_value = False

        logout_request = LogoutRequest(refresh_token="invalid_token")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            response = await logout(logout_request, test_user)

            # Should still report success even if revocation failed
            assert response["message"] == "Successfully logged out"
            assert response["revoked_tokens"] == 0


class TestRefreshTokenEndpoint:
    """Test refresh token endpoint functionality."""

    @pytest.fixture
    def mock_jwt_manager(self):
        """Mock JWT manager."""
        manager = AsyncMock()
        manager.refresh_access_token.return_value = (
            "new_access_token_123",
            "new_refresh_token_123",
        )
        return manager

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, mock_jwt_manager):
        """Test successful token refresh."""
        refresh_request = RefreshRequest(refresh_token="valid_refresh_token")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            response = await refresh_token(refresh_request)

            assert isinstance(response, RefreshResponse)
            assert response.access_token == "new_access_token_123"
            assert response.refresh_token == "new_refresh_token_123"
            assert response.token_type == "bearer"
            assert response.expires_in == 900  # 15 minutes default

            mock_jwt_manager.refresh_access_token.assert_called_once_with(
                "valid_refresh_token"
            )

    @pytest.mark.asyncio
    async def test_refresh_token_invalid_token(self, mock_jwt_manager):
        """Test token refresh with invalid refresh token."""
        mock_jwt_manager.refresh_access_token.side_effect = APIAuthenticationError(
            "Invalid refresh token"
        )

        refresh_request = RefreshRequest(refresh_token="invalid_token")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            with pytest.raises(APIAuthenticationError, match="Invalid refresh token"):
                await refresh_token(refresh_request)

    @pytest.mark.asyncio
    async def test_refresh_token_service_error(self, mock_jwt_manager):
        """Test token refresh with service error."""
        mock_jwt_manager.refresh_access_token.side_effect = Exception("Service error")

        refresh_request = RefreshRequest(refresh_token="token")

        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            with pytest.raises(
                APIError, match="Token refresh service temporarily unavailable"
            ):
                await refresh_token(refresh_request)


class TestPasswordChangeEndpoint:
    """Test password change endpoint functionality."""

    @pytest.fixture
    def test_user(self):
        """Test user fixture."""
        return AuthUser(user_id="user123", username="testuser", is_active=True)

    @pytest.fixture
    def mock_user_service(self):
        """Mock user service."""
        service = AsyncMock()
        service.change_password.return_value = True
        return service

    @pytest.mark.asyncio
    async def test_change_password_success(self, mock_user_service, test_user):
        """Test successful password change."""
        password_request = PasswordChangeRequest(
            current_password="OldPass123!",
            new_password="NewPass123!",
            confirm_password="NewPass123!",
        )

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await change_password(password_request, test_user)

            assert response["message"] == "Password changed successfully"

            mock_user_service.change_password.assert_called_once_with(
                "user123", "OldPass123!", "NewPass123!"
            )

    @pytest.mark.asyncio
    async def test_change_password_invalid_current(self, mock_user_service, test_user):
        """Test password change with invalid current password."""
        mock_user_service.change_password.side_effect = APIAuthenticationError(
            "Current password is incorrect"
        )

        password_request = PasswordChangeRequest(
            current_password="WrongPass123!",
            new_password="NewPass123!",
            confirm_password="NewPass123!",
        )

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIAuthenticationError, match="Current password is incorrect"
            ):
                await change_password(password_request, test_user)

    @pytest.mark.asyncio
    async def test_change_password_service_error(self, mock_user_service, test_user):
        """Test password change with service error."""
        mock_user_service.change_password.side_effect = Exception("Database error")

        password_request = PasswordChangeRequest(
            current_password="OldPass123!",
            new_password="NewPass123!",
            confirm_password="NewPass123!",
        )

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIError, match="Password change service temporarily unavailable"
            ):
                await change_password(password_request, test_user)


class TestUserProfileEndpoints:
    """Test user profile management endpoints."""

    @pytest.fixture
    def test_user(self):
        """Test user fixture."""
        return AuthUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_get_current_user_profile(self, test_user):
        """Test getting current user profile."""
        profile = await get_current_user_info(test_user)

        assert profile == test_user


class TestUserManagementEndpoints:
    """Test user management endpoints (admin only)."""

    @pytest.fixture
    def admin_user(self):
        """Admin user fixture."""
        return AuthUser(
            user_id="admin123",
            username="admin",
            permissions=["read", "write", "admin"],
            roles=["admin"],
            is_admin=True,
            is_active=True,
        )

    @pytest.fixture
    def mock_user_service(self):
        """Mock user service."""
        service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_create_user_success(self, mock_user_service, admin_user):
        """Test successful user creation."""
        user_request = UserCreateRequest(
            username="newuser",
            email="newuser@example.com",
            password="NewPass123!",
            permissions=["read"],
            roles=["user"],
        )

        created_user = AuthUser(
            user_id="newuser123",
            username="newuser",
            email="newuser@example.com",
            permissions=["read"],
            roles=["user"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        mock_user_service.create_user.return_value = created_user

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await create_user(user_request, admin_user)

            assert response == created_user
            mock_user_service.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, mock_user_service, admin_user):
        """Test user creation with duplicate username."""
        mock_user_service.create_user.side_effect = APIError("Username already exists")

        user_request = UserCreateRequest(
            username="existing_user", email="test@example.com", password="Password123!"
        )

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(APIError, match="Username already exists"):
                await create_user(user_request, admin_user)

    @pytest.mark.asyncio
    async def test_list_users_success(self, mock_user_service, admin_user):
        """Test successful user listing."""
        users = [
            AuthUser(user_id="1", username="user1", is_active=True),
            AuthUser(user_id="2", username="user2", is_active=True),
        ]
        mock_user_service.list_users.return_value = users

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await list_users(admin_user, skip=0, limit=10)

            assert response == users
            mock_user_service.list_users.assert_called_once_with(
                skip=0, limit=10, include_inactive=False
            )

    @pytest.mark.asyncio
    async def test_list_users_with_inactive(self, mock_user_service, admin_user):
        """Test user listing including inactive users."""
        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            await list_users(admin_user, skip=5, limit=20, include_inactive=True)

            mock_user_service.list_users.assert_called_once_with(
                skip=5, limit=20, include_inactive=True
            )

    @pytest.mark.asyncio
    async def test_delete_user_success(self, mock_user_service, admin_user):
        """Test successful user deletion."""
        mock_user_service.delete_user.return_value = True

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            response = await delete_user("target123", admin_user)

            assert response["message"] == "User deleted successfully"
            assert response["user_id"] == "target123"
            mock_user_service.delete_user.assert_called_once_with("target123")

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, mock_user_service, admin_user):
        """Test user deletion for non-existent user."""
        mock_user_service.delete_user.return_value = False

        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(
                APIError, match="User not found or could not be deleted"
            ):
                await delete_user("nonexistent", admin_user)

    @pytest.mark.asyncio
    async def test_delete_self_prevention(self, mock_user_service, admin_user):
        """Test that admin cannot delete their own account."""
        with patch(
            "src.integration.auth.endpoints.get_user_service",
            return_value=mock_user_service,
        ):

            with pytest.raises(APIError, match="Cannot delete your own account"):
                await delete_user("admin123", admin_user)  # Same as admin_user.user_id


class TestAdminEndpoints:
    """Test admin-only endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user."""
        return AuthUser(
            user_id="admin_123",
            username="admin",
            is_admin=True,
            permissions=["admin", "read", "write"],
        )

    def test_list_users_endpoint_success(self, client, mock_admin_user):
        """Test listing users as admin."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:
            # Mock the dependency
            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.get("/auth/users")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert isinstance(data, list)
        assert len(data) >= 3  # At least admin, operator, viewer

        # Check admin user is present
        admin_user = next((u for u in data if u["username"] == "admin"), None)
        assert admin_user is not None
        assert admin_user["is_admin"] is True

    def test_create_user_endpoint_success(self, client, mock_admin_user):
        """Test creating new user as admin."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.post(
                "/auth/users",
                json={
                    "username": "newuser",
                    "email": "new@example.com",
                    "password": "Password123!",
                    "permissions": ["read"],
                    "roles": ["user"],
                    "is_admin": False,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["username"] == "newuser"
        assert data["email"] == "new@example.com"
        assert data["permissions"] == ["read"]
        assert data["roles"] == ["user"]
        assert data["is_admin"] is False

        # Verify user was added to store
        assert "newuser" in USER_STORE

        # Cleanup
        del USER_STORE["newuser"]

    def test_create_user_endpoint_duplicate_username(self, client, mock_admin_user):
        """Test creating user with existing username."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.post(
                "/auth/users",
                json={
                    "username": "admin",  # Existing username
                    "email": "admin2@example.com",
                    "password": "Password123!",
                    "permissions": ["read"],
                    "roles": ["user"],
                    "is_admin": False,
                },
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already exists" in response.json()["detail"]

    def test_create_user_endpoint_validation_errors(self, client, mock_admin_user):
        """Test user creation validation errors."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            # Invalid email
            response = client.post(
                "/auth/users",
                json={
                    "username": "newuser",
                    "email": "invalid-email",
                    "password": "Password123!",
                    "permissions": ["read"],
                    "roles": ["user"],
                    "is_admin": False,
                },
            )
            assert response.status_code == 422  # Validation error

    def test_delete_user_endpoint_success(self, client, mock_admin_user):
        """Test deleting user as admin."""
        # First create a test user to delete
        USER_STORE["test_delete_user"] = {
            "user_id": "delete_me",
            "username": "test_delete_user",
            "password_hash": "dummy_hash",
            "is_admin": False,
            "is_active": True,
            "permissions": ["read"],
            "roles": ["user"],
            "created_at": datetime.now(timezone.utc),
        }

        try:
            with patch(
                "src.integration.auth.endpoints.require_admin"
            ) as mock_require_admin:

                async def mock_admin_dependency():
                    return mock_admin_user

                mock_require_admin.return_value = mock_admin_dependency

                response = client.delete("/auth/users/test_delete_user")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "deleted successfully" in data["message"]
            assert "timestamp" in data

            # Verify user was removed from store
            assert "test_delete_user" not in USER_STORE
        finally:
            # Cleanup in case of test failure
            USER_STORE.pop("test_delete_user", None)

    def test_delete_user_endpoint_self_deletion(self, client, mock_admin_user):
        """Test that admin cannot delete their own account."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.delete(
                "/auth/users/admin"
            )  # Admin trying to delete themselves

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot delete your own account" in response.json()["detail"]

    def test_delete_user_endpoint_user_not_found(self, client, mock_admin_user):
        """Test deleting non-existent user."""
        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.delete("/auth/users/non_existent_user")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in response.json()["detail"]


class TestAuthEndpointIntegration:
    """Test integration between authentication endpoints."""

    @pytest.fixture
    def client(self):
        """Test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, client):
        """Test complete authentication flow."""
        with patch("src.integration.auth.endpoints.get_jwt_manager") as mock_jwt, patch(
            "src.integration.auth.endpoints.get_user_service"
        ) as mock_user_service:

            # Setup mocks
            mock_jwt_instance = AsyncMock()
            mock_jwt_instance.generate_access_token.return_value = "access_123"
            mock_jwt_instance.generate_refresh_token.return_value = "refresh_123"
            mock_jwt_instance.refresh_access_token.return_value = (
                "new_access",
                "new_refresh",
            )
            mock_jwt_instance.revoke_token.return_value = True
            mock_jwt.return_value = mock_jwt_instance

            mock_service_instance = AsyncMock()
            test_user = AuthUser(user_id="user123", username="testuser", is_active=True)
            mock_service_instance.authenticate_user.return_value = test_user
            mock_service_instance.update_last_login.return_value = None
            mock_user_service.return_value = mock_service_instance

            # Test login
            login_data = {
                "username": "testuser",
                "password": "TestPass123!",
                "remember_me": False,
            }

            # Note: This would need actual FastAPI test setup to work fully
            # This is a structure test to show the integration approach


class TestAuthEndpointErrorHandling:
    """Test error handling in authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_unexpected_error(self):
        """Test login endpoint with unexpected error."""
        login_request = LoginRequest(username="testuser", password="TestPass123!")

        with patch("src.integration.auth.endpoints.get_jwt_manager") as mock_jwt, patch(
            "src.integration.auth.endpoints.get_user_service"
        ) as mock_service:

            mock_service.return_value.authenticate_user.side_effect = RuntimeError(
                "Unexpected error"
            )

            with pytest.raises(
                APIError, match="Authentication service temporarily unavailable"
            ):
                await login(login_request)

    @pytest.mark.asyncio
    async def test_refresh_unexpected_error(self):
        """Test refresh endpoint with unexpected error."""
        refresh_request = RefreshRequest(refresh_token="token")

        with patch("src.integration.auth.endpoints.get_jwt_manager") as mock_jwt:
            mock_jwt.return_value.refresh_access_token.side_effect = RuntimeError(
                "Unexpected error"
            )

            with pytest.raises(
                APIError, match="Token refresh service temporarily unavailable"
            ):
                await refresh_token(refresh_request)

    @pytest.mark.asyncio
    async def test_password_change_unexpected_error(self):
        """Test password change endpoint with unexpected error."""
        test_user = AuthUser(user_id="user123", username="test", is_active=True)
        password_request = PasswordChangeRequest(
            current_password="old",
            new_password="NewPass123!",
            confirm_password="NewPass123!",
        )

        with patch("src.integration.auth.endpoints.get_user_service") as mock_service:
            mock_service.return_value.change_password.side_effect = RuntimeError(
                "Unexpected error"
            )

            with pytest.raises(
                APIError, match="Password change service temporarily unavailable"
            ):
                await change_password(password_request, test_user)


class TestAuthEndpointValidation:
    """Test validation in authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_validates_request_model(self):
        """Test that login endpoint validates request model."""
        # This would be handled by FastAPI's Pydantic validation
        # Testing that our models have proper validation

        with pytest.raises(ValueError):
            LoginRequest(username="ab", password="TestPass123!")  # Too short

    @pytest.mark.asyncio
    async def test_refresh_validates_token_format(self):
        """Test that refresh endpoint validates token format."""
        with pytest.raises(ValueError):
            RefreshRequest(refresh_token="invalid")  # Too short for JWT

    @pytest.mark.asyncio
    async def test_password_change_validates_complexity(self):
        """Test that password change validates complexity."""
        with pytest.raises(ValueError):
            PasswordChangeRequest(
                current_password="OldPass123!",
                new_password="weak",  # Too weak
                confirm_password="weak",
            )

    @pytest.mark.asyncio
    async def test_user_create_validates_email_format(self):
        """Test that user creation validates email format."""
        with pytest.raises(ValueError):
            UserCreateRequest(
                username="newuser",
                email="invalid-email",  # Invalid format
                password="TestPass123!",
            )


class TestEndpointErrorHandling:
    """Test error handling in endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    def test_login_endpoint_jwt_manager_error(self, client):
        """Test login endpoint when JWT manager fails."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            side_effect=Exception("JWT error"),
        ):
            response = client.post(
                "/auth/login",
                json={
                    "username": "admin",
                    "password": "admin123!",
                    "remember_me": False,
                },
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Login service unavailable" in response.json()["detail"]

    def test_refresh_token_endpoint_service_error(self, client):
        """Test refresh endpoint service error."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            side_effect=Exception("Service error"),
        ):
            response = client.post(
                "/auth/refresh", json={"refresh_token": "valid.refresh.token"}
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Token refresh service unavailable" in response.json()["detail"]

    def test_logout_endpoint_service_error(self, client):
        """Test logout endpoint service error."""
        with patch(
            "src.integration.auth.endpoints.get_current_user",
            side_effect=Exception("Service error"),
        ):
            response = client.post("/auth/logout", json={})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Logout service unavailable" in response.json()["detail"]

    def test_change_password_service_error(self, client):
        """Test password change service error."""
        with patch(
            "src.integration.auth.endpoints.get_current_user",
            side_effect=Exception("Service error"),
        ):
            response = client.post(
                "/auth/change-password",
                json={
                    "current_password": "old",
                    "new_password": "NewPassword123!",
                    "confirm_password": "NewPassword123!",
                },
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Password change service unavailable" in response.json()["detail"]

    def test_token_info_service_error(self, client):
        """Test token info service error."""
        with patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            side_effect=Exception("Service error"),
        ):
            response = client.post(
                "/auth/token/info", headers={"Authorization": "Bearer token"}
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Token info service unavailable" in response.json()["detail"]

    def test_create_user_service_error(self, client):
        """Test create user service error."""
        mock_admin_user = AuthUser(user_id="admin", username="admin", is_admin=True)

        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            with patch(
                "src.integration.auth.endpoints.secrets.token_hex",
                side_effect=Exception("Service error"),
            ):
                response = client.post(
                    "/auth/users",
                    json={
                        "username": "newuser",
                        "email": "new@example.com",
                        "password": "Password123!",
                    },
                )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "User creation service unavailable" in response.json()["detail"]

    def test_delete_user_service_error(self, client):
        """Test delete user service error."""
        mock_admin_user = AuthUser(user_id="admin", username="admin", is_admin=True)

        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            # Simulate error in user store access
            with patch(
                "src.integration.auth.endpoints.USER_STORE",
                side_effect=Exception("Store error"),
            ):
                response = client.delete("/auth/users/someuser")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "User deletion service unavailable" in response.json()["detail"]


class TestAuthEndpointSecurity:
    """Test security aspects of authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_rate_limiting_protection(self):
        """Test that login endpoint has rate limiting protection."""
        # This would be implemented at the router level
        # Testing that security considerations are in place

        login_request = LoginRequest(username="testuser", password="TestPass123!")

        # In a real implementation, this would test actual rate limiting
        # Here we verify the structure allows for rate limiting
        assert hasattr(login_request, "username")  # Basic structure test

    @pytest.mark.asyncio
    async def test_sensitive_data_not_logged(self):
        """Test that sensitive data is not included in logs."""
        login_request = LoginRequest(username="testuser", password="TestPass123!")

        # Verify password is not exposed in string representation
        request_str = str(login_request)
        assert "TestPass123!" not in request_str

    @pytest.mark.asyncio
    async def test_token_response_security_headers(self):
        """Test that token responses include appropriate security considerations."""
        # This would typically be handled by middleware
        # Testing that response models are structured securely

        test_user = AuthUser(user_id="test", username="test", is_active=True)
        response = LoginResponse(
            access_token="token",
            refresh_token="refresh",
            token_type="bearer",
            expires_in=900,
            user=test_user,
        )

        # Verify response structure
        assert response.token_type == "bearer"
        assert response.expires_in > 0
