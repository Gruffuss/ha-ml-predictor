"""
Comprehensive unit tests for authentication endpoints module.

This test suite validates FastAPI endpoints for user authentication, token management,
user account operations, and admin functions.
"""

from datetime import datetime, timezone
import hashlib
import json
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
import pytest

from src.integration.auth.auth_models import (
    AuthUser,
    LoginRequest,
    PasswordChangeRequest,
    UserCreateRequest,
)
from src.integration.auth.endpoints import (
    USER_STORE,
    auth_router,
    get_user_by_username,
    hash_password,
    verify_password,
)


class TestPasswordUtilities:
    """Test password utility functions."""

    def test_hash_password(self):
        """Test password hashing function."""
        password = "test_password_123"
        hashed = hash_password(password)

        # Should return SHA256 hash
        expected_hash = hashlib.sha256(password.encode()).hexdigest()
        assert hashed == expected_hash

        # Same password should produce same hash
        assert hash_password(password) == hashed

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
    def mock_jwt_manager(self):
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


class TestEndpointsIntegration:
    """Test integration scenarios across endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    def test_complete_user_lifecycle(self, client):
        """Test complete user creation, login, and deletion lifecycle."""
        mock_admin_user = AuthUser(user_id="admin", username="admin", is_admin=True)
        mock_jwt_manager = Mock()
        mock_jwt_manager.generate_access_token.return_value = "access_token"
        mock_jwt_manager.generate_refresh_token.return_value = "refresh_token"
        mock_jwt_manager.config.access_token_expire_minutes = 60

        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin, patch(
            "src.integration.auth.endpoints.get_jwt_manager",
            return_value=mock_jwt_manager,
        ):

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            # 1. Create user
            create_response = client.post(
                "/auth/users",
                json={
                    "username": "lifecycle_user",
                    "email": "lifecycle@example.com",
                    "password": "Password123!",
                    "permissions": ["read"],
                    "roles": ["user"],
                    "is_admin": False,
                },
            )
            assert create_response.status_code == status.HTTP_200_OK

            # 2. Login as new user
            login_response = client.post(
                "/auth/login",
                json={
                    "username": "lifecycle_user",
                    "password": "Password123!",
                    "remember_me": False,
                },
            )
            assert login_response.status_code == status.HTTP_200_OK

            # 3. Delete user
            delete_response = client.delete("/auth/users/lifecycle_user")
            assert delete_response.status_code == status.HTTP_200_OK

            # 4. Verify user no longer exists
            login_after_delete = client.post(
                "/auth/login",
                json={
                    "username": "lifecycle_user",
                    "password": "Password123!",
                    "remember_me": False,
                },
            )
            assert login_after_delete.status_code == status.HTTP_401_UNAUTHORIZED

    def test_password_change_and_login(self, client):
        """Test password change affects login."""
        mock_jwt_manager = Mock()
        mock_jwt_manager.generate_access_token.return_value = "access_token"
        mock_jwt_manager.generate_refresh_token.return_value = "refresh_token"
        mock_jwt_manager.config.access_token_expire_minutes = 60

        # Create test user for password change
        test_username = "password_change_user"
        USER_STORE[test_username] = {
            "user_id": "pwd_user",
            "username": test_username,
            "password_hash": hash_password("OldPassword123!"),
            "permissions": ["read"],
            "roles": ["user"],
            "is_admin": False,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }

        try:
            with patch(
                "src.integration.auth.endpoints.get_current_user"
            ) as mock_get_user, patch(
                "src.integration.auth.endpoints.get_jwt_manager",
                return_value=mock_jwt_manager,
            ):

                mock_user = AuthUser(user_id="pwd_user", username=test_username)
                mock_get_user.return_value = mock_user

                # 1. Login with old password
                login_old = client.post(
                    "/auth/login",
                    json={
                        "username": test_username,
                        "password": "OldPassword123!",
                        "remember_me": False,
                    },
                )
                assert login_old.status_code == status.HTTP_200_OK

                # 2. Change password
                change_response = client.post(
                    "/auth/change-password",
                    json={
                        "current_password": "OldPassword123!",
                        "new_password": "NewPassword123!",
                        "confirm_password": "NewPassword123!",
                    },
                )
                assert change_response.status_code == status.HTTP_200_OK

                # 3. Login with new password should work
                login_new = client.post(
                    "/auth/login",
                    json={
                        "username": test_username,
                        "password": "NewPassword123!",
                        "remember_me": False,
                    },
                )
                assert login_new.status_code == status.HTTP_200_OK

                # 4. Login with old password should fail
                login_old_after = client.post(
                    "/auth/login",
                    json={
                        "username": test_username,
                        "password": "OldPassword123!",
                        "remember_me": False,
                    },
                )
                assert login_old_after.status_code == status.HTTP_401_UNAUTHORIZED

        finally:
            # Cleanup
            USER_STORE.pop(test_username, None)


class TestEndpointsPerformance:
    """Test performance aspects of endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    def test_login_performance_with_large_user_store(self, client):
        """Test login performance with many users in store."""
        # Temporarily add many users to store
        original_store = USER_STORE.copy()

        try:
            # Add 100 test users
            for i in range(100):
                username = f"test_user_{i}"
                USER_STORE[username] = {
                    "user_id": f"user_{i}",
                    "username": username,
                    "password_hash": hash_password("Password123!"),
                    "permissions": ["read"],
                    "roles": ["user"],
                    "is_admin": False,
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc),
                }

            mock_jwt_manager = Mock()
            mock_jwt_manager.generate_access_token.return_value = "access_token"
            mock_jwt_manager.generate_refresh_token.return_value = "refresh_token"
            mock_jwt_manager.config.access_token_expire_minutes = 60

            with patch(
                "src.integration.auth.endpoints.get_jwt_manager",
                return_value=mock_jwt_manager,
            ):
                # Login should still be fast
                response = client.post(
                    "/auth/login",
                    json={
                        "username": "test_user_50",
                        "password": "Password123!",
                        "remember_me": False,
                    },
                )

            assert response.status_code == status.HTTP_200_OK

        finally:
            # Restore original store
            USER_STORE.clear()
            USER_STORE.update(original_store)

    def test_user_list_performance(self, client):
        """Test user listing performance."""
        mock_admin_user = AuthUser(user_id="admin", username="admin", is_admin=True)

        with patch(
            "src.integration.auth.endpoints.require_admin"
        ) as mock_require_admin:

            async def mock_admin_dependency():
                return mock_admin_user

            mock_require_admin.return_value = mock_admin_dependency

            response = client.get("/auth/users")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 3  # At least the default users


class TestEndpointsDocumentation:
    """Test that endpoints are properly documented."""

    @pytest.fixture
    def client(self):
        """Create test client with auth router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(auth_router)

        return TestClient(app)

    def test_openapi_schema_generation(self, client):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        schema = response.json()
        assert "paths" in schema

        # Check that auth endpoints are documented
        auth_paths = [path for path in schema["paths"] if path.startswith("/auth")]
        assert len(auth_paths) > 0

        # Check specific endpoints exist
        assert "/auth/login" in schema["paths"]
        assert "/auth/refresh" in schema["paths"]
        assert "/auth/logout" in schema["paths"]
        assert "/auth/me" in schema["paths"]
