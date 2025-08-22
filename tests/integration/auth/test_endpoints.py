"""
Comprehensive test suite for Authentication Endpoints.

This module provides complete test coverage for JWT-based authentication endpoints,
user management, and API security flows.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict
from unittest.mock import Mock, patch

# Create a test app with the auth router
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
import pytest

from src.core.config import JWTConfig
from src.integration.auth.auth_models import (
    AuthUser,
    LoginRequest,
    LogoutRequest,
    PasswordChangeRequest,
    RefreshRequest,
    UserCreateRequest,
)
from src.integration.auth.endpoints import (
    USER_STORE,
    auth_router,
    get_user_by_username,
    hash_password,
    verify_password,
)
from src.integration.auth.jwt_manager import JWTManager

test_app = FastAPI()
test_app.include_router(auth_router)


class TestPasswordUtilities:
    """Test password hashing and verification utilities."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123!"
        hashed = hash_password(password)

        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA256 hex digest length
        assert hashed != password  # Should be hashed, not plain text

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

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "correct_password"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        correct_password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hash_password(correct_password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        password = "test_password"
        hashed = hash_password(password)

        assert verify_password("", hashed) is False

    def test_get_user_by_username_existing(self):
        """Test retrieving existing user."""
        user = get_user_by_username("admin")

        assert user is not None
        assert user["username"] == "admin"
        assert user["user_id"] == "admin"
        assert "password_hash" in user

    def test_get_user_by_username_case_insensitive(self):
        """Test user retrieval is case insensitive."""
        user_lower = get_user_by_username("admin")
        user_upper = get_user_by_username("ADMIN")
        user_mixed = get_user_by_username("Admin")

        assert user_lower == user_upper == user_mixed

    def test_get_user_by_username_nonexistent(self):
        """Test retrieving non-existent user."""
        user = get_user_by_username("nonexistent_user")

        assert user is None


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(test_app)

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
        """Create mock JWT manager."""
        manager = JWTManager(jwt_config)
        return manager

    @pytest.fixture(autouse=True)
    def setup_dependencies(self, mock_jwt_manager):
        """Setup dependency overrides."""
        from src.integration.auth.dependencies import get_jwt_manager

        def get_test_jwt_manager():
            return mock_jwt_manager

        test_app.dependency_overrides[get_jwt_manager] = get_test_jwt_manager
        yield
        test_app.dependency_overrides.clear()

    def test_login_success(self, client):
        """Test successful login."""
        login_data = {
            "username": "admin",
            "password": "admin123!",
            "remember_me": False,
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "user" in data

        user_data = data["user"]
        assert user_data["username"] == "admin"
        assert user_data["is_admin"] is True
        assert "admin" in user_data["permissions"]

    def test_login_invalid_username(self, client):
        """Test login with invalid username."""
        login_data = {
            "username": "nonexistent",
            "password": "password",
            "remember_me": False,
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_invalid_password(self, client):
        """Test login with invalid password."""
        login_data = {
            "username": "admin",
            "password": "wrong_password",
            "remember_me": False,
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_inactive_user(self, client):
        """Test login with inactive user account."""
        # Temporarily deactivate user
        original_status = USER_STORE["operator"]["is_active"]
        USER_STORE["operator"]["is_active"] = False

        try:
            login_data = {
                "username": "operator",
                "password": "operator123!",
                "remember_me": False,
            }

            response = client.post("/auth/login", json=login_data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Account is disabled" in response.json()["detail"]
        finally:
            # Restore original status
            USER_STORE["operator"]["is_active"] = original_status

    def test_login_remember_me(self, client):
        """Test login with remember me option."""
        login_data = {"username": "admin", "password": "admin123!", "remember_me": True}

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK
        # The remember_me logic would extend refresh token expiration

    def test_login_missing_fields(self, client):
        """Test login with missing required fields."""
        # Missing password
        response = client.post("/auth/login", json={"username": "admin"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Missing username
        response = client.post("/auth/login", json={"password": "password"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_case_sensitivity(self, client):
        """Test login username case sensitivity."""
        login_data = {
            "username": "ADMIN",  # Uppercase
            "password": "admin123!",
            "remember_me": False,
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK

    def test_refresh_token_success(self, client, mock_jwt_manager):
        """Test successful token refresh."""
        # First login to get refresh token
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        refresh_token = login_response.json()["refresh_token"]

        # Use refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_refresh_token_invalid(self, client):
        """Test refresh with invalid token."""
        refresh_data = {"refresh_token": "invalid.refresh.token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_missing(self, client):
        """Test refresh with missing token."""
        response = client.post("/auth/refresh", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_logout_success(self, client):
        """Test successful logout."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]
        refresh_token = login_response.json()["refresh_token"]

        # Logout
        logout_data = {"refresh_token": refresh_token}
        response = client.post(
            "/auth/logout",
            json=logout_data,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "message" in data
        assert "revoked_tokens" in data
        assert "timestamp" in data

    def test_logout_without_refresh_token(self, client):
        """Test logout without providing refresh token."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Logout without refresh token
        response = client.post(
            "/auth/logout", json={}, headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_logout_unauthorized(self, client):
        """Test logout without authentication."""
        response = client.post("/auth/logout", json={})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_info(self, client):
        """Test getting current user information."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={
                "username": "operator",
                "password": "operator123!",
                "remember_me": False,
            },
        )

        access_token = login_response.json()["access_token"]

        # Get user info
        response = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["username"] == "operator"
        assert data["is_admin"] is False
        assert "operator" in data["roles"]

    def test_get_current_user_info_unauthorized(self, client):
        """Test getting user info without authentication."""
        response = client.get("/auth/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_change_password_success(self, client):
        """Test successful password change."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "viewer", "password": "viewer123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Change password
        password_data = {
            "current_password": "viewer123!",
            "new_password": "new_secure_password123!",
        }

        response = client.post(
            "/auth/change-password",
            json=password_data,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "message" in data
        assert "timestamp" in data

        # Verify old password no longer works
        old_login = client.post(
            "/auth/login",
            json={"username": "viewer", "password": "viewer123!", "remember_me": False},
        )
        assert old_login.status_code == status.HTTP_401_UNAUTHORIZED

        # Verify new password works
        new_login = client.post(
            "/auth/login",
            json={
                "username": "viewer",
                "password": "new_secure_password123!",
                "remember_me": False,
            },
        )
        assert new_login.status_code == status.HTTP_200_OK

        # Restore original password for other tests
        USER_STORE["viewer"]["password_hash"] = hash_password("viewer123!")

    def test_change_password_wrong_current(self, client):
        """Test password change with wrong current password."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Try to change with wrong current password
        password_data = {
            "current_password": "wrong_password",
            "new_password": "new_password123!",
        }

        response = client.post(
            "/auth/change-password",
            json=password_data,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Current password is incorrect" in response.json()["detail"]

    def test_change_password_unauthorized(self, client):
        """Test password change without authentication."""
        password_data = {
            "current_password": "current",
            "new_password": "new_password123!",
        }

        response = client.post("/auth/change-password", json=password_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_token_info(self, client):
        """Test getting token information."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Get token info
        response = client.post(
            "/auth/token/info", headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "user_id" in data
        assert "token_type" in data
        assert "permissions" in data
        assert "issued_at" in data
        assert "expires_at" in data
        assert "is_expired" in data
        assert "is_active" in data

    def test_get_token_info_without_token(self, client):
        """Test getting token info without providing token."""
        response = client.post("/auth/token/info")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestUserManagementEndpoints:
    """Test user management endpoints (admin only)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(test_app)

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
        """Create mock JWT manager."""
        manager = JWTManager(jwt_config)
        return manager

    @pytest.fixture(autouse=True)
    def setup_dependencies(self, mock_jwt_manager):
        """Setup dependency overrides."""
        from src.integration.auth.dependencies import get_jwt_manager

        def get_test_jwt_manager():
            return mock_jwt_manager

        test_app.dependency_overrides[get_jwt_manager] = get_test_jwt_manager
        yield
        test_app.dependency_overrides.clear()

    @pytest.fixture
    def admin_token(self, client):
        """Get admin access token."""
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )
        return login_response.json()["access_token"]

    @pytest.fixture
    def operator_token(self, client):
        """Get operator access token (non-admin)."""
        login_response = client.post(
            "/auth/login",
            json={
                "username": "operator",
                "password": "operator123!",
                "remember_me": False,
            },
        )
        return login_response.json()["access_token"]

    def test_list_users_as_admin(self, client, admin_token):
        """Test listing users as admin."""
        response = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        users = response.json()

        assert isinstance(users, list)
        assert len(users) >= 3  # admin, operator, viewer

        usernames = [user["username"] for user in users]
        assert "admin" in usernames
        assert "operator" in usernames
        assert "viewer" in usernames

    def test_list_users_as_non_admin(self, client, operator_token):
        """Test listing users as non-admin (should fail)."""
        response = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {operator_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_list_users_unauthorized(self, client):
        """Test listing users without authentication."""
        response = client.get("/auth/users")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_user_as_admin(self, client, admin_token):
        """Test creating user as admin."""
        user_data = {
            "username": "new_test_user",
            "email": "newuser@test.com",
            "password": "secure_password123!",
            "permissions": ["read", "prediction_view"],
            "roles": ["viewer"],
            "is_admin": False,
        }

        response = client.post(
            "/auth/users",
            json=user_data,
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        created_user = response.json()

        assert created_user["username"] == "new_test_user"
        assert created_user["email"] == "newuser@test.com"
        assert created_user["permissions"] == ["read", "prediction_view"]
        assert created_user["is_admin"] is False
        assert "password_hash" not in created_user  # Password should not be returned

        # Verify user was added to store
        assert "new_test_user" in USER_STORE

        # Clean up
        if "new_test_user" in USER_STORE:
            del USER_STORE["new_test_user"]

    def test_create_user_duplicate_username(self, client, admin_token):
        """Test creating user with existing username."""
        user_data = {
            "username": "admin",  # Already exists
            "email": "duplicate@test.com",
            "password": "password123!",
            "permissions": ["read"],
            "roles": ["viewer"],
            "is_admin": False,
        }

        response = client.post(
            "/auth/users",
            json=user_data,
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already exists" in response.json()["detail"]

    def test_create_user_as_non_admin(self, client, operator_token):
        """Test creating user as non-admin (should fail)."""
        user_data = {
            "username": "unauthorized_creation",
            "email": "test@test.com",
            "password": "password123!",
            "permissions": ["read"],
            "roles": ["viewer"],
            "is_admin": False,
        }

        response = client.post(
            "/auth/users",
            json=user_data,
            headers={"Authorization": f"Bearer {operator_token}"},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_create_user_invalid_data(self, client, admin_token):
        """Test creating user with invalid data."""
        # Missing required fields
        response = client.post(
            "/auth/users",
            json={"username": "incomplete"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_delete_user_as_admin(self, client, admin_token):
        """Test deleting user as admin."""
        # First create a user to delete
        USER_STORE["deleteme"] = {
            "user_id": "deleteme",
            "username": "deleteme",
            "email": "deleteme@test.com",
            "password_hash": hash_password("password123!"),
            "permissions": ["read"],
            "roles": ["viewer"],
            "is_admin": False,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }

        response = client.delete(
            "/auth/users/deleteme", headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "deleted successfully" in data["message"]
        assert "deleteme" not in USER_STORE

    def test_delete_user_nonexistent(self, client, admin_token):
        """Test deleting non-existent user."""
        response = client.delete(
            "/auth/users/nonexistent",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in response.json()["detail"]

    def test_delete_user_self_prevention(self, client, admin_token):
        """Test that admin cannot delete their own account."""
        response = client.delete(
            "/auth/users/admin", headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot delete your own account" in response.json()["detail"]

    def test_delete_user_as_non_admin(self, client, operator_token):
        """Test deleting user as non-admin (should fail)."""
        response = client.delete(
            "/auth/users/viewer", headers={"Authorization": f"Bearer {operator_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_user_unauthorized(self, client):
        """Test deleting user without authentication."""
        response = client.delete("/auth/users/viewer")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestErrorHandling:
    """Test error handling in authentication endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(test_app)

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
        """Create mock JWT manager."""
        manager = JWTManager(jwt_config)
        return manager

    @pytest.fixture(autouse=True)
    def setup_dependencies(self, mock_jwt_manager):
        """Setup dependency overrides."""
        from src.integration.auth.dependencies import get_jwt_manager

        def get_test_jwt_manager():
            return mock_jwt_manager

        test_app.dependency_overrides[get_jwt_manager] = get_test_jwt_manager
        yield
        test_app.dependency_overrides.clear()

    def test_login_service_error(self, client):
        """Test login with service error."""
        # Mock JWT manager to raise exception
        with patch(
            "src.integration.auth.endpoints.get_user_by_username",
            side_effect=Exception("Database error"),
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

    def test_refresh_service_error(self, client):
        """Test refresh token with service error."""
        with patch(
            "src.integration.auth.jwt_manager.JWTManager.refresh_access_token",
            side_effect=Exception("Service error"),
        ):
            response = client.post(
                "/auth/refresh", json={"refresh_token": "some.token.here"}
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_logout_service_error(self, client, mock_jwt_manager):
        """Test logout with service error."""
        # Login first to get valid token
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Mock JWT manager to raise exception on revoke
        with patch.object(
            mock_jwt_manager, "revoke_token", side_effect=Exception("Revoke error")
        ):
            response = client.post(
                "/auth/logout",
                json={"refresh_token": "some.token.here"},
                headers={"Authorization": f"Bearer {access_token}"},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_password_change_service_error(self, client):
        """Test password change with service error."""
        # Login first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Mock USER_STORE access to raise exception
        with patch(
            "src.integration.auth.endpoints.USER_STORE",
            side_effect=Exception("Storage error"),
        ):
            response = client.post(
                "/auth/change-password",
                json={
                    "current_password": "admin123!",
                    "new_password": "new_password123!",
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_token_info_service_error(self, client):
        """Test token info with service error."""
        with patch(
            "src.integration.auth.jwt_manager.JWTManager.get_token_info",
            side_effect=Exception("Token service error"),
        ):
            response = client.post(
                "/auth/token/info", headers={"Authorization": "Bearer some.token.here"}
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_user_creation_service_error(self, client):
        """Test user creation with service error."""
        # Login as admin first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Mock secrets.token_hex to raise exception
        with patch(
            "src.integration.auth.endpoints.secrets.token_hex",
            side_effect=Exception("UUID error"),
        ):
            response = client.post(
                "/auth/users",
                json={
                    "username": "error_user",
                    "email": "error@test.com",
                    "password": "password123!",
                    "permissions": ["read"],
                    "roles": ["viewer"],
                    "is_admin": False,
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_user_deletion_service_error(self, client):
        """Test user deletion with service error."""
        # Login as admin first
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Create a user to delete
        USER_STORE["error_delete"] = {
            "user_id": "error_delete",
            "username": "error_delete",
            "email": "error@test.com",
            "password_hash": hash_password("password"),
            "permissions": ["read"],
            "roles": ["viewer"],
            "is_admin": False,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }

        # Mock USER_STORE.__delitem__ to raise exception
        with patch.dict("src.integration.auth.endpoints.USER_STORE", USER_STORE):
            with patch(
                "src.integration.auth.endpoints.USER_STORE.__delitem__",
                side_effect=Exception("Delete error"),
            ):
                response = client.delete(
                    "/auth/users/error_delete",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Clean up
        if "error_delete" in USER_STORE:
            del USER_STORE["error_delete"]


class TestSecurityFeatures:
    """Test security features of authentication system."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(test_app)

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
        """Create mock JWT manager."""
        manager = JWTManager(jwt_config)
        return manager

    @pytest.fixture(autouse=True)
    def setup_dependencies(self, mock_jwt_manager):
        """Setup dependency overrides."""
        from src.integration.auth.dependencies import get_jwt_manager

        def get_test_jwt_manager():
            return mock_jwt_manager

        test_app.dependency_overrides[get_jwt_manager] = get_test_jwt_manager
        yield
        test_app.dependency_overrides.clear()

    def test_password_not_returned_in_responses(self, client):
        """Test that passwords are never returned in API responses."""
        # Login
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        response_text = response.text.lower()
        assert "password" not in response_text
        assert "admin123!" not in response_text

        # Get user info
        access_token = response.json()["access_token"]
        user_response = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {access_token}"}
        )

        user_response_text = user_response.text.lower()
        assert "password" not in user_response_text

    def test_token_revocation_on_refresh(self, client):
        """Test that old refresh tokens are revoked when refreshed."""
        # Login to get refresh token
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        old_refresh_token = login_response.json()["refresh_token"]

        # Use refresh token
        refresh_response = client.post(
            "/auth/refresh", json={"refresh_token": old_refresh_token}
        )

        assert refresh_response.status_code == status.HTTP_200_OK

        # Try to use old refresh token again (should fail)
        second_refresh = client.post(
            "/auth/refresh", json={"refresh_token": old_refresh_token}
        )

        assert second_refresh.status_code == status.HTTP_401_UNAUTHORIZED

    def test_admin_privilege_enforcement(self, client):
        """Test that admin privileges are properly enforced."""
        # Login as non-admin
        login_response = client.post(
            "/auth/login",
            json={
                "username": "operator",
                "password": "operator123!",
                "remember_me": False,
            },
        )

        operator_token = login_response.json()["access_token"]

        # Try admin-only operations
        admin_endpoints = [
            ("GET", "/auth/users"),
            ("POST", "/auth/users"),
            ("DELETE", "/auth/users/viewer"),
        ]

        for method, endpoint in admin_endpoints:
            if method == "GET":
                response = client.get(
                    endpoint, headers={"Authorization": f"Bearer {operator_token}"}
                )
            elif method == "POST":
                response = client.post(
                    endpoint,
                    json={},
                    headers={"Authorization": f"Bearer {operator_token}"},
                )
            elif method == "DELETE":
                response = client.delete(
                    endpoint, headers={"Authorization": f"Bearer {operator_token}"}
                )

            assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_sql_injection_prevention(self, client):
        """Test that SQL injection attempts are handled safely."""
        # Attempt SQL injection in username
        malicious_usernames = [
            "admin'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "admin'; SELECT * FROM users; --",
        ]

        for username in malicious_usernames:
            response = client.post(
                "/auth/login",
                json={
                    "username": username,
                    "password": "admin123!",
                    "remember_me": False,
                },
            )

            # Should fail authentication, not cause server error
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_xss_prevention(self, client):
        """Test that XSS attempts are handled safely."""
        # Login as admin
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123!", "remember_me": False},
        )

        access_token = login_response.json()["access_token"]

        # Attempt XSS in user creation
        xss_payload = "<script>alert('XSS')</script>"
        response = client.post(
            "/auth/users",
            json={
                "username": xss_payload,
                "email": f"{xss_payload}@test.com",
                "password": "password123!",
                "permissions": ["read"],
                "roles": ["viewer"],
                "is_admin": False,
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Should either fail validation or escape the content
        if response.status_code == status.HTTP_200_OK:
            # If successful, ensure XSS payload is escaped/sanitized
            response_text = response.text
            assert "<script>" not in response_text or "&lt;script&gt;" in response_text

    def test_timing_attack_resistance(self, client):
        """Test resistance to timing attacks on login."""
        import time

        # Test with valid username, invalid password
        start_time = time.time()
        response1 = client.post(
            "/auth/login",
            json={
                "username": "admin",
                "password": "wrong_password",
                "remember_me": False,
            },
        )
        time1 = time.time() - start_time

        # Test with invalid username
        start_time = time.time()
        response2 = client.post(
            "/auth/login",
            json={
                "username": "nonexistent_user",
                "password": "some_password",
                "remember_me": False,
            },
        )
        time2 = time.time() - start_time

        # Both should fail with similar response times
        assert response1.status_code == status.HTTP_401_UNAUTHORIZED
        assert response2.status_code == status.HTTP_401_UNAUTHORIZED

        # Timing should be relatively similar (allowing for some variance)
        time_difference = abs(time1 - time2)
        assert time_difference < 0.1  # Less than 100ms difference
