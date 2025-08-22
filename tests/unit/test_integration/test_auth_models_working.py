"""
Working unit tests for authentication models module.

This test suite validates Pydantic models used for JWT authentication
with current code compatibility.
"""

from datetime import datetime, timedelta, timezone
import hashlib

from pydantic import ValidationError
import pytest

from src.integration.auth.auth_models import (
    APIKey,
    AuthUser,
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    PasswordChangeRequest,
    RefreshRequest,
    RefreshResponse,
    TokenInfo,
    UserCreateRequest,
)


class TestAuthUserBasic:
    """Test AuthUser model basic functionality."""

    def test_auth_user_creation_minimal(self):
        """Test creating AuthUser with minimal required fields."""
        user = AuthUser(user_id="test_user", username="testuser")

        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email is None
        assert user.permissions == []
        assert user.roles == []
        assert user.is_active is True
        assert user.is_admin is False
        assert user.last_login is None
        assert isinstance(user.created_at, datetime)

    def test_auth_user_creation_full(self):
        """Test creating AuthUser with all fields."""
        created_at = datetime.now(timezone.utc)
        last_login = datetime.now(timezone.utc)

        user = AuthUser(
            user_id="admin_user",
            username="admin",
            email="admin@example.com",
            permissions=["read", "write", "admin"],
            roles=["admin", "user"],
            is_active=True,
            is_admin=True,
            last_login=last_login,
            created_at=created_at,
        )

        assert user.user_id == "admin_user"
        assert user.username == "admin"
        assert user.email == "admin@example.com"
        assert user.permissions == ["read", "write", "admin"]
        assert user.roles == ["admin", "user"]
        assert user.is_active is True
        assert user.is_admin is True
        assert user.last_login == last_login
        assert user.created_at == created_at

    def test_auth_user_permission_validation(self):
        """Test validation of permission strings."""
        valid_permissions = [
            "read",
            "write",
            "admin",
            "model_retrain",
            "system_config",
            "prediction_view",
            "accuracy_view",
            "health_check",
        ]

        # Valid permissions should work
        user = AuthUser(user_id="test", username="test", permissions=valid_permissions)
        assert user.permissions == valid_permissions

        # Invalid permission should raise error
        with pytest.raises(ValidationError):
            AuthUser(
                user_id="test", username="test", permissions=["invalid_permission"]
            )

    def test_auth_user_role_validation(self):
        """Test validation of role strings."""
        valid_roles = ["user", "admin", "operator", "viewer"]

        # Valid roles should work
        user = AuthUser(user_id="test", username="test", roles=valid_roles)
        assert user.roles == valid_roles

        # Invalid role should raise error
        with pytest.raises(ValidationError):
            AuthUser(user_id="test", username="test", roles=["invalid_role"])

    def test_auth_user_has_permission_method(self):
        """Test has_permission method logic."""
        user = AuthUser(
            user_id="test",
            username="test",
            permissions=["read", "write"],
            is_admin=False,
        )

        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("admin") is False

        # Admin user should have all permissions
        admin_user = AuthUser(
            user_id="admin", username="admin", permissions=["read"], is_admin=True
        )
        assert admin_user.has_permission("admin") is True
        assert admin_user.has_permission("system_config") is True

    def test_auth_user_has_role_method(self):
        """Test has_role method logic."""
        user = AuthUser(user_id="test", username="test", roles=["user", "operator"])

        assert user.has_role("user") is True
        assert user.has_role("operator") is True
        assert user.has_role("admin") is False

    def test_auth_user_to_token_claims(self):
        """Test token claims generation."""
        last_login = datetime.now(timezone.utc)
        user = AuthUser(
            user_id="test",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
            last_login=last_login,
        )

        claims = user.to_token_claims()

        expected_claims = {
            "username": "testuser",
            "email": "test@example.com",
            "permissions": ["read", "write"],
            "roles": ["user"],
            "is_admin": False,
            "last_login": last_login.isoformat(),
        }

        assert claims == expected_claims


class TestLoginRequestBasic:
    """Test LoginRequest model basic functionality."""

    def test_login_request_valid(self):
        """Test valid login request."""
        request = LoginRequest(
            username="testuser", password="Password123!", remember_me=True
        )

        assert request.username == "testuser"  # Should be normalized to lowercase
        assert request.password == "Password123!"
        assert request.remember_me is True

    def test_login_request_username_validation(self):
        """Test username format validation."""
        # Valid usernames
        valid_usernames = ["user123", "test_user", "user.name", "user-name"]
        for username in valid_usernames:
            request = LoginRequest(username=username, password="Password123!")
            assert request.username == username.lower()  # Should be normalized

        # Invalid username
        with pytest.raises(ValidationError):
            LoginRequest(username="user@name", password="Password123!")

    def test_login_request_password_validation(self):
        """Test password strength validation."""
        # Valid passwords
        valid_passwords = [
            "Password123!",
            "MyP@ssw0rd",
            "StrongP@ss1",
        ]

        for password in valid_passwords:
            request = LoginRequest(username="testuser", password=password)
            assert request.password == password

        # Too short
        with pytest.raises(ValidationError):
            LoginRequest(username="testuser", password="Short1!")

        # Not complex enough
        with pytest.raises(ValidationError):
            LoginRequest(username="testuser", password="password")  # Too simple

    def test_login_request_defaults(self):
        """Test default values."""
        request = LoginRequest(username="testuser", password="Password123!")
        assert request.remember_me is False


class TestLoginResponseBasic:
    """Test LoginResponse model basic functionality."""

    def test_login_response_creation(self):
        """Test LoginResponse creation."""
        user = AuthUser(user_id="test", username="testuser")

        response = LoginResponse(
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            expires_in=3600,
            user=user,
        )

        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_123"
        assert response.token_type == "bearer"  # Default value
        assert response.expires_in == 3600
        assert response.user == user


class TestRefreshRequestBasic:
    """Test RefreshRequest model basic functionality."""

    def test_refresh_request_valid(self):
        """Test valid refresh request."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        request = RefreshRequest(refresh_token=token)
        assert request.refresh_token == token

    def test_refresh_request_invalid_format(self):
        """Test invalid refresh token format."""
        # Too short
        with pytest.raises(ValidationError):
            RefreshRequest(refresh_token="short")

        # Invalid JWT format (not 3 parts)
        with pytest.raises(ValidationError):
            RefreshRequest(refresh_token="invalid.token")


class TestTokenInfoBasic:
    """Test TokenInfo model basic functionality."""

    def test_token_info_creation(self):
        """Test TokenInfo creation."""
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(hours=1)

        token_info = TokenInfo(
            user_id="test_user",
            username="testuser",
            token_type="access",
            permissions=["read", "write"],
            issued_at=issued_at,
            expires_at=expires_at,
            is_expired=False,
            is_active=True,
            jti="unique_token_id",
        )

        assert token_info.user_id == "test_user"
        assert token_info.username == "testuser"
        assert token_info.token_type == "access"
        assert token_info.permissions == ["read", "write"]
        assert token_info.issued_at == issued_at
        assert token_info.expires_at == expires_at
        assert token_info.is_expired is False
        assert token_info.is_active is True
        assert token_info.jti == "unique_token_id"


class TestPasswordChangeRequestBasic:
    """Test PasswordChangeRequest model basic functionality."""

    def test_password_change_request_valid(self):
        """Test valid password change request."""
        request = PasswordChangeRequest(
            current_password="OldPassword123!",
            new_password="NewPassword123!",
            confirm_password="NewPassword123!",
        )

        assert request.current_password == "OldPassword123!"
        assert request.new_password == "NewPassword123!"
        assert request.confirm_password == "NewPassword123!"

    def test_password_change_request_weak_new_password(self):
        """Test weak new password validation."""
        with pytest.raises(ValidationError):
            PasswordChangeRequest(
                current_password="OldPassword123!",
                new_password="weak",  # Too short and simple
                confirm_password="weak",
            )


class TestUserCreateRequestBasic:
    """Test UserCreateRequest model basic functionality."""

    def test_user_create_request_valid(self):
        """Test valid user creation request."""
        request = UserCreateRequest(
            username="newuser",
            email="newuser@example.com",
            password="Password123!",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
        )

        assert request.username == "newuser"
        assert request.email == "newuser@example.com"
        assert request.password == "Password123!"
        assert request.permissions == ["read", "write"]
        assert request.roles == ["user"]
        assert request.is_admin is False

    def test_user_create_request_email_validation(self):
        """Test email format validation."""
        # Valid email
        request = UserCreateRequest(
            username="testuser", email="user@example.com", password="Password123!"
        )
        assert request.email == "user@example.com"

        # Invalid email
        with pytest.raises(ValidationError):
            UserCreateRequest(
                username="testuser", email="invalid-email", password="Password123!"
            )


class TestAPIKeyBasic:
    """Test APIKey model basic functionality."""

    def test_api_key_creation(self):
        """Test APIKey creation."""
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(days=30)

        api_key = APIKey(
            key_id="key_123",
            name="Test API Key",
            key_hash="hashed_key_value",
            permissions=["read", "write"],
            created_at=created_at,
            expires_at=expires_at,
            is_active=True,
            usage_count=0,
        )

        assert api_key.key_id == "key_123"
        assert api_key.name == "Test API Key"
        assert api_key.key_hash == "hashed_key_value"
        assert api_key.permissions == ["read", "write"]
        assert api_key.created_at == created_at
        assert api_key.expires_at == expires_at
        assert api_key.is_active is True
        assert api_key.usage_count == 0

    def test_api_key_is_expired_method(self):
        """Test is_expired method."""
        # Non-expiring key
        api_key = APIKey(key_id="key1", name="Test", key_hash="hash")
        assert api_key.is_expired() is False

        # Expired key
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        expired_key = APIKey(
            key_id="key2", name="Expired", key_hash="hash", expires_at=past_date
        )
        assert expired_key.is_expired() is True

    def test_api_key_has_permission_method(self):
        """Test has_permission method."""
        api_key = APIKey(
            key_id="key1",
            name="Test",
            key_hash="hash",
            permissions=["read", "health_check"],
        )

        assert api_key.has_permission("read") is True
        assert api_key.has_permission("health_check") is True
        assert api_key.has_permission("admin") is False


class TestModelsIntegration:
    """Test basic integration scenarios between auth models."""

    def test_complete_login_flow_models(self):
        """Test models work together in login flow."""
        # Login request
        login_req = LoginRequest(
            username="testuser", password="Password123!", remember_me=True
        )

        # Create user
        user = AuthUser(
            user_id="user_123",
            username=login_req.username,
            permissions=["read", "write"],
            roles=["user"],
        )

        # Login response
        login_resp = LoginResponse(
            access_token="access_token",
            refresh_token="refresh_token",
            expires_in=3600,
            user=user,
        )

        # Verify flow
        assert login_resp.user.username == login_req.username
        assert login_resp.user.has_permission("read")

    def test_token_refresh_flow_models(self):
        """Test models work together in token refresh flow."""
        # Refresh request
        refresh_req = RefreshRequest(
            refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dosomething"
        )

        # Refresh response
        refresh_resp = RefreshResponse(
            access_token="new_access_token",
            refresh_token="new_refresh_token",
            expires_in=3600,
        )

        assert refresh_req.refresh_token != refresh_resp.refresh_token
        assert refresh_resp.token_type == "bearer"

    def test_model_serialization_deserialization(self):
        """Test that models can be serialized and deserialized properly."""
        user = AuthUser(
            user_id="test",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
        )

        # Serialize to dict
        user_dict = user.model_dump()

        # Deserialize back
        user_restored = AuthUser(**user_dict)

        assert user_restored.user_id == user.user_id
        assert user_restored.username == user.username
        assert user_restored.permissions == user.permissions
