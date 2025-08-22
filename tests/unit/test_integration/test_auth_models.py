"""
Comprehensive unit tests for authentication models.

This test suite validates Pydantic models for authentication, including user models,
request/response models, token models, and all validation logic.
"""

from datetime import datetime, timezone
import re

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


class TestAuthUser:
    """Test AuthUser model validation and functionality."""

    def test_auth_user_creation_minimal(self):
        """Test AuthUser creation with minimal required fields."""
        user = AuthUser(user_id="test_user_123", username="testuser")

        assert user.user_id == "test_user_123"
        assert user.username == "testuser"
        assert user.email is None
        assert user.permissions == []
        assert user.roles == []
        assert user.is_active is True
        assert user.is_admin is False
        assert user.last_login is None
        assert isinstance(user.created_at, datetime)

    def test_auth_user_creation_full(self):
        """Test AuthUser creation with all fields."""
        created_time = datetime.now(timezone.utc)
        last_login_time = datetime.now(timezone.utc)

        user = AuthUser(
            user_id="admin_123",
            username="admin",
            email="admin@example.com",
            permissions=["read", "write", "admin"],
            roles=["admin", "operator"],
            is_active=True,
            is_admin=True,
            last_login=last_login_time,
            created_at=created_time,
        )

        assert user.user_id == "admin_123"
        assert user.username == "admin"
        assert user.email == "admin@example.com"
        assert user.permissions == ["read", "write", "admin"]
        assert user.roles == ["admin", "operator"]
        assert user.is_admin is True
        assert user.last_login == last_login_time
        assert user.created_at == created_time

    def test_auth_user_invalid_permission(self):
        """Test AuthUser validation with invalid permission."""
        with pytest.raises(ValidationError, match="Invalid permission"):
            AuthUser(
                user_id="test_user",
                username="testuser",
                permissions=["invalid_permission"],
            )

    def test_auth_user_valid_permissions(self):
        """Test AuthUser with all valid permissions."""
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

        user = AuthUser(
            user_id="test_user", username="testuser", permissions=valid_permissions
        )

        assert user.permissions == valid_permissions

    def test_auth_user_invalid_role(self):
        """Test AuthUser validation with invalid role."""
        with pytest.raises(ValidationError, match="Invalid role"):
            AuthUser(user_id="test_user", username="testuser", roles=["invalid_role"])

    def test_auth_user_valid_roles(self):
        """Test AuthUser with all valid roles."""
        valid_roles = ["user", "admin", "operator", "viewer"]

        user = AuthUser(user_id="test_user", username="testuser", roles=valid_roles)

        assert user.roles == valid_roles

    def test_auth_user_has_permission(self):
        """Test AuthUser has_permission method."""
        user = AuthUser(
            user_id="test_user",
            username="testuser",
            permissions=["read", "write"],
            is_admin=False,
        )

        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("admin") is False

    def test_auth_user_has_permission_admin_override(self):
        """Test AuthUser has_permission with admin override."""
        user = AuthUser(
            user_id="admin_user", username="admin", permissions=["read"], is_admin=True
        )

        # Admin should have all permissions
        assert user.has_permission("read") is True
        assert user.has_permission("admin") is True
        assert user.has_permission("system_config") is True

    def test_auth_user_has_role(self):
        """Test AuthUser has_role method."""
        user = AuthUser(
            user_id="test_user", username="testuser", roles=["user", "viewer"]
        )

        assert user.has_role("user") is True
        assert user.has_role("viewer") is True
        assert user.has_role("admin") is False

    def test_auth_user_to_token_claims(self):
        """Test AuthUser to_token_claims method."""
        last_login_time = datetime.now(timezone.utc)

        user = AuthUser(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
            last_login=last_login_time,
        )

        claims = user.to_token_claims()

        assert claims["username"] == "testuser"
        assert claims["email"] == "test@example.com"
        assert claims["permissions"] == ["read", "write"]
        assert claims["roles"] == ["user"]
        assert claims["is_admin"] is False
        assert claims["last_login"] == last_login_time.isoformat()

    def test_auth_user_to_token_claims_no_login(self):
        """Test AuthUser to_token_claims with no last_login."""
        user = AuthUser(user_id="test_user", username="testuser", last_login=None)

        claims = user.to_token_claims()
        assert claims["last_login"] is None


class TestLoginRequest:
    """Test LoginRequest model validation."""

    def test_login_request_valid(self):
        """Test valid LoginRequest."""
        request = LoginRequest(
            username="testuser", password="TestPass123!", remember_me=True
        )

        assert request.username == "testuser"
        assert request.password == "TestPass123!"
        assert request.remember_me is True

    def test_login_request_username_normalization(self):
        """Test username is normalized to lowercase."""
        request = LoginRequest(username="TestUser", password="TestPass123!")

        assert request.username == "testuser"

    def test_login_request_invalid_username_short(self):
        """Test LoginRequest with too short username."""
        with pytest.raises(
            ValidationError, match="String should have at least 3 characters"
        ):
            LoginRequest(username="ab", password="TestPass123!")

    def test_login_request_invalid_username_long(self):
        """Test LoginRequest with too long username."""
        with pytest.raises(
            ValidationError, match="String should have at most 50 characters"
        ):
            LoginRequest(username="a" * 51, password="TestPass123!")

    def test_login_request_invalid_username_characters(self):
        """Test LoginRequest with invalid username characters."""
        with pytest.raises(ValidationError, match="Username can only contain"):
            LoginRequest(username="test@user", password="TestPass123!")  # @ not allowed

    def test_login_request_valid_username_characters(self):
        """Test LoginRequest with valid username characters."""
        valid_usernames = ["testuser", "test_user", "test-user", "test.user", "test123"]

        for username in valid_usernames:
            request = LoginRequest(username=username, password="TestPass123!")
            assert request.username.lower() == username.lower()

    def test_login_request_invalid_password_short(self):
        """Test LoginRequest with too short password."""
        with pytest.raises(
            ValidationError, match="Password must be at least 8 characters long"
        ):
            LoginRequest(username="testuser", password="short")

    def test_login_request_invalid_password_weak(self):
        """Test LoginRequest with weak password (insufficient complexity)."""
        weak_passwords = [
            "password",  # No uppercase, digit, or special
            "PASSWORD",  # No lowercase, digit, or special
            "12345678",  # No letters or special
            "Password",  # No digit or special
            "Password1",  # No special characters
        ]

        for password in weak_passwords:
            with pytest.raises(
                ValidationError, match="Password must contain at least 3 of"
            ):
                LoginRequest(username="testuser", password=password)

    def test_login_request_valid_password_complexity(self):
        """Test LoginRequest with valid password complexity."""
        valid_passwords = [
            "TestPass123!",  # All 4 types
            "TestPass123",  # 3 types: upper, lower, digit
            "TestPass!@#",  # 3 types: upper, lower, special
            "testpass123!",  # 3 types: lower, digit, special
        ]

        for password in valid_passwords:
            request = LoginRequest(username="testuser", password=password)
            assert request.password == password


class TestLoginResponse:
    """Test LoginResponse model."""

    def test_login_response_creation(self):
        """Test LoginResponse creation."""
        user = AuthUser(user_id="test", username="testuser")

        response = LoginResponse(
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            token_type="bearer",
            expires_in=3600,
            user=user,
        )

        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_123"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.user == user

    def test_login_response_schema_example(self):
        """Test LoginResponse schema example is valid."""
        user_data = {
            "user_id": "user123",
            "username": "admin",
            "email": "admin@example.com",
            "permissions": ["read", "write", "admin"],
            "roles": ["admin"],
            "is_active": True,
            "is_admin": True,
        }

        user = AuthUser(**user_data)

        response = LoginResponse(
            access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            token_type="bearer",
            expires_in=3600,
            user=user,
        )

        assert response.access_token.startswith("eyJhbGciOi")
        assert response.token_type == "bearer"


class TestRefreshRequest:
    """Test RefreshRequest model validation."""

    def test_refresh_request_valid(self):
        """Test valid RefreshRequest."""
        request = RefreshRequest(
            refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )

        assert request.refresh_token.startswith("eyJhbGciOi")

    def test_refresh_request_invalid_too_short(self):
        """Test RefreshRequest with too short token."""
        with pytest.raises(ValidationError, match="Invalid refresh token format"):
            RefreshRequest(refresh_token="short")

    def test_refresh_request_invalid_format(self):
        """Test RefreshRequest with invalid JWT format."""
        with pytest.raises(ValidationError, match="Invalid JWT token format"):
            RefreshRequest(refresh_token="invalid.token")  # Only 2 parts

    def test_refresh_request_valid_jwt_format(self):
        """Test RefreshRequest with valid JWT format."""
        # Valid JWT format (3 parts separated by dots)
        valid_token = "header.payload.signature"

        request = RefreshRequest(refresh_token=valid_token)
        assert request.refresh_token == valid_token


class TestRefreshResponse:
    """Test RefreshResponse model."""

    def test_refresh_response_creation(self):
        """Test RefreshResponse creation."""
        response = RefreshResponse(
            access_token="new_access_token",
            refresh_token="new_refresh_token",
            token_type="bearer",
            expires_in=3600,
        )

        assert response.access_token == "new_access_token"
        assert response.refresh_token == "new_refresh_token"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600


class TestLogoutRequest:
    """Test LogoutRequest model."""

    def test_logout_request_minimal(self):
        """Test LogoutRequest with minimal fields."""
        request = LogoutRequest()

        assert request.refresh_token is None
        assert request.revoke_all_tokens is False

    def test_logout_request_full(self):
        """Test LogoutRequest with all fields."""
        request = LogoutRequest(
            refresh_token="refresh_token_123", revoke_all_tokens=True
        )

        assert request.refresh_token == "refresh_token_123"
        assert request.revoke_all_tokens is True


class TestTokenInfo:
    """Test TokenInfo model."""

    def test_token_info_creation(self):
        """Test TokenInfo creation."""
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at.replace(hour=issued_at.hour + 1)

        token_info = TokenInfo(
            user_id="test_user",
            username="testuser",
            token_type="access",
            permissions=["read", "write"],
            issued_at=issued_at,
            expires_at=expires_at,
            is_expired=False,
            is_active=True,
            jti="token_id_123",
        )

        assert token_info.user_id == "test_user"
        assert token_info.username == "testuser"
        assert token_info.token_type == "access"
        assert token_info.permissions == ["read", "write"]
        assert token_info.issued_at == issued_at
        assert token_info.expires_at == expires_at
        assert token_info.is_expired is False
        assert token_info.is_active is True
        assert token_info.jti == "token_id_123"


class TestPasswordChangeRequest:
    """Test PasswordChangeRequest model validation."""

    def test_password_change_request_valid(self):
        """Test valid PasswordChangeRequest."""
        request = PasswordChangeRequest(
            current_password="OldPass123!",
            new_password="NewPass123!",
            confirm_password="NewPass123!",
        )

        assert request.current_password == "OldPass123!"
        assert request.new_password == "NewPass123!"
        assert request.confirm_password == "NewPass123!"

    def test_password_change_request_mismatch(self):
        """Test PasswordChangeRequest with mismatched passwords."""
        with pytest.raises(ValidationError, match="Passwords do not match"):
            PasswordChangeRequest(
                current_password="OldPass123!",
                new_password="NewPass123!",
                confirm_password="DifferentPass123!",
            )

    def test_password_change_request_weak_new_password(self):
        """Test PasswordChangeRequest with weak new password."""
        with pytest.raises(
            ValidationError, match="Password must contain at least 3 of"
        ):
            PasswordChangeRequest(
                current_password="OldPass123!",
                new_password="weakpass",
                confirm_password="weakpass",
            )

    def test_password_change_request_short_new_password(self):
        """Test PasswordChangeRequest with too short new password."""
        with pytest.raises(
            ValidationError, match="Password must be at least 8 characters long"
        ):
            PasswordChangeRequest(
                current_password="OldPass123!",
                new_password="Short1!",
                confirm_password="Short1!",
            )


class TestUserCreateRequest:
    """Test UserCreateRequest model validation."""

    def test_user_create_request_valid(self):
        """Test valid UserCreateRequest."""
        request = UserCreateRequest(
            username="newuser",
            email="newuser@example.com",
            password="NewPass123!",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
        )

        assert request.username == "newuser"
        assert request.email == "newuser@example.com"
        assert request.password == "NewPass123!"
        assert request.permissions == ["read", "write"]
        assert request.roles == ["user"]
        assert request.is_admin is False

    def test_user_create_request_email_normalization(self):
        """Test email is normalized to lowercase."""
        request = UserCreateRequest(
            username="newuser", email="NewUser@Example.Com", password="NewPass123!"
        )

        assert request.email == "newuser@example.com"

    def test_user_create_request_invalid_email(self):
        """Test UserCreateRequest with invalid email."""
        invalid_emails = [
            "invalid",
            "invalid@",
            "@example.com",
            "invalid@.com",
            "invalid@example",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError, match="Invalid email format"):
                UserCreateRequest(
                    username="newuser", email=email, password="NewPass123!"
                )

    def test_user_create_request_valid_emails(self):
        """Test UserCreateRequest with valid emails."""
        valid_emails = [
            "test@example.com",
            "test.user@example.co.uk",
            "test+tag@example-site.org",
            "123@example.com",
        ]

        for email in valid_emails:
            request = UserCreateRequest(
                username="newuser", email=email, password="NewPass123!"
            )
            assert request.email == email.lower()

    def test_user_create_request_username_normalization(self):
        """Test username is normalized to lowercase."""
        request = UserCreateRequest(
            username="NewUser", email="test@example.com", password="NewPass123!"
        )

        assert request.username == "newuser"

    def test_user_create_request_defaults(self):
        """Test UserCreateRequest default values."""
        request = UserCreateRequest(
            username="newuser", email="test@example.com", password="NewPass123!"
        )

        assert request.permissions == []
        assert request.roles == []
        assert request.is_admin is False


class TestAPIKey:
    """Test APIKey model functionality."""

    def test_api_key_creation(self):
        """Test APIKey creation."""
        created_time = datetime.now(timezone.utc)
        expires_time = created_time.replace(month=created_time.month + 1)

        api_key = APIKey(
            key_id="key_123",
            name="Test API Key",
            key_hash="hashed_key_value",
            permissions=["read", "prediction_view"],
            created_at=created_time,
            expires_at=expires_time,
            is_active=True,
            usage_count=50,
        )

        assert api_key.key_id == "key_123"
        assert api_key.name == "Test API Key"
        assert api_key.key_hash == "hashed_key_value"
        assert api_key.permissions == ["read", "prediction_view"]
        assert api_key.created_at == created_time
        assert api_key.expires_at == expires_time
        assert api_key.is_active is True
        assert api_key.usage_count == 50

    def test_api_key_defaults(self):
        """Test APIKey default values."""
        api_key = APIKey(key_id="key_123", name="Test Key", key_hash="hash")

        assert api_key.permissions == []
        assert api_key.last_used is None
        assert api_key.expires_at is None
        assert api_key.is_active is True
        assert api_key.usage_count == 0
        assert isinstance(api_key.created_at, datetime)

    def test_api_key_is_expired_no_expiration(self):
        """Test APIKey is_expired when no expiration is set."""
        api_key = APIKey(
            key_id="key_123", name="Test Key", key_hash="hash", expires_at=None
        )

        assert api_key.is_expired() is False

    def test_api_key_is_expired_not_expired(self):
        """Test APIKey is_expired when not expired."""
        future_time = datetime.now(timezone.utc).replace(day=datetime.now().day + 1)

        api_key = APIKey(
            key_id="key_123", name="Test Key", key_hash="hash", expires_at=future_time
        )

        assert api_key.is_expired() is False

    def test_api_key_is_expired_expired(self):
        """Test APIKey is_expired when expired."""
        past_time = datetime.now(timezone.utc).replace(day=datetime.now().day - 1)

        api_key = APIKey(
            key_id="key_123", name="Test Key", key_hash="hash", expires_at=past_time
        )

        assert api_key.is_expired() is True

    def test_api_key_has_permission(self):
        """Test APIKey has_permission method."""
        api_key = APIKey(
            key_id="key_123",
            name="Test Key",
            key_hash="hash",
            permissions=["read", "prediction_view"],
        )

        assert api_key.has_permission("read") is True
        assert api_key.has_permission("prediction_view") is True
        assert api_key.has_permission("admin") is False

    def test_api_key_has_permission_empty(self):
        """Test APIKey has_permission with no permissions."""
        api_key = APIKey(key_id="key_123", name="Test Key", key_hash="hash")

        assert api_key.has_permission("read") is False
        assert api_key.has_permission("admin") is False


class TestModelEdgeCases:
    """Test edge cases and error conditions."""

    def test_auth_user_empty_permissions_and_roles(self):
        """Test AuthUser with empty permissions and roles lists."""
        user = AuthUser(user_id="test", username="test", permissions=[], roles=[])

        assert user.permissions == []
        assert user.roles == []
        assert user.has_permission("read") is False
        assert user.has_role("user") is False

    def test_login_request_boundary_lengths(self):
        """Test LoginRequest with boundary length values."""
        # Minimum valid length
        min_request = LoginRequest(
            username="abc", password="Test123!"  # 3 chars (minimum)  # 8 chars, complex
        )
        assert min_request.username == "abc"

        # Maximum valid length
        max_request = LoginRequest(
            username="a" * 50,  # 50 chars (maximum)
            password="Test123!" + "a" * (128 - 8),  # 128 chars max
        )
        assert len(max_request.username) == 50
        assert len(max_request.password) == 128

    def test_password_complexity_edge_cases(self):
        """Test password complexity validation edge cases."""
        # Exactly 3 categories should pass
        passwords_3_categories = [
            "Abcdefgh",  # upper, lower, no digit, no special (should fail)
            "Abcdefg1",  # upper, lower, digit, no special (should pass)
            "Abcdefg!",  # upper, lower, special, no digit (should pass)
            "abcdefg1!",  # lower, digit, special, no upper (should pass)
        ]

        # Test the one that should fail
        with pytest.raises(
            ValidationError, match="Password must contain at least 3 of"
        ):
            LoginRequest(username="test", password="Abcdefgh")

        # Test the ones that should pass
        for password in passwords_3_categories[1:]:
            request = LoginRequest(username="test", password=password)
            assert request.password == password

    def test_token_info_with_minimal_data(self):
        """Test TokenInfo with minimal required data."""
        token_info = TokenInfo(
            user_id="test",
            token_type="access",
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            is_expired=False,
            is_active=True,
        )

        assert token_info.user_id == "test"
        assert token_info.username is None
        assert token_info.permissions == []
        assert token_info.jti is None

    def test_special_characters_in_usernames(self):
        """Test various special characters in usernames."""
        valid_chars = ["_", "-", "."]

        for char in valid_chars:
            username = f"test{char}user"
            request = LoginRequest(username=username, password="Test123!")
            assert request.username == username.lower()

    def test_email_validation_comprehensive(self):
        """Test comprehensive email validation scenarios."""
        # Valid complex email formats
        valid_emails = [
            "test@example.com",
            "test.email@example.com",
            "test+tag@example.com",
            "test-email@example-site.com",
            "test123@example123.com",
            "a@b.co",
            "very.long.email.address@very-long-domain-name.com",
        ]

        for email in valid_emails:
            request = UserCreateRequest(
                username="test", email=email, password="Test123!"
            )
            assert request.email == email.lower()

        # Invalid email formats
        invalid_emails = [
            "",
            "test",
            "test@",
            "@example.com",
            "test@.com",
            "test@example.",
            "test@example",
            "test.@example.com",
            ".test@example.com",
            "test..test@example.com",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError, match="Invalid email format"):
                UserCreateRequest(username="test", email=email, password="Test123!")
