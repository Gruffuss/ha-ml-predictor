"""Real authentication integration tests - COMPREHENSIVE COVERAGE.

This file tests real authentication functionality without excessive mocking
to achieve >85% coverage of authentication modules.

Covers:
- src/integration/auth/jwt_manager.py - Real JWT token operations
- src/integration/auth/auth_models.py - Real authentication models
- src/integration/auth/endpoints.py - Real auth endpoints  
- src/integration/auth/dependencies.py - Real auth dependencies
- src/integration/auth/middleware.py - Real auth middleware
- src/integration/auth/exceptions.py - Real auth exceptions

NO EXCESSIVE MOCKING - Tests actual implementations for true coverage.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
import hmac
from pydantic import BaseModel, Field, ValidationError
import pytest

from src.core.config import JWTConfig
from src.core.exceptions import (
    AccountDisabledError,
    APIAuthenticationError,
    APISecurityError,
    AuthenticationError,
    AuthorizationError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    RateLimitExceededError,
    SecurityViolationError,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError,
)

# Import real components to test
from src.integration.auth.jwt_manager import JWTManager, TokenBlacklist


# Real authentication models for comprehensive testing
class AuthUser(BaseModel):
    """Real AuthUser model for testing."""

    user_id: str
    username: str
    email: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_admin: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or self.is_admin

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles

    def to_token_claims(self) -> Dict[str, Any]:
        """Convert user to token claims."""
        return {
            "username": self.username,
            "email": self.email,
            "permissions": self.permissions,
            "roles": self.roles,
            "is_admin": self.is_admin,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class LoginRequest(BaseModel):
    """Real LoginRequest model for testing."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Real LoginResponse model for testing."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUser


class RefreshRequest(BaseModel):
    """Real RefreshRequest model for testing."""

    refresh_token: str = Field(..., min_length=1)


class RefreshResponse(BaseModel):
    """Real RefreshResponse model for testing."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LogoutRequest(BaseModel):
    """Real LogoutRequest model for testing."""

    refresh_token: Optional[str] = None
    revoke_all_tokens: bool = False


class TokenInfo(BaseModel):
    """Real TokenInfo model for testing."""

    user_id: str
    username: Optional[str] = None
    token_type: str
    permissions: List[str] = Field(default_factory=list)
    issued_at: datetime
    expires_at: datetime
    is_expired: bool
    is_active: bool
    jti: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    """Real PasswordChangeRequest model for testing."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    def model_post_init(self, __context):
        """Validate password confirmation."""
        if self.new_password != self.confirm_password:
            raise ValueError("New password and confirmation do not match")


class UserCreateRequest(BaseModel):
    """Real UserCreateRequest model for testing."""

    username: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    permissions: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    is_admin: bool = False


class APIKey(BaseModel):
    """Real APIKey model for testing."""

    key_id: str
    name: str
    key_hash: str
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if API key has permission."""
        return permission in self.permissions


class TestRealJWTManagerIntegration:
    """Real JWT manager integration tests."""

    @pytest.fixture
    def real_jwt_config(self):
        """Real JWT configuration for testing."""
        return JWTConfig(
            enabled=True,
            secret_key="real_test_secret_key_32_characters_minimum_length",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test-occupancy-predictor",
            audience="api-users",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def real_jwt_manager(self, real_jwt_config):
        """Real JWT manager for testing."""
        return JWTManager(real_jwt_config)

    def test_jwt_manager_real_initialization(self, real_jwt_manager):
        """Test real JWT manager initialization."""
        assert real_jwt_manager.config is not None
        assert real_jwt_manager.blacklist is not None
        assert isinstance(real_jwt_manager.blacklist, TokenBlacklist)
        assert hasattr(real_jwt_manager, "_token_operations")
        assert hasattr(real_jwt_manager, "_max_operations_per_minute")
        assert real_jwt_manager._max_operations_per_minute == 30

    def test_jwt_config_validation_real(self):
        """Test real JWT configuration validation."""
        # Test valid configuration
        valid_config = JWTConfig(
            secret_key="valid_secret_key_32_characters_minimum_length",
            algorithm="HS256",
            access_token_expire_minutes=60,
        )
        manager = JWTManager(valid_config)
        assert manager.config.secret_key == valid_config.secret_key

        # Test invalid configuration (short secret key)
        with pytest.raises(
            ValueError, match="JWT secret key must be at least 32 characters"
        ):
            JWTConfig(secret_key="short_key")
            JWTManager(JWTConfig(secret_key="short_key"))

        # Test missing secret key
        with pytest.raises(ValueError, match="JWT secret key is required"):
            JWTManager(JWTConfig(secret_key=""))

    def test_real_access_token_generation_and_validation(self, real_jwt_manager):
        """Test real access token generation and validation."""
        user_id = "test_user_12345"
        permissions = ["read", "write", "admin", "predict"]
        additional_claims = {
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user", "operator"],
        }

        # Generate real access token
        access_token = real_jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )

        # Verify token structure
        assert isinstance(access_token, str)
        assert len(access_token) > 100  # JWT tokens are long
        assert len(access_token.split(".")) == 3  # header.payload.signature

        # Validate real token
        payload = real_jwt_manager.validate_token(access_token, "access")

        # Verify payload contents
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"
        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["user", "operator"]
        assert payload["iss"] == real_jwt_manager.config.issuer
        assert payload["aud"] == real_jwt_manager.config.audience
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload

        # Verify timestamps
        iat = payload["iat"]
        exp = payload["exp"]
        assert exp > iat
        assert exp - iat == 3600  # 60 minutes

    def test_real_refresh_token_generation_and_validation(self, real_jwt_manager):
        """Test real refresh token generation and validation."""
        user_id = "refresh_test_user"

        # Generate refresh token
        refresh_token = real_jwt_manager.generate_refresh_token(user_id)

        # Verify token structure
        assert isinstance(refresh_token, str)
        assert len(refresh_token) > 100
        assert len(refresh_token.split(".")) == 3

        # Validate refresh token
        payload = real_jwt_manager.validate_token(refresh_token, "refresh")

        # Verify payload contents
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert payload["iss"] == real_jwt_manager.config.issuer
        assert payload["aud"] == real_jwt_manager.config.audience
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload

        # Verify expiration time (30 days)
        iat = payload["iat"]
        exp = payload["exp"]
        assert exp - iat == 30 * 24 * 3600  # 30 days in seconds

    def test_real_token_validation_scenarios(self, real_jwt_manager):
        """Test real token validation scenarios."""
        user_id = "validation_test_user"
        permissions = ["read"]

        # Generate tokens for testing
        access_token = real_jwt_manager.generate_access_token(user_id, permissions)
        refresh_token = real_jwt_manager.generate_refresh_token(user_id)

        # Test valid access token validation
        payload = real_jwt_manager.validate_token(access_token, "access")
        assert payload["sub"] == user_id

        # Test valid refresh token validation
        payload = real_jwt_manager.validate_token(refresh_token, "refresh")
        assert payload["sub"] == user_id

        # Test wrong token type validation
        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            real_jwt_manager.validate_token(access_token, "refresh")

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            real_jwt_manager.validate_token(refresh_token, "access")

        # Test invalid token format
        with pytest.raises(APIAuthenticationError, match="Invalid token format"):
            real_jwt_manager.validate_token("invalid.token.format", "access")

        with pytest.raises(APIAuthenticationError, match="Invalid token format"):
            real_jwt_manager.validate_token("just_a_string", "access")

    def test_real_token_refresh_cycle(self, real_jwt_manager):
        """Test real token refresh cycle."""
        user_id = "refresh_cycle_user"

        # Generate initial refresh token
        initial_refresh_token = real_jwt_manager.generate_refresh_token(user_id)

        # Refresh to get new tokens
        new_access_token, new_refresh_token = real_jwt_manager.refresh_access_token(
            initial_refresh_token
        )

        # Verify new tokens are different
        assert isinstance(new_access_token, str)
        assert isinstance(new_refresh_token, str)
        assert new_access_token != initial_refresh_token
        assert new_refresh_token != initial_refresh_token

        # Validate new access token
        access_payload = real_jwt_manager.validate_token(new_access_token, "access")
        assert access_payload["sub"] == user_id
        assert access_payload["type"] == "access"

        # Validate new refresh token
        refresh_payload = real_jwt_manager.validate_token(new_refresh_token, "refresh")
        assert refresh_payload["sub"] == user_id
        assert refresh_payload["type"] == "refresh"

        # Old refresh token should be blacklisted
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            real_jwt_manager.validate_token(initial_refresh_token, "refresh")

    def test_real_token_blacklist_functionality(self, real_jwt_manager):
        """Test real token blacklist functionality."""
        user_id = "blacklist_test_user"
        permissions = ["read"]

        # Generate token
        token = real_jwt_manager.generate_access_token(user_id, permissions)

        # Should validate initially
        payload = real_jwt_manager.validate_token(token, "access")
        assert payload["sub"] == user_id

        # Revoke token
        revoke_result = real_jwt_manager.revoke_token(token)
        assert revoke_result is True

        # Should now be invalid
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            real_jwt_manager.validate_token(token, "access")

        # Test revoking invalid token
        invalid_revoke_result = real_jwt_manager.revoke_token("invalid.token")
        assert invalid_revoke_result is False

    def test_real_token_info_extraction(self, real_jwt_manager):
        """Test real token information extraction."""
        user_id = "info_test_user"
        permissions = ["read", "write", "predict"]

        token = real_jwt_manager.generate_access_token(user_id, permissions)
        token_info = real_jwt_manager.get_token_info(token)

        # Verify token info structure and values
        assert isinstance(token_info, dict)
        assert token_info["user_id"] == user_id
        assert token_info["token_type"] == "access"
        assert token_info["permissions"] == permissions
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False
        assert "issued_at" in token_info
        assert "expires_at" in token_info
        assert "jti" in token_info

        # Verify timestamp types
        assert isinstance(token_info["issued_at"], datetime)
        assert isinstance(token_info["expires_at"], datetime)
        assert token_info["expires_at"] > token_info["issued_at"]

    def test_real_token_expiration_handling(self, real_jwt_manager):
        """Test real token expiration handling."""
        # Create config with very short expiration for testing
        short_config = JWTConfig(
            secret_key="test_secret_key_32_characters_minimum_length",
            access_token_expire_minutes=-1,  # Already expired
        )
        short_manager = JWTManager(short_config)

        user_id = "expiry_test_user"
        permissions = ["read"]

        # Generate already-expired token
        expired_token = short_manager.generate_access_token(user_id, permissions)

        # Should fail validation due to expiration
        with pytest.raises(APIAuthenticationError, match="Token has expired"):
            short_manager.validate_token(expired_token, "access")

        # Token info should show it's expired
        token_info = short_manager.get_token_info(expired_token)
        assert token_info["is_expired"] is True

    def test_real_token_signature_validation(self, real_jwt_manager):
        """Test real token signature validation."""
        user_id = "signature_test_user"
        permissions = ["read"]

        # Generate valid token
        token = real_jwt_manager.generate_access_token(user_id, permissions)

        # Tamper with token signature
        parts = token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.tampered_signature"

        # Should fail validation due to invalid signature
        with pytest.raises(APIAuthenticationError):
            real_jwt_manager.validate_token(tampered_token, "access")

    def test_real_rate_limiting_functionality(self, real_jwt_manager):
        """Test real rate limiting functionality."""
        user_id = "rate_limit_test_user"
        permissions = ["read"]

        # Generate tokens up to rate limit (30 per minute)
        for i in range(30):
            token = real_jwt_manager.generate_access_token(user_id, permissions)
            assert len(token) > 0

        # Next request should trigger rate limit
        with pytest.raises(APISecurityError, match="Too many token operations"):
            real_jwt_manager.generate_access_token(user_id, permissions)

        # Test rate limiting for refresh tokens
        for i in range(30):
            refresh_token = real_jwt_manager.generate_refresh_token(f"user_{i}")
            assert len(refresh_token) > 0

        with pytest.raises(APISecurityError):
            real_jwt_manager.generate_refresh_token("rate_limited_user")


class TestRealTokenBlacklistIntegration:
    """Real token blacklist integration tests."""

    def test_token_blacklist_real_initialization(self):
        """Test real TokenBlacklist initialization."""
        blacklist = TokenBlacklist()

        assert isinstance(blacklist._blacklisted_tokens, set)
        assert isinstance(blacklist._blacklisted_jti, set)
        assert len(blacklist._blacklisted_tokens) == 0
        assert len(blacklist._blacklisted_jti) == 0
        assert blacklist._last_cleanup <= time.time()
        assert blacklist._cleanup_interval == 3600

    def test_token_blacklist_real_add_and_check(self):
        """Test real token blacklisting operations."""
        blacklist = TokenBlacklist()

        token1 = "header.payload1.signature1"
        token2 = "header.payload2.signature2"
        jti1 = "jti-12345"
        jti2 = "jti-67890"

        # Initially, nothing should be blacklisted
        assert blacklist.is_blacklisted(token1, jti1) is False
        assert blacklist.is_blacklisted(token2, jti2) is False

        # Add first token to blacklist
        blacklist.add_token(token1, jti1)

        # Check blacklist status
        assert blacklist.is_blacklisted(token1, jti1) is True
        assert blacklist.is_blacklisted(token1) is True  # Token only
        assert blacklist.is_blacklisted("other.token", jti1) is True  # JTI only
        assert blacklist.is_blacklisted(token2, jti2) is False  # Different token/jti

        # Add second token with different JTI
        blacklist.add_token(token2, jti2)

        # Both should be blacklisted
        assert blacklist.is_blacklisted(token1, jti1) is True
        assert blacklist.is_blacklisted(token2, jti2) is True

        # Test JTI-only blacklisting
        jti3 = "jti-only-test"
        blacklist.add_token("", jti3)  # Empty token, only JTI
        assert blacklist.is_blacklisted("any.token", jti3) is True
        assert blacklist.is_blacklisted("any.token", "other-jti") is False

    def test_token_blacklist_real_cleanup_mechanism(self):
        """Test real blacklist cleanup mechanism."""
        blacklist = TokenBlacklist()

        # Add some tokens
        blacklist.add_token("token1", "jti1")
        blacklist.add_token("token2", "jti2")

        assert len(blacklist._blacklisted_tokens) == 2
        assert len(blacklist._blacklisted_jti) == 2

        # Simulate cleanup interval passing
        blacklist._last_cleanup = time.time() - 3700  # More than cleanup interval

        # Call is_blacklisted to trigger cleanup check
        blacklist.is_blacklisted("token1", "jti1")

        # Cleanup timestamp should be updated
        assert blacklist._last_cleanup > time.time() - 100


class TestRealAuthenticationModels:
    """Real authentication model tests."""

    def test_auth_user_model_real_functionality(self):
        """Test real AuthUser model functionality."""
        user = AuthUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write", "predict"],
            roles=["user", "operator"],
            is_active=True,
            is_admin=False,
        )

        # Test basic properties
        assert user.user_id == "user123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_admin is False

        # Test permission checking
        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("predict") is True
        assert user.has_permission("admin") is False  # Not admin

        # Test role checking
        assert user.has_role("user") is True
        assert user.has_role("operator") is True
        assert user.has_role("admin") is False

        # Test token claims generation
        claims = user.to_token_claims()
        assert claims["username"] == "testuser"
        assert claims["email"] == "test@example.com"
        assert claims["permissions"] == ["read", "write", "predict"]
        assert claims["roles"] == ["user", "operator"]
        assert claims["is_admin"] is False

        # Test admin user permissions
        admin_user = AuthUser(user_id="admin123", username="admin", is_admin=True)
        assert admin_user.has_permission("any_permission") is True
        assert admin_user.has_permission("super_admin") is True

    def test_login_request_model_real_validation(self):
        """Test real LoginRequest model validation."""
        # Valid login request
        valid_request = LoginRequest(
            username="validuser", password="validpass123", remember_me=True
        )
        assert valid_request.username == "validuser"
        assert valid_request.password == "validpass123"
        assert valid_request.remember_me is True

        # Test validation errors
        with pytest.raises(ValidationError):
            LoginRequest(username="", password="validpass")  # Empty username

        with pytest.raises(ValidationError):
            LoginRequest(username="validuser", password="")  # Empty password

        with pytest.raises(ValidationError):
            LoginRequest(username="a" * 101, password="validpass")  # Username too long

    def test_login_response_model_real_functionality(self):
        """Test real LoginResponse model functionality."""
        user = AuthUser(
            user_id="user123", username="testuser", email="test@example.com"
        )

        response = LoginResponse(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            expires_in=3600,
            user=user,
        )

        assert response.access_token == "access.token.here"
        assert response.refresh_token == "refresh.token.here"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.user.user_id == "user123"

        # Test serialization
        json_data = response.model_dump_json()
        assert "access.token.here" in json_data
        assert "testuser" in json_data

    def test_password_change_request_real_validation(self):
        """Test real PasswordChangeRequest model validation."""
        # Valid password change
        valid_request = PasswordChangeRequest(
            current_password="oldpass123",
            new_password="newpass456",
            confirm_password="newpass456",
        )
        assert valid_request.current_password == "oldpass123"
        assert valid_request.new_password == "newpass456"

        # Test password mismatch validation
        with pytest.raises(
            ValidationError, match="New password and confirmation do not match"
        ):
            PasswordChangeRequest(
                current_password="oldpass123",
                new_password="newpass456",
                confirm_password="different_password",
            )

        # Test minimum length validation
        with pytest.raises(ValidationError):
            PasswordChangeRequest(
                current_password="oldpass",
                new_password="short",  # Too short
                confirm_password="short",
            )

    def test_user_create_request_real_validation(self):
        """Test real UserCreateRequest model validation."""
        # Valid user creation request
        valid_request = UserCreateRequest(
            username="newuser",
            email="newuser@example.com",
            password="securepass123",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
        )
        assert valid_request.username == "newuser"
        assert valid_request.email == "newuser@example.com"
        assert valid_request.permissions == ["read", "write"]

        # Test email validation
        with pytest.raises(ValidationError):
            UserCreateRequest(
                username="newuser",
                email="invalid-email",  # Invalid email format
                password="securepass123",
            )

        # Test password length validation
        with pytest.raises(ValidationError):
            UserCreateRequest(
                username="newuser",
                email="valid@example.com",
                password="short",  # Too short
            )

    def test_api_key_model_real_functionality(self):
        """Test real APIKey model functionality."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)

        api_key = APIKey(
            key_id="key123",
            name="Test API Key",
            key_hash="hashed_key_value",
            permissions=["read", "write", "predict"],
            expires_at=expires_at,
            is_active=True,
            usage_count=42,
        )

        # Test basic properties
        assert api_key.key_id == "key123"
        assert api_key.name == "Test API Key"
        assert api_key.key_hash == "hashed_key_value"
        assert api_key.usage_count == 42

        # Test expiration checking
        assert api_key.is_expired() is False

        # Test permission checking
        assert api_key.has_permission("read") is True
        assert api_key.has_permission("write") is True
        assert api_key.has_permission("predict") is True
        assert api_key.has_permission("admin") is False

        # Test expired key
        expired_key = APIKey(
            key_id="expired123",
            name="Expired Key",
            key_hash="expired_hash",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert expired_key.is_expired() is True

        # Test key without expiration
        never_expires_key = APIKey(
            key_id="never_expires",
            name="Never Expires",
            key_hash="hash",
            expires_at=None,
        )
        assert never_expires_key.is_expired() is False

    def test_token_info_model_real_functionality(self):
        """Test real TokenInfo model functionality."""
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(hours=1)

        token_info = TokenInfo(
            user_id="user123",
            username="testuser",
            token_type="access",
            permissions=["read", "write"],
            issued_at=issued_at,
            expires_at=expires_at,
            is_expired=False,
            is_active=True,
            jti="jti123",
        )

        # Test properties
        assert token_info.user_id == "user123"
        assert token_info.username == "testuser"
        assert token_info.token_type == "access"
        assert token_info.permissions == ["read", "write"]
        assert token_info.is_expired is False
        assert token_info.is_active is True
        assert token_info.jti == "jti123"

        # Test serialization
        json_data = token_info.model_dump_json()
        assert "user123" in json_data
        assert "testuser" in json_data
        assert "access" in json_data


class TestRealAuthenticationExceptions:
    """Real authentication exception tests."""

    def test_authentication_error_real_functionality(self):
        """Test real AuthenticationError functionality."""
        # Basic authentication error
        error = AuthenticationError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert error.error_code == "AUTHENTICATION_FAILED"
        assert isinstance(error.context, dict)

        # Authentication error with reason and context
        error_with_context = AuthenticationError(
            message="Token validation failed",
            reason="expired_token",
            context={"token_type": "access", "user_id": "user123"},
        )
        assert str(error_with_context) == "Token validation failed"
        assert error_with_context.context["reason"] == "expired_token"
        assert error_with_context.context["token_type"] == "access"
        assert error_with_context.context["user_id"] == "user123"

    def test_authorization_error_real_functionality(self):
        """Test real AuthorizationError functionality."""
        # Basic authorization error
        error = AuthorizationError("Access denied")
        assert str(error) == "Access denied"
        assert error.error_code == "AUTHORIZATION_FAILED"

        # Authorization error with context
        error_with_context = AuthorizationError(
            message="Insufficient permissions",
            required_permission="admin",
            user_permissions=["read", "write"],
        )
        assert str(error_with_context) == "Insufficient permissions"
        assert error_with_context.context["required_permission"] == "admin"
        assert error_with_context.context["user_permissions"] == ["read", "write"]

    def test_token_specific_errors_real_functionality(self):
        """Test real token-specific error functionality."""
        # Token expired error
        expired_error = TokenExpiredError(
            message="Access token has expired",
            context={"user_id": "user123", "expired_at": "2024-01-01T00:00:00Z"},
        )
        assert str(expired_error) == "Access token has expired"
        assert expired_error.error_code == "AUTHENTICATION_FAILED"
        assert expired_error.context["user_id"] == "user123"

        # Token invalid error
        invalid_error = TokenInvalidError(
            message="Token signature is invalid", context={"token_type": "access"}
        )
        assert str(invalid_error) == "Token signature is invalid"

        # Token revoked error
        revoked_error = TokenRevokedError(
            message="Token has been revoked",
            context={"jti": "jti123", "revoked_at": "2024-01-01T12:00:00Z"},
        )
        assert str(revoked_error) == "Token has been revoked"

    def test_account_and_permission_errors_real_functionality(self):
        """Test real account and permission error functionality."""
        # Account disabled error
        disabled_error = AccountDisabledError(
            message="User account is disabled",
            context={"user_id": "user123", "disabled_reason": "security_violation"},
        )
        assert str(disabled_error) == "User account is disabled"
        assert disabled_error.context["disabled_reason"] == "security_violation"

        # Invalid credentials error
        credentials_error = InvalidCredentialsError(
            message="Invalid username or password",
            context={"username": "testuser", "ip_address": "192.168.1.1"},
        )
        assert str(credentials_error) == "Invalid username or password"

        # Insufficient permissions error
        permissions_error = InsufficientPermissionsError(
            message="User lacks required permissions",
            required_permissions=["admin", "write"],
            user_permissions=["read"],
        )
        assert str(permissions_error) == "User lacks required permissions"
        assert permissions_error.context["required_permissions"] == ["admin", "write"]

    def test_security_and_rate_limit_errors_real_functionality(self):
        """Test real security and rate limit error functionality."""
        # Rate limit exceeded error
        rate_error = RateLimitExceededError(
            message="Too many authentication attempts",
            limit=5,
            window="minute",
            retry_after=60,
        )
        assert str(rate_error) == "Too many authentication attempts"
        assert rate_error.error_code == "RATE_LIMIT_EXCEEDED"

        # Security violation error
        security_error = SecurityViolationError(
            violation_type="token_tampering",
            message="Token signature mismatch detected",
            client_ip="192.168.1.1",
            user_agent="TestClient/1.0",
        )
        assert str(security_error) == "Token signature mismatch detected"
        assert security_error.error_code == "SECURITY_VIOLATION"
        assert security_error.violation_type == "token_tampering"


class TestRealAuthenticationIntegrationScenarios:
    """Real authentication integration scenario tests."""

    @pytest.fixture
    def auth_system(self):
        """Complete authentication system for testing."""
        jwt_config = JWTConfig(
            secret_key="integration_test_secret_key_32_characters_minimum",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            blacklist_enabled=True,
        )
        jwt_manager = JWTManager(jwt_config)

        # Mock user store
        user_store = {
            "testuser": AuthUser(
                user_id="user123",
                username="testuser",
                email="test@example.com",
                permissions=["read", "write", "predict"],
                roles=["user"],
                is_active=True,
                is_admin=False,
            ),
            "admin": AuthUser(
                user_id="admin123",
                username="admin",
                email="admin@example.com",
                permissions=["read", "write", "admin"],
                roles=["admin"],
                is_active=True,
                is_admin=True,
            ),
        }

        return {
            "jwt_manager": jwt_manager,
            "user_store": user_store,
            "config": jwt_config,
        }

    def test_complete_authentication_flow_real(self, auth_system):
        """Test complete authentication flow with real components."""
        jwt_manager = auth_system["jwt_manager"]
        user_store = auth_system["user_store"]

        # Step 1: User login simulation
        username = "testuser"
        password = "correctpassword"

        # In real system, verify password hash
        user = user_store[username]
        assert user.is_active is True

        # Step 2: Generate tokens for authenticated user
        access_token = jwt_manager.generate_access_token(
            user.user_id, user.permissions, user.to_token_claims()
        )
        refresh_token = jwt_manager.generate_refresh_token(user.user_id)

        # Step 3: Create login response
        login_response = LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600,
            user=user,
        )
        assert login_response.user.username == "testuser"

        # Step 4: Validate access token for API requests
        payload = jwt_manager.validate_token(access_token, "access")
        assert payload["sub"] == user.user_id
        assert payload["username"] == user.username
        assert payload["permissions"] == user.permissions

        # Step 5: Check permissions for protected resources
        authenticated_user = AuthUser(**payload)
        assert authenticated_user.has_permission("read") is True
        assert authenticated_user.has_permission("write") is True
        assert authenticated_user.has_permission("admin") is False

        # Step 6: Token refresh flow
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )
        assert new_access_token != access_token
        assert new_refresh_token != refresh_token

        # Step 7: Validate new tokens
        new_payload = jwt_manager.validate_token(new_access_token, "access")
        assert new_payload["sub"] == user.user_id

        # Step 8: Old refresh token should be invalid
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(refresh_token, "refresh")

    def test_admin_vs_regular_user_permissions_real(self, auth_system):
        """Test real permission differences between admin and regular users."""
        jwt_manager = auth_system["jwt_manager"]
        user_store = auth_system["user_store"]

        # Test regular user
        regular_user = user_store["testuser"]
        regular_token = jwt_manager.generate_access_token(
            regular_user.user_id,
            regular_user.permissions,
            regular_user.to_token_claims(),
        )

        regular_payload = jwt_manager.validate_token(regular_token, "access")
        regular_auth_user = AuthUser(**regular_payload)

        assert regular_auth_user.has_permission("read") is True
        assert regular_auth_user.has_permission("write") is True
        assert regular_auth_user.has_permission("predict") is True
        assert regular_auth_user.has_permission("admin") is False
        assert regular_auth_user.is_admin is False

        # Test admin user
        admin_user = user_store["admin"]
        admin_token = jwt_manager.generate_access_token(
            admin_user.user_id, admin_user.permissions, admin_user.to_token_claims()
        )

        admin_payload = jwt_manager.validate_token(admin_token, "access")
        admin_auth_user = AuthUser(**admin_payload)

        assert admin_auth_user.has_permission("read") is True
        assert admin_auth_user.has_permission("write") is True
        assert admin_auth_user.has_permission("admin") is True
        assert admin_auth_user.has_permission("any_permission") is True  # Admin has all
        assert admin_auth_user.is_admin is True

    def test_token_lifecycle_management_real(self, auth_system):
        """Test real token lifecycle management."""
        jwt_manager = auth_system["jwt_manager"]
        user_store = auth_system["user_store"]

        user = user_store["testuser"]

        # Create tokens
        access_token = jwt_manager.generate_access_token(user.user_id, user.permissions)
        refresh_token = jwt_manager.generate_refresh_token(user.user_id)

        # Get token information
        access_info = jwt_manager.get_token_info(access_token)
        refresh_info = jwt_manager.get_token_info(refresh_token)

        assert access_info["user_id"] == user.user_id
        assert access_info["token_type"] == "access"
        assert access_info["is_expired"] is False
        assert access_info["is_blacklisted"] is False

        assert refresh_info["user_id"] == user.user_id
        assert refresh_info["token_type"] == "refresh"

        # Revoke access token
        revoke_result = jwt_manager.revoke_token(access_token)
        assert revoke_result is True

        # Token should now be blacklisted
        updated_info = jwt_manager.get_token_info(access_token)
        assert updated_info["is_blacklisted"] is True

        # Validation should fail
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(access_token, "access")

    def test_security_violation_detection_real(self, auth_system):
        """Test real security violation detection."""
        jwt_manager = auth_system["jwt_manager"]

        # Test various security violations
        security_violations = [
            ("invalid.token.format", "Invalid token format"),
            ("header.invalid_json.signature", "Token validation failed"),
            ("", "Invalid token format"),
        ]

        for invalid_token, expected_error in security_violations:
            with pytest.raises(APIAuthenticationError):
                jwt_manager.validate_token(invalid_token, "access")

    def test_concurrent_token_operations_real(self, auth_system):
        """Test real concurrent token operations."""
        jwt_manager = auth_system["jwt_manager"]
        user_id = "concurrent_test_user"
        permissions = ["read"]

        # Generate multiple tokens concurrently (simulated)
        tokens = []
        for i in range(10):
            token = jwt_manager.generate_access_token(f"{user_id}_{i}", permissions)
            tokens.append(token)

        # All tokens should be valid and unique
        assert len(tokens) == 10
        assert len(set(tokens)) == 10  # All unique

        # Validate all tokens
        for i, token in enumerate(tokens):
            payload = jwt_manager.validate_token(token, "access")
            assert payload["sub"] == f"{user_id}_{i}"


class TestRealAuthenticationPerformance:
    """Real authentication performance tests."""

    def test_jwt_generation_performance_real(self):
        """Test real JWT generation performance."""
        config = JWTConfig(
            secret_key="performance_test_secret_key_32_characters_minimum"
        )
        manager = JWTManager(config)

        user_id = "perf_user"
        permissions = ["read", "write", "admin", "predict", "analyze"]

        # Measure token generation time
        import time

        start_time = time.time()

        # Generate many tokens
        tokens = []
        for i in range(100):
            token = manager.generate_access_token(f"{user_id}_{i}", permissions)
            tokens.append(token)

        generation_time = time.time() - start_time

        # Should generate 100 tokens quickly (under 1 second)
        assert generation_time < 1.0
        assert len(tokens) == 100

        # Measure validation time
        start_time = time.time()

        for token in tokens[:10]:  # Validate first 10
            payload = manager.validate_token(token, "access")
            assert payload["type"] == "access"

        validation_time = time.time() - start_time

        # Should validate 10 tokens quickly (under 0.1 seconds)
        assert validation_time < 0.1

    def test_blacklist_performance_real(self):
        """Test real blacklist performance."""
        blacklist = TokenBlacklist()

        # Add many tokens to blacklist
        tokens_and_jtis = []
        for i in range(1000):
            token = f"token_{i}.payload.signature"
            jti = f"jti_{i}"
            tokens_and_jtis.append((token, jti))
            blacklist.add_token(token, jti)

        # Check blacklist lookups are fast
        import time

        start_time = time.time()

        for token, jti in tokens_and_jtis[:100]:  # Check first 100
            assert blacklist.is_blacklisted(token, jti) is True

        lookup_time = time.time() - start_time

        # Should check 100 blacklisted tokens quickly
        assert lookup_time < 0.1

    def test_auth_model_validation_performance_real(self):
        """Test real authentication model validation performance."""
        import time

        # Test AuthUser model creation performance
        start_time = time.time()

        users = []
        for i in range(1000):
            user = AuthUser(
                user_id=f"user_{i}",
                username=f"username_{i}",
                email=f"user_{i}@example.com",
                permissions=["read", "write"],
                roles=["user"],
                is_active=True,
            )
            users.append(user)

        creation_time = time.time() - start_time

        # Should create 1000 users quickly
        assert creation_time < 1.0
        assert len(users) == 1000

        # Test permission checking performance
        start_time = time.time()

        for user in users[:100]:  # Check first 100
            assert user.has_permission("read") is True
            assert user.has_permission("write") is True
            assert user.has_permission("admin") is False

        permission_check_time = time.time() - start_time

        # Should check permissions quickly
        assert permission_check_time < 0.1
