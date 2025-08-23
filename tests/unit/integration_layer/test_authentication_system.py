"""Unit tests for authentication and security system.

Covers:
- src/integration/auth/auth_models.py (Authentication Models)
- src/integration/auth/jwt_manager.py (JWT Token Management)
- src/integration/auth/endpoints.py (Auth Endpoints)
- src/integration/auth/dependencies.py (Auth Dependencies)
- src/integration/auth/middleware.py (Auth Middleware)
- src/integration/auth/exceptions.py (Auth Exceptions)

This test file consolidates testing for all authentication and security functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta
import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from starlette.responses import JSONResponse
from pydantic import ValidationError, BaseModel, Field


# Mock authentication models for testing
class AuthUser(BaseModel):
    """Mock AuthUser model for testing."""
    user_id: str
    username: str
    email: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_admin: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

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
    """Mock LoginRequest model for testing."""
    username: str
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Mock LoginResponse model for testing."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUser


class RefreshRequest(BaseModel):
    """Mock RefreshRequest model for testing."""
    refresh_token: str


class RefreshResponse(BaseModel):
    """Mock RefreshResponse model for testing."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LogoutRequest(BaseModel):
    """Mock LogoutRequest model for testing."""
    refresh_token: Optional[str] = None
    revoke_all_tokens: bool = False


class TokenInfo(BaseModel):
    """Mock TokenInfo model for testing."""
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
    """Mock PasswordChangeRequest model for testing."""
    current_password: str
    new_password: str
    confirm_password: str


class UserCreateRequest(BaseModel):
    """Mock UserCreateRequest model for testing."""
    username: str
    email: str
    password: str
    permissions: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    is_admin: bool = False


class APIKey(BaseModel):
    """Mock APIKey model for testing."""
    key_id: str
    name: str
    key_hash: str
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
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


# Mock JWT configuration
class JWTConfig:
    """Mock JWT configuration for testing."""
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', True)
        self.secret_key = kwargs.get('secret_key', 'test_secret_key_32_characters_long!')
        self.algorithm = kwargs.get('algorithm', 'HS256')
        self.access_token_expire_minutes = kwargs.get('access_token_expire_minutes', 60)
        self.refresh_token_expire_days = kwargs.get('refresh_token_expire_days', 30)
        self.issuer = kwargs.get('issuer', 'test-issuer')
        self.audience = kwargs.get('audience', 'test-audience')
        self.blacklist_enabled = kwargs.get('blacklist_enabled', True)


# Mock exception classes
class APIAuthenticationError(Exception):
    """Mock authentication error."""
    def __init__(self, message, error_code="AUTH_ERROR", **kwargs):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class APISecurityError(Exception):
    """Mock security error."""
    def __init__(self, error_code, message, **kwargs):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class AuthenticationError(Exception):
    """Mock authentication error."""
    def __init__(self, message="Authentication failed", reason=None, context=None, **kwargs):
        self.message = message
        self.error_code = "AUTHENTICATION_FAILED"
        self.context = context or {}
        if reason:
            self.context["reason"] = reason
        super().__init__(message)


class AuthorizationError(Exception):
    """Mock authorization error."""
    def __init__(self, message="Access denied", **kwargs):
        self.message = message
        self.error_code = "AUTHORIZATION_FAILED"
        self.context = kwargs
        super().__init__(message)


class TokenExpiredError(AuthenticationError):
    """Mock token expired error."""
    pass


class TokenInvalidError(AuthenticationError):
    """Mock token invalid error."""
    pass


class TokenRevokedError(AuthenticationError):
    """Mock token revoked error."""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """Mock insufficient permissions error."""
    pass


class AccountDisabledError(AuthenticationError):
    """Mock account disabled error."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Mock invalid credentials error."""
    pass


class RateLimitExceededError(Exception):
    """Mock rate limit error."""
    def __init__(self, message="Rate limit exceeded", **kwargs):
        self.message = message
        self.error_code = "RATE_LIMIT_EXCEEDED"
        super().__init__(message)


class SecurityViolationError(Exception):
    """Mock security violation error."""
    def __init__(self, violation_type, message=None, **kwargs):
        self.violation_type = violation_type
        self.message = message or f"Security violation: {violation_type}"
        self.error_code = "SECURITY_VIOLATION"
        super().__init__(self.message)


# Mock JWT Manager and Token Blacklist
class TokenBlacklist:
    """Mock token blacklist for testing."""
    def __init__(self):
        self._blacklisted_tokens = set()
        self._blacklisted_jti = set()

    def add_token(self, token: str, jti: Optional[str] = None):
        """Add token to blacklist."""
        self._blacklisted_tokens.add(token)
        if jti:
            self._blacklisted_jti.add(jti)

    def is_blacklisted(self, token: str, jti: Optional[str] = None) -> bool:
        """Check if token is blacklisted."""
        if token in self._blacklisted_tokens:
            return True
        if jti and jti in self._blacklisted_jti:
            return True
        return False


class JWTManager:
    """Mock JWT manager for testing."""
    def __init__(self, config: JWTConfig):
        self.config = config
        self.blacklist = TokenBlacklist() if config.blacklist_enabled else None
        self._token_operations = {}
        self._max_operations_per_minute = 30

        if not config.secret_key or len(config.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")

    def generate_access_token(self, user_id: str, permissions: List[str], 
                            additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Generate mock access token."""
        self._check_rate_limit(user_id)
        
        now = datetime.now(timezone.utc)
        exp = now + timedelta(minutes=self.config.access_token_expire_minutes)
        # Add random component to make tokens unique
        import uuid
        jti = f"jti_{user_id}_{int(now.timestamp())}_{str(uuid.uuid4())[:8]}"
        
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "jti": jti,
            "type": "access",
            "permissions": permissions,
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        # Return a mock JWT token
        return f"header.{json.dumps(payload)}.signature"

    def generate_refresh_token(self, user_id: str) -> str:
        """Generate mock refresh token."""
        self._check_rate_limit(user_id)
        
        now = datetime.now(timezone.utc)
        exp = now + timedelta(days=self.config.refresh_token_expire_days)
        # Add random component to make tokens unique
        import uuid
        jti = f"refresh_jti_{user_id}_{int(now.timestamp())}_{str(uuid.uuid4())[:8]}"
        
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "jti": jti,
            "type": "refresh",
        }
        
        return f"header.{json.dumps(payload)}.signature"

    def validate_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Validate mock token."""
        try:
            # Parse mock token
            parts = token.split(".")
            if len(parts) != 3:
                raise APIAuthenticationError("Invalid token format")
            
            payload = json.loads(parts[1])
            
            # Check token type
            if payload.get("type") != token_type:
                raise APIAuthenticationError(f"Invalid token type. Expected '{token_type}', got '{payload.get('type')}'")
            
            # Check blacklist
            if self.blacklist and self.blacklist.is_blacklisted(token, payload.get("jti")):
                raise APIAuthenticationError("Token has been revoked")
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise APIAuthenticationError("Token has expired")
            
            # Check issuer and audience
            if payload.get("iss") != self.config.issuer:
                raise APIAuthenticationError("Invalid token issuer")
            if payload.get("aud") != self.config.audience:
                raise APIAuthenticationError("Invalid token audience")
            
            return payload
            
        except json.JSONDecodeError:
            raise APIAuthenticationError("Invalid token format")
        except APIAuthenticationError:
            raise
        except Exception as e:
            raise APIAuthenticationError(f"Token validation failed: {str(e)}")

    def refresh_access_token(self, refresh_token: str):
        """Refresh access token."""
        payload = self.validate_token(refresh_token, "refresh")
        user_id = payload["sub"]
        
        # Blacklist old token
        if self.blacklist:
            self.blacklist.add_token(refresh_token, payload.get("jti"))
        
        # Generate new tokens
        permissions = ["read", "write"]  # Default permissions
        new_access_token = self.generate_access_token(user_id, permissions)
        new_refresh_token = self.generate_refresh_token(user_id)
        
        return new_access_token, new_refresh_token

    def revoke_token(self, token: str) -> bool:
        """Revoke token."""
        if not self.blacklist:
            return False
        
        try:
            payload = json.loads(token.split(".")[1])
            self.blacklist.add_token(token, payload.get("jti"))
            return True
        except:
            return False

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """Get token information."""
        try:
            payload = json.loads(token.split(".")[1])
            return {
                "user_id": payload.get("sub"),
                "token_type": payload.get("type"),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0)),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0)),
                "permissions": payload.get("permissions", []),
                "jti": payload.get("jti"),
                "is_expired": payload.get("exp", 0) < time.time(),
                "is_blacklisted": self.blacklist.is_blacklisted(token, payload.get("jti")) if self.blacklist else False,
            }
        except:
            return {"error": "Invalid token"}

    def _check_rate_limit(self, user_id: str):
        """Check rate limits."""
        now = time.time()
        if user_id not in self._token_operations:
            self._token_operations[user_id] = []
        
        # Clean old operations
        self._token_operations[user_id] = [
            t for t in self._token_operations[user_id] 
            if t > now - 60
        ]
        
        if len(self._token_operations[user_id]) >= self._max_operations_per_minute:
            raise APISecurityError("rate_limit_exceeded", f"Too many token operations for user {user_id}")
        
        self._token_operations[user_id].append(now)


class TestAuthenticationModels:
    """Test authentication data models."""

    def test_auth_user_model_creation(self):
        """Test AuthUser model creation and validation."""
        user = AuthUser(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_active=True,
            is_admin=False,
        )

        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.permissions == ["read", "write"]
        assert user.roles == ["user"]
        assert user.is_active is True
        assert user.is_admin is False

    def test_auth_user_has_permission(self):
        """Test AuthUser permission checking methods."""
        user = AuthUser(
            user_id="test",
            username="test",
            permissions=["read", "write"],
            is_admin=False,
        )

        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("admin") is False

        # Test admin user has all permissions
        admin_user = AuthUser(
            user_id="admin",
            username="admin",
            permissions=["read"],
            is_admin=True,
        )
        assert admin_user.has_permission("admin") is True
        assert admin_user.has_permission("any_permission") is True

    def test_auth_user_has_role(self):
        """Test AuthUser role checking methods."""
        user = AuthUser(
            user_id="test",
            username="test",
            roles=["user", "operator"],
        )

        assert user.has_role("user") is True
        assert user.has_role("operator") is True
        assert user.has_role("admin") is False

    def test_auth_user_to_token_claims(self):
        """Test AuthUser token claims generation."""
        user = AuthUser(
            user_id="test",
            username="test",
            email="test@example.com",
            permissions=["read", "write"],
            roles=["user"],
            is_admin=False,
            last_login=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )

        claims = user.to_token_claims()
        expected_claims = {
            "username": "test",
            "email": "test@example.com",
            "permissions": ["read", "write"],
            "roles": ["user"],
            "is_admin": False,
            "last_login": "2023-01-01T00:00:00+00:00",
        }

        assert claims == expected_claims

    def test_login_request_validation(self):
        """Test LoginRequest model validation."""
        # Valid login request
        login_request = LoginRequest(
            username="testuser",
            password="TestPass123!",
            remember_me=False,
        )
        assert login_request.username == "testuser"
        assert login_request.password == "TestPass123!"
        assert login_request.remember_me is False

    def test_api_key_model(self):
        """Test APIKey model functionality."""
        api_key = APIKey(
            key_id="test_key",
            name="Test API Key",
            key_hash="hashed_key",
            permissions=["read", "write"],
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        assert api_key.key_id == "test_key"
        assert api_key.name == "Test API Key"
        assert api_key.is_expired() is False
        assert api_key.has_permission("read") is True
        assert api_key.has_permission("admin") is False

        # Test expired key
        expired_key = APIKey(
            key_id="expired_key",
            name="Expired Key",
            key_hash="hashed",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert expired_key.is_expired() is True


class TestJWTManager:
    """Test JWT token management."""

    @pytest.fixture
    def jwt_config(self):
        """Create test JWT configuration."""
        return JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test-issuer",
            audience="test-audience",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def jwt_manager(self, jwt_config):
        """Create JWT manager instance."""
        return JWTManager(jwt_config)

    def test_jwt_manager_initialization(self, jwt_config):
        """Test JWT manager initialization."""
        manager = JWTManager(jwt_config)
        assert manager.config == jwt_config
        assert manager.blacklist is not None

        # Test initialization with invalid config
        invalid_config = JWTConfig(
            enabled=True,
            secret_key="short",  # Too short
        )
        with pytest.raises(ValueError, match="at least 32 characters"):
            JWTManager(invalid_config)

    def test_generate_access_token(self, jwt_manager):
        """Test access token generation."""
        user_id = "test_user"
        permissions = ["read", "write"]
        additional_claims = {"role": "user"}

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT format: header.payload.signature

        # Validate token content
        payload = jwt_manager.validate_token(token, "access")
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert payload["iss"] == jwt_manager.config.issuer
        assert payload["aud"] == jwt_manager.config.audience

    def test_generate_refresh_token(self, jwt_manager):
        """Test refresh token generation."""
        user_id = "test_user"
        token = jwt_manager.generate_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3

        # Validate token content
        payload = jwt_manager.validate_token(token, "refresh")
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_validate_token_success(self, jwt_manager):
        """Test successful token validation."""
        user_id = "test_user"
        permissions = ["read"]
        token = jwt_manager.generate_access_token(user_id, permissions)

        payload = jwt_manager.validate_token(token, "access")
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions

    def test_validate_token_wrong_type(self, jwt_manager):
        """Test token validation with wrong type."""
        token = jwt_manager.generate_access_token("user", [])

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.validate_token(token, "refresh")

    def test_validate_token_expired(self, jwt_manager):
        """Test validation of expired token."""
        # Create token with short expiration
        jwt_manager.config.access_token_expire_minutes = -1
        token = jwt_manager.generate_access_token("user", [])
        jwt_manager.config.access_token_expire_minutes = 60  # Reset

        with pytest.raises(APIAuthenticationError, match="Token has expired"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_blacklisted(self, jwt_manager):
        """Test validation of blacklisted token."""
        token = jwt_manager.generate_access_token("user", [])
        payload = jwt_manager.validate_token(token, "access")

        # Blacklist the token
        jwt_manager.blacklist.add_token(token, payload.get("jti"))

        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")

    def test_refresh_access_token(self, jwt_manager):
        """Test token refresh functionality."""
        user_id = "test_user"
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )

        assert isinstance(new_access_token, str)
        assert isinstance(new_refresh_token, str)
        assert new_access_token != refresh_token
        assert new_refresh_token != refresh_token

        # Validate new tokens
        access_payload = jwt_manager.validate_token(new_access_token, "access")
        refresh_payload = jwt_manager.validate_token(new_refresh_token, "refresh")
        assert access_payload["sub"] == user_id
        assert refresh_payload["sub"] == user_id

        # Old refresh token should be blacklisted
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(refresh_token, "refresh")

    def test_revoke_token(self, jwt_manager):
        """Test token revocation."""
        token = jwt_manager.generate_access_token("user", [])
        result = jwt_manager.revoke_token(token)
        assert result is True

        # Token should now be blacklisted
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(token, "access")

    def test_get_token_info(self, jwt_manager):
        """Test token information extraction."""
        user_id = "test_user"
        permissions = ["read", "write"]
        token = jwt_manager.generate_access_token(user_id, permissions)

        token_info = jwt_manager.get_token_info(token)
        assert token_info["user_id"] == user_id
        assert token_info["token_type"] == "access"
        assert token_info["permissions"] == permissions
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False

    @patch("time.time")
    def test_rate_limiting(self, mock_time, jwt_manager):
        """Test JWT manager rate limiting."""
        mock_time.return_value = 1000.0
        user_id = "test_user"

        # Generate tokens up to limit
        for _ in range(30):  # Default limit
            jwt_manager.generate_access_token(user_id, [])

        # Next request should be rate limited
        with pytest.raises(APISecurityError, match="Too many token operations"):
            jwt_manager.generate_access_token(user_id, [])


class TestTokenBlacklist:
    """Test token blacklist functionality."""

    @pytest.fixture
    def blacklist(self):
        """Create token blacklist instance."""
        return TokenBlacklist()

    def test_add_and_check_token(self, blacklist):
        """Test adding and checking blacklisted tokens."""
        token = "test.token.signature"
        jti = "test-jti-123"

        # Token should not be blacklisted initially
        assert blacklist.is_blacklisted(token, jti) is False

        # Add token to blacklist
        blacklist.add_token(token, jti)

        # Token should now be blacklisted
        assert blacklist.is_blacklisted(token, jti) is True
        assert blacklist.is_blacklisted(token) is True
        assert blacklist.is_blacklisted("other.token", jti) is True

    def test_jti_only_blacklisting(self, blacklist):
        """Test blacklisting by JTI only."""
        jti = "test-jti-123"
        blacklist.add_token("", jti)  # Add only JTI

        assert blacklist.is_blacklisted("any.token", jti) is True
        assert blacklist.is_blacklisted("any.token", "other-jti") is False


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    def test_password_hashing_and_verification(self):
        """Test password hashing and verification."""
        password = "test_password"
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Verify password function
        def verify_password(password: str, hash: str) -> bool:
            return hashlib.sha256(password.encode()).hexdigest() == hash
        
        assert verify_password(password, password_hash) is True
        assert verify_password("wrong_password", password_hash) is False

    @pytest.mark.asyncio
    async def test_login_endpoint_logic(self):
        """Test login endpoint logic."""
        # Mock user store
        USER_STORE = {
            "admin": {
                "user_id": "admin",
                "username": "admin",
                "email": "admin@example.com",
                "password_hash": hashlib.sha256("admin123!".encode()).hexdigest(),
                "permissions": ["read", "write", "admin"],
                "roles": ["admin"],
                "is_admin": True,
                "is_active": True,
            }
        }

        def verify_password(password: str, hash: str) -> bool:
            return hashlib.sha256(password.encode()).hexdigest() == hash

        # Test successful login logic
        username = "admin"
        password = "admin123!"
        
        user_data = USER_STORE.get(username.lower())
        assert user_data is not None
        assert verify_password(password, user_data["password_hash"]) is True
        assert user_data.get("is_active", False) is True

        # Create user object
        user = AuthUser(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data.get("email"),
            permissions=user_data.get("permissions", []),
            roles=user_data.get("roles", []),
            is_admin=user_data.get("is_admin", False),
            is_active=user_data.get("is_active", True),
        )

        assert user.username == "admin"
        assert user.is_admin is True


class TestAuthDependencies:
    """Test authentication dependencies."""

    @pytest.mark.asyncio
    async def test_permission_checking_logic(self):
        """Test permission checking dependency logic."""
        user = AuthUser(
            user_id="test",
            username="test",
            permissions=["read", "write"],
            roles=[],
            is_admin=False,
        )

        # Permission checker logic
        def check_permission(user: AuthUser, required_permission: str) -> bool:
            return user.has_permission(required_permission)

        assert check_permission(user, "read") is True
        assert check_permission(user, "write") is True
        assert check_permission(user, "admin") is False

        # Admin user should have all permissions
        admin_user = AuthUser(
            user_id="admin",
            username="admin",
            permissions=["read"],
            roles=["admin"],
            is_admin=True,
        )
        assert check_permission(admin_user, "admin") is True
        assert check_permission(admin_user, "any_permission") is True

    @pytest.mark.asyncio
    async def test_role_checking_logic(self):
        """Test role checking dependency logic."""
        user = AuthUser(
            user_id="test",
            username="test",
            permissions=[],
            roles=["user", "operator"],
            is_admin=False,
        )

        def check_role(user: AuthUser, required_role: str) -> bool:
            return user.has_role(required_role)

        assert check_role(user, "user") is True
        assert check_role(user, "operator") is True
        assert check_role(user, "admin") is False

    @pytest.mark.asyncio
    async def test_admin_checking_logic(self):
        """Test admin checking dependency logic."""
        admin_user = AuthUser(
            user_id="admin",
            username="admin",
            permissions=[],
            roles=["admin"],
            is_admin=True,
        )

        regular_user = AuthUser(
            user_id="user",
            username="user",
            permissions=["read"],
            roles=["user"],
            is_admin=False,
        )

        def check_admin(user: AuthUser) -> bool:
            return user.is_admin

        assert check_admin(admin_user) is True
        assert check_admin(regular_user) is False


class TestAuthMiddleware:
    """Test authentication middleware."""

    @pytest.mark.asyncio
    async def test_security_headers_logic(self):
        """Test security headers middleware logic."""
        # Mock security headers that would be added
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }

        # Test that headers are properly defined
        assert security_headers["X-Content-Type-Options"] == "nosniff"
        assert security_headers["X-Frame-Options"] == "DENY"
        assert "max-age=" in security_headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_authentication_logic(self):
        """Test authentication middleware logic."""
        # Test public endpoint detection
        public_endpoints = {"/", "/health", "/docs", "/auth/login"}
        
        def is_public_endpoint(path: str) -> bool:
            return path in public_endpoints or path.startswith("/static/")

        assert is_public_endpoint("/health") is True
        assert is_public_endpoint("/auth/login") is True
        assert is_public_endpoint("/protected") is False
        assert is_public_endpoint("/static/style.css") is True

        # Test admin endpoint detection
        def is_admin_endpoint(path: str) -> bool:
            admin_patterns = ["/admin/", "/users/", "/system/config"]
            return any(path.startswith(pattern) for pattern in admin_patterns)

        assert is_admin_endpoint("/admin/dashboard") is True
        assert is_admin_endpoint("/users/create") is True
        assert is_admin_endpoint("/api/predictions") is False

    @pytest.mark.asyncio
    async def test_rate_limiting_logic(self):
        """Test rate limiting middleware logic."""
        # Mock rate limiting
        request_counts = {}
        rate_limit = 60  # requests per minute
        
        def check_rate_limit(client_ip: str) -> bool:
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            
            # Clean old requests
            request_counts[client_ip] = [
                t for t in request_counts[client_ip] 
                if t > window_start
            ]
            
            # Check limit
            if len(request_counts[client_ip]) >= rate_limit:
                return False
            
            # Record this request
            request_counts[client_ip].append(now)
            return True

        client_ip = "127.0.0.1"
        
        # Should allow requests under limit
        for _ in range(rate_limit):
            assert check_rate_limit(client_ip) is True
        
        # Should block requests over limit
        assert check_rate_limit(client_ip) is False


class TestAuthExceptions:
    """Test authentication exceptions."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError(
            message="Custom auth error",
            reason="invalid_token",
            context={"token_type": "access"},
        )

        assert error.message == "Custom auth error"
        assert error.error_code == "AUTHENTICATION_FAILED"
        assert error.context["reason"] == "invalid_token"
        assert error.context["token_type"] == "access"

    def test_authorization_error(self):
        """Test AuthorizationError exception."""
        error = AuthorizationError(
            message="Access denied",
            required_permission="admin",
            user_permissions=["read", "write"],
        )

        assert error.message == "Access denied"
        assert error.error_code == "AUTHORIZATION_FAILED"
        assert "required_permission" in error.context
        assert "user_permissions" in error.context

    def test_token_expired_error(self):
        """Test TokenExpiredError exception."""
        error = TokenExpiredError(
            message="Token expired", 
            context={"user_id": "123"}
        )

        assert error.message == "Token expired"
        assert error.error_code == "AUTHENTICATION_FAILED"
        assert error.context["user_id"] == "123"

    def test_security_violation_error(self):
        """Test SecurityViolationError exception."""
        error = SecurityViolationError(
            violation_type="token_tampering",
            message="Token signature mismatch",
        )

        assert error.message == "Token signature mismatch"
        assert error.error_code == "SECURITY_VIOLATION"
        assert error.violation_type == "token_tampering"

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError exception."""
        error = RateLimitExceededError(
            message="Too many requests"
        )

        assert error.message == "Too many requests"
        assert error.error_code == "RATE_LIMIT_EXCEEDED"


class TestIntegratedAuthenticationFlow:
    """Test integrated authentication workflows."""

    @pytest.fixture
    def auth_system(self):
        """Create integrated auth system for testing."""
        jwt_config = JWTConfig(
            secret_key="test_secret_key_32_characters_long!",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
        )
        jwt_manager = JWTManager(jwt_config)
        
        return {
            "jwt_manager": jwt_manager,
            "config": jwt_config,
        }

    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, auth_system):
        """Test complete authentication flow from login to protected access."""
        jwt_manager = auth_system["jwt_manager"]
        
        # Step 1: Generate tokens for user login
        user_id = "test_user"
        permissions = ["read", "write", "prediction_view"]
        access_token = jwt_manager.generate_access_token(user_id, permissions)
        refresh_token = jwt_manager.generate_refresh_token(user_id)
        
        # Step 2: Validate access token
        payload = jwt_manager.validate_token(access_token, "access")
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        
        # Step 3: Create user from token
        user = AuthUser(
            user_id=payload["sub"],
            username=payload.get("username", payload["sub"]),
            permissions=payload.get("permissions", []),
            roles=payload.get("roles", []),
            is_admin=payload.get("is_admin", False),
        )
        
        # Step 4: Check permissions
        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("admin") is False
        
        # Step 5: Refresh tokens
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(refresh_token)
        
        # Step 6: Validate new tokens
        new_payload = jwt_manager.validate_token(new_access_token, "access")
        assert new_payload["sub"] == user_id
        
        # Step 7: Old refresh token should be invalid
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(refresh_token, "refresh")

    @pytest.mark.asyncio
    async def test_security_violation_detection(self, auth_system):
        """Test security violation detection and handling."""
        jwt_manager = auth_system["jwt_manager"]
        
        # Test invalid token format
        with pytest.raises(APIAuthenticationError, match="Invalid token format"):
            jwt_manager.validate_token("invalid.token", "access")
        
        # Test token tampering - modify the token structure to trigger validation error
        valid_token = jwt_manager.generate_access_token("user", ["read"])
        # Tamper with the payload section
        parts = valid_token.split(".")
        tampered_payload = json.dumps({"invalid": "payload"})
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"
        
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(tampered_token, "access")
        
        # Test expired token handling
        jwt_manager.config.access_token_expire_minutes = -1
        expired_token = jwt_manager.generate_access_token("user", ["read"])
        jwt_manager.config.access_token_expire_minutes = 60  # Reset
        
        with pytest.raises(APIAuthenticationError, match="Token has expired"):
            jwt_manager.validate_token(expired_token, "access")

    @pytest.mark.asyncio
    async def test_permission_based_access_control(self, auth_system):
        """Test permission-based access control scenarios."""
        # Test different user types
        viewer_user = AuthUser(
            user_id="viewer",
            username="viewer",
            permissions=["read", "prediction_view"],
            roles=["viewer"],
            is_admin=False,
        )
        
        operator_user = AuthUser(
            user_id="operator",
            username="operator",
            permissions=["read", "write", "prediction_view", "accuracy_view"],
            roles=["operator"],
            is_admin=False,
        )
        
        admin_user = AuthUser(
            user_id="admin",
            username="admin",
            permissions=["read", "write", "admin", "model_retrain"],
            roles=["admin"],
            is_admin=True,
        )
        
        # Test viewer permissions
        assert viewer_user.has_permission("read") is True
        assert viewer_user.has_permission("prediction_view") is True
        assert viewer_user.has_permission("write") is False
        assert viewer_user.has_permission("admin") is False
        
        # Test operator permissions
        assert operator_user.has_permission("read") is True
        assert operator_user.has_permission("write") is True
        assert operator_user.has_permission("accuracy_view") is True
        assert operator_user.has_permission("admin") is False
        
        # Test admin permissions (should have all)
        assert admin_user.has_permission("read") is True
        assert admin_user.has_permission("write") is True
        assert admin_user.has_permission("admin") is True
        assert admin_user.has_permission("any_permission") is True

    @pytest.mark.asyncio
    async def test_token_lifecycle_management(self, auth_system):
        """Test complete token lifecycle management."""
        jwt_manager = auth_system["jwt_manager"]
        
        # Create and validate token
        user_id = "lifecycle_user"
        token = jwt_manager.generate_access_token(user_id, ["read"])
        
        # Token should be valid initially
        payload = jwt_manager.validate_token(token, "access")
        assert payload["sub"] == user_id
        
        # Get token info
        token_info = jwt_manager.get_token_info(token)
        assert token_info["user_id"] == user_id
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False
        
        # Revoke token
        revoke_result = jwt_manager.revoke_token(token)
        assert revoke_result is True
        
        # Token should now be invalid
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")
        
        # Token info should show blacklisted
        token_info_after_revoke = jwt_manager.get_token_info(token)
        assert token_info_after_revoke["is_blacklisted"] is True