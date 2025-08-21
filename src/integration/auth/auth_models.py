"""
Authentication models for JWT-based API security.

This module defines Pydantic models for authentication requests, responses,
and user representations used throughout the authentication system.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AuthUser(BaseModel):
    """Authenticated user model with permissions and metadata."""

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username for display")
    email: Optional[str] = Field(None, description="User email address")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    roles: List[str] = Field(default_factory=list, description="User roles")
    is_active: bool = Field(True, description="Whether user account is active")
    is_admin: bool = Field(False, description="Whether user has admin privileges")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Account creation time"
    )

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v):
        """Validate permission strings."""
        valid_permissions = {
            "read",
            "write",
            "admin",
            "model_retrain",
            "system_config",
            "prediction_view",
            "accuracy_view",
            "health_check",
        }

        for permission in v:
            if permission not in valid_permissions:
                raise ValueError(f"Invalid permission: {permission}")

        return v

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, v):
        """Validate role strings."""
        valid_roles = {"user", "admin", "operator", "viewer"}

        for role in v:
            if role not in valid_roles:
                raise ValueError(f"Invalid role: {role}")

        return v

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or self.is_admin

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def to_token_claims(self) -> Dict[str, Any]:
        """Convert user to JWT token claims."""
        return {
            "username": self.username,
            "email": self.email,
            "permissions": self.permissions,
            "roles": self.roles,
            "is_admin": self.is_admin,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class LoginRequest(BaseModel):
    """User login request model."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    remember_me: bool = Field(
        False, description="Whether to issue long-lived refresh token"
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if not v.isalnum() and not all(c.isalnum() or c in "_-." for c in v):
            raise ValueError(
                "Username can only contain alphanumeric characters, underscores, hyphens, and dots"
            )
        return v.lower()  # Normalize to lowercase

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for basic password complexity
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_score < 3:
            raise ValueError(
                "Password must contain at least 3 of: uppercase, lowercase, digit, special character"
            )

        return v


class LoginResponse(BaseModel):
    """Successful login response model."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")
    user: AuthUser = Field(..., description="Authenticated user information")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "user_id": "user123",
                    "username": "admin",
                    "email": "admin@example.com",
                    "permissions": ["read", "write", "admin"],
                    "roles": ["admin"],
                    "is_active": True,
                    "is_admin": True,
                },
            }
        }


class RefreshRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str = Field(..., description="Valid refresh token")

    @field_validator("refresh_token")
    @classmethod
    def validate_refresh_token(cls, v):
        """Basic validation of refresh token format."""
        if not v or len(v) < 10:
            raise ValueError("Invalid refresh token format")

        # Basic JWT format check (3 parts separated by dots)
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format")

        return v


class RefreshResponse(BaseModel):
    """Token refresh response model."""

    access_token: str = Field(..., description="New JWT access token")
    refresh_token: str = Field(..., description="New JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")


class LogoutRequest(BaseModel):
    """User logout request model."""

    refresh_token: Optional[str] = Field(None, description="Refresh token to revoke")
    revoke_all_tokens: bool = Field(
        False, description="Whether to revoke all user tokens"
    )


class TokenInfo(BaseModel):
    """Token information model for introspection."""

    user_id: str = Field(..., description="User ID from token")
    username: Optional[str] = Field(None, description="Username")
    token_type: str = Field(..., description="Type of token (access/refresh)")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    issued_at: datetime = Field(..., description="Token issuance time")
    expires_at: datetime = Field(..., description="Token expiration time")
    is_expired: bool = Field(..., description="Whether token is expired")
    is_active: bool = Field(..., description="Whether token is active (not revoked)")
    jti: Optional[str] = Field(None, description="JWT ID")


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )
    confirm_password: str = Field(..., description="Confirm new password")

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, values):
        """Validate that passwords match."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for basic password complexity
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_score < 3:
            raise ValueError(
                "Password must contain at least 3 of: uppercase, lowercase, digit, special character"
            )

        return v


class UserCreateRequest(BaseModel):
    """User creation request model for admin endpoints."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    roles: List[str] = Field(default_factory=list, description="User roles")
    is_admin: bool = Field(False, description="Whether user has admin privileges")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Basic email validation."""
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if not v.isalnum() and not all(c.isalnum() or c in "_-." for c in v):
            raise ValueError(
                "Username can only contain alphanumeric characters, underscores, hyphens, and dots"
            )
        return v.lower()


class APIKey(BaseModel):
    """API key model for service-to-service authentication."""

    key_id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="Human-readable key name")
    key_hash: str = Field(..., description="Hashed API key")
    permissions: List[str] = Field(
        default_factory=list, description="API key permissions"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(True, description="Whether key is active")
    usage_count: int = Field(0, description="Number of times key has been used")

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if API key has a specific permission."""
        return permission in self.permissions
