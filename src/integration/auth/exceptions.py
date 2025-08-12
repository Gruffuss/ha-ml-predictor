"""
Custom authentication and authorization exceptions.

This module defines specific exception classes for authentication and authorization
errors in the JWT-based security system.
"""

from typing import Any, Dict, Optional

from ...core.exceptions import APIError, ErrorSeverity


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if reason:
            error_context["reason"] = reason

        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            context=error_context,
            severity=ErrorSeverity.HIGH,
        )


class AuthorizationError(APIError):
    """Raised when authorization fails due to insufficient permissions."""

    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
        user_permissions: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if required_permission:
            error_context["required_permission"] = required_permission
        if user_permissions:
            error_context["user_permissions"] = user_permissions

        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            context=error_context,
            severity=ErrorSeverity.MEDIUM,
        )


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired."""

    def __init__(
        self,
        message: str = "Token has expired",
        expired_at: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if expired_at:
            error_context["expired_at"] = expired_at

        super().__init__(message=message, reason="token_expired", context=error_context)


class TokenInvalidError(AuthenticationError):
    """Raised when JWT token is invalid or malformed."""

    def __init__(
        self,
        message: str = "Invalid token",
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if validation_error:
            error_context["validation_error"] = validation_error

        super().__init__(message=message, reason="token_invalid", context=error_context)


class TokenRevokedError(AuthenticationError):
    """Raised when JWT token has been revoked."""

    def __init__(
        self,
        message: str = "Token has been revoked",
        revoked_at: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if revoked_at:
            error_context["revoked_at"] = revoked_at

        super().__init__(message=message, reason="token_revoked", context=error_context)


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        required_permission: str,
        user_permissions: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"Insufficient permissions: '{required_permission}' required"

        super().__init__(
            message=message,
            required_permission=required_permission,
            user_permissions=user_permissions,
            context=context,
        )


class AccountDisabledError(AuthenticationError):
    """Raised when user account is disabled."""

    def __init__(
        self, username: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ):
        message = "Account is disabled"
        if username:
            message = f"Account '{username}' is disabled"

        error_context = context or {}
        if username:
            error_context["username"] = username

        super().__init__(
            message=message, reason="account_disabled", context=error_context
        )


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""

    def __init__(
        self,
        message: str = "Invalid username or password",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, reason="invalid_credentials", context=context)


class RateLimitExceededError(APIError):
    """Raised when authentication rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Too many authentication attempts",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if limit:
            error_context["rate_limit"] = limit
        if window_seconds:
            error_context["window_seconds"] = window_seconds

        super().__init__(
            message=message,
            error_code="AUTH_RATE_LIMIT_EXCEEDED",
            context=error_context,
            severity=ErrorSeverity.MEDIUM,
        )


class SecurityViolationError(APIError):
    """Raised when a security violation is detected."""

    def __init__(
        self,
        violation_type: str,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        if not message:
            message = f"Security violation detected: {violation_type}"

        error_context = context or {}
        error_context["violation_type"] = violation_type

        super().__init__(
            message=message,
            error_code="SECURITY_VIOLATION",
            context=error_context,
            severity=ErrorSeverity.HIGH,
        )
