"""
FastAPI dependencies for authentication and authorization.

This module provides dependency functions for JWT authentication,
permission checking, and user context injection in FastAPI endpoints.
"""

import logging
from typing import Any, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...core.config import get_config
from ...core.exceptions import APIAuthenticationError, APIAuthorizationError
from .auth_models import AuthUser
from .jwt_manager import JWTManager

logger = logging.getLogger(__name__)

# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None

# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance."""
    global _jwt_manager

    if _jwt_manager is None:
        config = get_config()
        if not config.api.jwt.enabled:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT authentication is not enabled",
            )
        _jwt_manager = JWTManager(config.api.jwt)

    return _jwt_manager


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> AuthUser:
    """
    Dependency to get the current authenticated user.

    This dependency validates the JWT token and returns the authenticated user.
    It can be used as a FastAPI dependency in protected endpoints.

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        Authenticated user object

    Raises:
        HTTPException: If authentication fails
    """
    # Check if user is already set by middleware
    if hasattr(request.state, "user") and request.state.user:
        return request.state.user

    # Manual authentication if middleware didn't process
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        jwt_manager = get_jwt_manager()
        payload = jwt_manager.validate_token(credentials.credentials, "access")

        # Create user object from token payload
        user = AuthUser(
            user_id=payload["sub"],
            username=payload.get("username", payload["sub"]),
            email=payload.get("email"),
            permissions=payload.get("permissions", []),
            roles=payload.get("roles", []),
            is_admin=payload.get("is_admin", False),
            is_active=True,
        )

        return user

    except APIAuthenticationError as e:
        logger.warning(f"Authentication failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error",
        )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[AuthUser]:
    """
    Dependency to get the current user if authenticated, None otherwise.

    This dependency is useful for endpoints that work with both authenticated
    and anonymous users.

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        Authenticated user object or None
    """
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_permission(permission: str):
    """
    Dependency factory for permission-based access control.

    Creates a dependency that checks if the current user has the specified permission.

    Args:
        permission: Required permission string

    Returns:
        FastAPI dependency function

    Example:
        @app.get("/admin/users")
        async def list_users(
            user: AuthUser = Depends(require_permission("admin"))
        ):
            return {"users": [...]}
    """

    async def permission_checker(
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        if not user.has_permission(permission):
            logger.warning(
                f"Permission denied: user {user.user_id} lacks permission '{permission}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: '{permission}' required",
            )
        return user

    return permission_checker


def require_role(role: str):
    """
    Dependency factory for role-based access control.

    Creates a dependency that checks if the current user has the specified role.

    Args:
        role: Required role string

    Returns:
        FastAPI dependency function

    Example:
        @app.get("/operator/status")
        async def get_status(
            user: AuthUser = Depends(require_role("operator"))
        ):
            return {"status": "ok"}
    """

    async def role_checker(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if not user.has_role(role):
            logger.warning(
                f"Role access denied: user {user.user_id} lacks role '{role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role: '{role}' required",
            )
        return user

    return role_checker


def require_admin():
    """
    Dependency for admin-only endpoints.

    Returns:
        FastAPI dependency function that requires admin privileges

    Example:
        @app.delete("/admin/users/{user_id}")
        async def delete_user(
            user_id: str,
            admin: AuthUser = Depends(require_admin())
        ):
            # Delete user logic
    """

    async def admin_checker(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if not user.is_admin:
            logger.warning(f"Admin access denied for user: {user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required",
            )
        return user

    return admin_checker


def require_permissions(permissions: List[str], require_all: bool = True):
    """
    Dependency factory for multiple permission requirements.

    Creates a dependency that checks if the current user has the specified permissions.

    Args:
        permissions: List of required permission strings
        require_all: If True, user must have ALL permissions. If False, ANY permission

    Returns:
        FastAPI dependency function

    Example:
        @app.post("/model/retrain")
        async def retrain_model(
            user: AuthUser = Depends(require_permissions(["model_retrain", "admin"], require_all=False))
        ):
            # Model retraining logic
    """

    async def permissions_checker(
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        user_permissions = set(user.permissions)
        required_permissions = set(permissions)

        if require_all:
            # User must have ALL required permissions
            missing_permissions = required_permissions - user_permissions
            if missing_permissions and not user.is_admin:
                logger.warning(
                    f"Insufficient permissions: user {user.user_id} missing {missing_permissions}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {', '.join(missing_permissions)}",
                )
        else:
            # User must have ANY of the required permissions
            if not (user_permissions & required_permissions) and not user.is_admin:
                logger.warning(
                    f"No required permissions: user {user.user_id} needs one of {required_permissions}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"One of these permissions required: {', '.join(permissions)}",
                )

        return user

    return permissions_checker


async def validate_api_key(
    api_key: Optional[str] = None, request: Request = None
) -> bool:
    """
    Validate API key for service-to-service authentication.

    This is an alternative authentication method for non-human clients.

    Args:
        api_key: API key string
        request: FastAPI request object

    Returns:
        True if API key is valid, False otherwise
    """
    config = get_config()

    if not config.api.api_key_enabled:
        return True

    # Extract API key from different sources
    if not api_key:
        # Try header
        if request:
            api_key = request.headers.get("X-API-Key")

        # Try query parameter (less secure, for debugging only)
        if not api_key and request and config.api.debug:
            api_key = request.query_params.get("api_key")

    if not api_key:
        return False

    # Simple API key validation (in production, use hashed keys from database)
    return api_key == config.api.api_key


def get_request_context(request: Request) -> dict:
    """
    Get request context information for logging and security monitoring.

    Args:
        request: FastAPI request object

    Returns:
        Dictionary with request context information
    """
    context = {
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("User-Agent"),
        "timestamp": request.state.__dict__.get("start_time"),
    }

    # Add user context if available
    if hasattr(request.state, "user") and request.state.user:
        context["user_id"] = request.state.user.user_id
        context["user_permissions"] = request.state.user.permissions
        context["is_admin"] = request.state.user.is_admin

    return context
