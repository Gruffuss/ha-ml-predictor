"""
Authentication endpoints for JWT-based API security.

This module provides FastAPI endpoints for user authentication, token management,
and user account operations.
"""

from datetime import datetime, timezone
import hashlib
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
import secrets

from ...core.exceptions import APIAuthenticationError
from .auth_models import (
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
from .dependencies import (
    get_current_user,
    get_jwt_manager,
    require_admin,
    security_scheme,
)
from .jwt_manager import JWTManager

logger = logging.getLogger(__name__)

# Create router for authentication endpoints
auth_router = APIRouter(prefix="/auth", tags=["authentication"])


# Simple in-memory user store for demonstration
# In production, this would be a proper database with hashed passwords
USER_STORE = {
    "admin": {
        "user_id": "admin",
        "username": "admin",
        "email": "admin@ha-ml-predictor.local",
        "password_hash": hashlib.sha256("admin123!".encode()).hexdigest(),
        "permissions": ["read", "write", "admin", "model_retrain", "system_config"],
        "roles": ["admin"],
        "is_admin": True,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    },
    "operator": {
        "user_id": "operator",
        "username": "operator",
        "email": "operator@ha-ml-predictor.local",
        "password_hash": hashlib.sha256("operator123!".encode()).hexdigest(),
        "permissions": ["read", "write", "prediction_view", "accuracy_view"],
        "roles": ["operator"],
        "is_admin": False,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    },
    "viewer": {
        "user_id": "viewer",
        "username": "viewer",
        "email": "viewer@ha-ml-predictor.local",
        "password_hash": hashlib.sha256("viewer123!".encode()).hexdigest(),
        "permissions": ["read", "prediction_view", "health_check"],
        "roles": ["viewer"],
        "is_admin": False,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    },
}


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hashlib.sha256(password.encode()).hexdigest() == password_hash


def hash_password(password: str) -> str:
    """Hash password for storage."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username from the user store."""
    return USER_STORE.get(username.lower())


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    login_request: LoginRequest,
    request: Request,
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    """
    Authenticate user and return JWT tokens.

    This endpoint validates user credentials and returns access and refresh tokens
    for subsequent API requests.
    """
    try:
        # Get user from store
        user_data = get_user_by_username(login_request.username)
        if not user_data:
            logger.warning(
                f"Login attempt for non-existent user: {login_request.username}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # Verify password
        if not verify_password(login_request.password, user_data["password_hash"]):
            logger.warning(f"Invalid password for user: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # Check if account is active
        if not user_data.get("is_active", False):
            logger.warning(f"Login attempt for inactive user: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled"
            )

        # Create user object
        user = AuthUser(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data.get("email"),
            permissions=user_data.get("permissions", []),
            roles=user_data.get("roles", []),
            is_admin=user_data.get("is_admin", False),
            is_active=user_data.get("is_active", True),
            last_login=datetime.now(timezone.utc),
            created_at=user_data.get("created_at", datetime.now(timezone.utc)),
        )

        # Generate tokens
        additional_claims = user.to_token_claims()
        access_token = jwt_manager.generate_access_token(
            user.user_id, user.permissions, additional_claims
        )

        # Adjust refresh token expiration for "remember me"
        original_refresh_days = jwt_manager.config.refresh_token_expire_days
        if login_request.remember_me:
            jwt_manager.config.refresh_token_expire_days = 90  # 3 months

        refresh_token = jwt_manager.generate_refresh_token(user.user_id)

        # Restore original expiration
        jwt_manager.config.refresh_token_expire_days = original_refresh_days

        # Update last login time in user store
        USER_STORE[user.username]["last_login"] = datetime.now(timezone.utc)

        # Log successful login
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"Successful login: user={user.username}, ip={client_ip}",
            extra={
                "user_id": user.user_id,
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent"),
                "remember_me": login_request.remember_me,
            },
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=jwt_manager.config.access_token_expire_minutes * 60,
            user=user,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login service unavailable",
        )


@auth_router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    refresh_request: RefreshRequest, jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    Refresh access token using a valid refresh token.

    This endpoint generates new access and refresh tokens, invalidating the old refresh token.
    """
    try:
        # Refresh tokens
        access_token, refresh_token = jwt_manager.refresh_access_token(
            refresh_request.refresh_token
        )

        logger.info("Token refreshed successfully")

        return RefreshResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=jwt_manager.config.access_token_expire_minutes * 60,
        )

    except APIAuthenticationError as e:
        logger.warning(f"Token refresh failed: {e.message}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.message)
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service unavailable",
        )


@auth_router.post("/logout")
async def logout(
    logout_request: LogoutRequest,
    user: AuthUser = Depends(get_current_user),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    """
    Logout user by revoking tokens.

    This endpoint revokes the user's refresh token to prevent further token refresh.
    """
    try:
        revoked_tokens = 0

        # Revoke refresh token if provided
        if logout_request.refresh_token:
            if jwt_manager.revoke_token(logout_request.refresh_token):
                revoked_tokens += 1

        # Log logout
        logger.info(f"User logged out: {user.username}")

        return {
            "message": "Logout successful",
            "revoked_tokens": revoked_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service unavailable",
        )


@auth_router.get("/me", response_model=AuthUser)
async def get_current_user_info(user: AuthUser = Depends(get_current_user)):
    """
    Get current user information.

    This endpoint returns the authenticated user's profile and permissions.
    """
    return user


@auth_router.post("/change-password")
async def change_password(
    password_request: PasswordChangeRequest, user: AuthUser = Depends(get_current_user)
):
    """
    Change user password.

    This endpoint allows authenticated users to change their password.
    """
    try:
        # Get user from store
        user_data = get_user_by_username(user.username)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Verify current password
        if not verify_password(
            password_request.current_password, user_data["password_hash"]
        ):
            logger.warning(f"Invalid current password for user: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

        # Update password
        new_password_hash = hash_password(password_request.new_password)
        USER_STORE[user.username]["password_hash"] = new_password_hash

        logger.info(f"Password changed for user: {user.username}")

        return {
            "message": "Password changed successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service unavailable",
        )


@auth_router.post("/token/info", response_model=TokenInfo)
async def get_token_info(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    """
    Get information about a JWT token.

    This endpoint provides introspection capabilities for JWT tokens.
    """
    try:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token required"
            )

        token_info = jwt_manager.get_token_info(credentials.credentials)

        if "error" in token_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=token_info["error"]
            )

        return TokenInfo(
            user_id=token_info["user_id"],
            username=token_info.get("username"),
            token_type=token_info["token_type"],
            permissions=token_info.get("permissions", []),
            issued_at=token_info["issued_at"],
            expires_at=token_info["expires_at"],
            is_expired=token_info["is_expired"],
            is_active=not token_info["is_blacklisted"],
            jti=token_info.get("jti"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token info service unavailable",
        )


# Admin endpoints for user management


@auth_router.get("/users", response_model=List[AuthUser])
async def list_users(admin: AuthUser = Depends(require_admin())):
    """
    List all users (admin only).

    This endpoint returns a list of all users in the system.
    """
    users = []
    for user_data in USER_STORE.values():
        users.append(
            AuthUser(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data.get("email"),
                permissions=user_data.get("permissions", []),
                roles=user_data.get("roles", []),
                is_admin=user_data.get("is_admin", False),
                is_active=user_data.get("is_active", True),
                created_at=user_data.get("created_at", datetime.now(timezone.utc)),
            )
        )

    return users


@auth_router.post("/users", response_model=AuthUser)
async def create_user(
    user_request: UserCreateRequest, admin: AuthUser = Depends(require_admin())
):
    """
    Create a new user (admin only).

    This endpoint allows administrators to create new user accounts.
    """
    try:
        # Check if user already exists
        if get_user_by_username(user_request.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        # Create user
        user_id = f"user_{secrets.token_hex(8)}"
        password_hash = hash_password(user_request.password)

        user_data = {
            "user_id": user_id,
            "username": user_request.username,
            "email": user_request.email,
            "password_hash": password_hash,
            "permissions": user_request.permissions,
            "roles": user_request.roles,
            "is_admin": user_request.is_admin,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }

        USER_STORE[user_request.username] = user_data

        # Create response user object (without password)
        user = AuthUser(
            user_id=user_id,
            username=user_request.username,
            email=user_request.email,
            permissions=user_request.permissions,
            roles=user_request.roles,
            is_admin=user_request.is_admin,
            is_active=True,
            created_at=user_data["created_at"],
        )

        logger.info(f"User created: {user_request.username} by admin {admin.username}")

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation service unavailable",
        )


@auth_router.delete("/users/{username}")
async def delete_user(username: str, admin: AuthUser = Depends(require_admin())):
    """
    Delete a user (admin only).

    This endpoint allows administrators to delete user accounts.
    """
    try:
        # Prevent admin from deleting themselves
        if username.lower() == admin.username.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account",
            )

        # Check if user exists
        if username.lower() not in USER_STORE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Delete user
        del USER_STORE[username.lower()]

        logger.info(f"User deleted: {username} by admin {admin.username}")

        return {
            "message": f"User '{username}' deleted successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deletion service unavailable",
        )


# Add router to be imported by main application
__all__ = ["auth_router"]
