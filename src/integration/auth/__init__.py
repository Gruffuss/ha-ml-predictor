"""
Authentication module for the Home Assistant ML Predictor API.

This module provides comprehensive JWT-based authentication and authorization
functionality for the API endpoints.
"""

from .jwt_manager import JWTManager
from .auth_models import AuthUser, LoginRequest, LoginResponse, RefreshRequest
from .middleware import AuthenticationMiddleware
from .dependencies import get_current_user, require_permission
from .exceptions import AuthenticationError, AuthorizationError

__all__ = [
    "JWTManager",
    "AuthUser", 
    "LoginRequest",
    "LoginResponse",
    "RefreshRequest",
    "AuthenticationMiddleware",
    "get_current_user",
    "require_permission",
    "AuthenticationError",
    "AuthorizationError"
]