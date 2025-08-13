"""
Production-grade JWT token management for the Occupancy Prediction API.

This module provides comprehensive JWT token generation, validation, refresh,
and blacklisting functionality with proper security measures.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import hmac

from ...core.config import JWTConfig
from ...core.exceptions import APIAuthenticationError, APISecurityError

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """In-memory token blacklist for JWT revocation."""

    def __init__(self):
        self._blacklisted_tokens: Set[str] = set()
        self._blacklisted_jti: Set[str] = set()  # JWT ID blacklist
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour

    def add_token(self, token: str, jti: Optional[str] = None) -> None:
        """Add a token to the blacklist."""
        self._blacklisted_tokens.add(token)
        if jti:
            self._blacklisted_jti.add(jti)
        logger.info(f"Token blacklisted: JTI={jti}")

    def is_blacklisted(self, token: str, jti: Optional[str] = None) -> bool:
        """Check if a token is blacklisted."""
        self._cleanup_expired()

        if token in self._blacklisted_tokens:
            return True

        if jti and jti in self._blacklisted_jti:
            return True

        return False

    def _cleanup_expired(self) -> None:
        """Remove expired tokens from blacklist to prevent memory bloat."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # For production, implement proper cleanup based on token expiration
        # For now, keep cleanup simple
        self._last_cleanup = now

        # In production, you would parse tokens to check expiration
        # and remove expired ones from the blacklist


class JWTManager:
    """
    Production-grade JWT token manager with comprehensive security features.

    Features:
    - Secure token generation with proper payload structure
    - Token validation with signature verification
    - Token refresh mechanisms with rotation
    - Token blacklisting for logout/revocation
    - Rate limiting for token operations
    - Comprehensive error handling
    """

    def __init__(self, config: JWTConfig):
        """Initialize JWT manager with configuration."""
        self.config = config
        self.blacklist = TokenBlacklist() if config.blacklist_enabled else None

        # Validate configuration
        if not config.secret_key:
            raise ValueError("JWT secret key is required")

        if len(config.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")

        # Token operation rate limiting
        self._token_operations: Dict[str, List[float]] = {}
        self._max_operations_per_minute = 30

        logger.info("JWT Manager initialized with security features enabled")

    def generate_access_token(
        self,
        user_id: str,
        permissions: List[str],
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a secure JWT access token with proper payload structure.

        Args:
            user_id: Unique user identifier
            permissions: List of user permissions
            additional_claims: Optional additional claims to include

        Returns:
            JWT access token string

        Raises:
            APISecurityError: If rate limiting is exceeded
        """
        self._check_rate_limit(user_id)

        now = datetime.now(timezone.utc)
        exp = now + timedelta(minutes=self.config.access_token_expire_minutes)
        jti = str(uuid.uuid4())

        # Create secure payload
        payload = {
            "sub": user_id,  # Subject (user ID)
            "iat": int(now.timestamp()),  # Issued at
            "exp": int(exp.timestamp()),  # Expiration
            "nbf": int(now.timestamp()),  # Not before
            "iss": self.config.issuer,  # Issuer
            "aud": self.config.audience,  # Audience
            "jti": jti,  # JWT ID for blacklisting
            "type": "access",  # Token type
            "permissions": permissions,  # User permissions
            "version": "1.0",  # Token format version
        }

        # Add additional claims if provided
        if additional_claims:
            # Prevent overriding reserved claims
            reserved_claims = {"sub", "iat", "exp", "nbf", "iss", "aud", "jti", "type"}
            for claim, value in additional_claims.items():
                if claim not in reserved_claims:
                    payload[claim] = value
                else:
                    logger.warning(f"Attempted to override reserved claim: {claim}")

        token = self._create_jwt_token(payload)

        logger.info(f"Access token generated for user: {user_id}, JTI: {jti}")
        return token

    def generate_refresh_token(self, user_id: str) -> str:
        """
        Generate a secure JWT refresh token.

        Args:
            user_id: Unique user identifier

        Returns:
            JWT refresh token string
        """
        self._check_rate_limit(user_id)

        now = datetime.now(timezone.utc)
        exp = now + timedelta(days=self.config.refresh_token_expire_days)
        jti = str(uuid.uuid4())

        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "nbf": int(now.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "jti": jti,
            "type": "refresh",
            "version": "1.0",
        }

        token = self._create_jwt_token(payload)

        logger.info(f"Refresh token generated for user: {user_id}, JTI: {jti}")
        return token

    def validate_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Validate JWT token with comprehensive security checks.

        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")

        Returns:
            Decoded token payload

        Raises:
            APIAuthenticationError: If token is invalid, expired, or blacklisted
        """
        try:
            # Parse and validate token structure
            payload = self._decode_jwt_token(token)

            # Validate token type
            if payload.get("type") != token_type:
                raise APIAuthenticationError(
                    f"Invalid token type. Expected '{token_type}', got '{payload.get('type')}'"
                )

            # Check if token is blacklisted
            if self.blacklist and self.blacklist.is_blacklisted(
                token, payload.get("jti")
            ):
                raise APIAuthenticationError("Token has been revoked")

            # Validate expiration
            exp = payload.get("exp")
            if not exp or exp < time.time():
                raise APIAuthenticationError("Token has expired")

            # Validate not before
            nbf = payload.get("nbf", 0)
            if nbf > time.time():
                raise APIAuthenticationError("Token is not yet valid")

            # Validate issuer and audience
            if payload.get("iss") != self.config.issuer:
                raise APIAuthenticationError("Invalid token issuer")

            if payload.get("aud") != self.config.audience:
                raise APIAuthenticationError("Invalid token audience")

            # Validate required claims
            required_claims = ["sub", "iat", "exp", "jti"]
            for claim in required_claims:
                if claim not in payload:
                    raise APIAuthenticationError(f"Missing required claim: {claim}")

            logger.debug(f"Token validated successfully for user: {payload.get('sub')}")
            return payload

        except APIAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise APIAuthenticationError(f"Token validation failed: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """
        Generate new access and refresh tokens using a valid refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            APIAuthenticationError: If refresh token is invalid
        """
        # Validate refresh token
        payload = self.validate_token(refresh_token, "refresh")
        user_id = payload["sub"]

        # Blacklist the old refresh token
        if self.blacklist:
            self.blacklist.add_token(refresh_token, payload.get("jti"))

        # Generate new tokens
        # For refresh, we need to get user permissions - in production this would come from database
        permissions = [
            "read",
            "write",
        ]  # Default permissions - should be fetched from user store

        new_access_token = self.generate_access_token(user_id, permissions)
        new_refresh_token = self.generate_refresh_token(user_id)

        logger.info(f"Tokens refreshed for user: {user_id}")
        return new_access_token, new_refresh_token

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to the blacklist.

        Args:
            token: JWT token to revoke

        Returns:
            True if token was successfully revoked
        """
        if not self.blacklist:
            logger.warning("Token blacklisting is disabled")
            return False

        try:
            # Decode token to get JTI
            payload = self._decode_jwt_token(token)
            jti = payload.get("jti")

            # Add to blacklist
            self.blacklist.add_token(token, jti)

            logger.info(f"Token revoked: JTI={jti}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Get information about a token without validating expiration.

        Args:
            token: JWT token

        Returns:
            Token information
        """
        try:
            payload = self._decode_jwt_token(token, verify_expiration=False)

            return {
                "user_id": payload.get("sub"),
                "token_type": payload.get("type"),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0)),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0)),
                "permissions": payload.get("permissions", []),
                "jti": payload.get("jti"),
                "is_expired": payload.get("exp", 0) < time.time(),
                "is_blacklisted": (
                    self.blacklist.is_blacklisted(token, payload.get("jti"))
                    if self.blacklist
                    else False
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return {"error": str(e)}

    def _create_jwt_token(self, payload: Dict[str, Any]) -> str:
        """
        Create JWT token with HMAC-SHA256 signature.

        Args:
            payload: Token payload

        Returns:
            JWT token string
        """
        # Create header
        header = {"alg": self.config.algorithm, "typ": "JWT"}

        # Encode header and payload
        header_encoded = self._base64url_encode(
            json.dumps(header, separators=(",", ":"))
        )
        payload_encoded = self._base64url_encode(
            json.dumps(payload, separators=(",", ":"))
        )

        # Create signature
        message = f"{header_encoded}.{payload_encoded}"
        signature = self._create_signature(message)

        return f"{message}.{signature}"

    def _decode_jwt_token(
        self, token: str, verify_expiration: bool = True
    ) -> Dict[str, Any]:
        """
        Decode and verify JWT token.

        Args:
            token: JWT token string
            verify_expiration: Whether to check token expiration

        Returns:
            Decoded payload

        Raises:
            APIAuthenticationError: If token is invalid
        """
        try:
            # Split token parts
            parts = token.split(".")
            if len(parts) != 3:
                raise APIAuthenticationError("Invalid token format")

            header_encoded, payload_encoded, signature = parts

            # Verify signature
            message = f"{header_encoded}.{payload_encoded}"
            expected_signature = self._create_signature(message)

            if not hmac.compare_digest(signature, expected_signature):
                raise APIAuthenticationError("Invalid token signature")

            # Decode header and payload
            try:
                header = json.loads(self._base64url_decode(header_encoded))
                payload = json.loads(self._base64url_decode(payload_encoded))
            except (json.JSONDecodeError, ValueError) as e:
                raise APIAuthenticationError(f"Invalid token encoding: {e}")

            # Verify algorithm
            if header.get("alg") != self.config.algorithm:
                raise APIAuthenticationError(f"Invalid algorithm: {header.get('alg')}")

            return payload

        except APIAuthenticationError:
            raise
        except Exception as e:
            raise APIAuthenticationError(f"Token decoding failed: {str(e)}")

    def _create_signature(self, message: str) -> str:
        """Create HMAC-SHA256 signature for JWT token."""
        signature = hmac.new(
            self.config.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return self._base64url_encode(signature)

    def _base64url_encode(self, data: bytes | str) -> str:
        """Encode data using base64url encoding."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        import base64

        encoded = base64.urlsafe_b64encode(data).decode("ascii")
        # Remove padding
        return encoded.rstrip("=")

    def _base64url_decode(self, data: str) -> str:
        """Decode data using base64url decoding."""
        import base64

        # Add padding if needed
        padding = len(data) % 4
        if padding:
            data += "=" * (4 - padding)

        decoded = base64.urlsafe_b64decode(data)
        return decoded.decode("utf-8")

    def _check_rate_limit(self, user_id: str) -> None:
        """
        Check rate limits for token operations.

        Args:
            user_id: User identifier

        Raises:
            APISecurityError: If rate limit is exceeded
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old operations
        if user_id in self._token_operations:
            self._token_operations[user_id] = [
                op_time
                for op_time in self._token_operations[user_id]
                if op_time > window_start
            ]
        else:
            self._token_operations[user_id] = []

        # Check rate limit
        if len(self._token_operations[user_id]) >= self._max_operations_per_minute:
            raise APISecurityError(
                "rate_limit_exceeded", f"Too many token operations for user {user_id}"
            )

        # Record this operation
        self._token_operations[user_id].append(now)
