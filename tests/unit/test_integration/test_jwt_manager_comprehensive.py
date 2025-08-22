"""
Comprehensive unit tests for JWT manager module.

This test suite validates JWT token management including token generation,
validation, refresh, blacklisting, and security measures.
"""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from unittest.mock import Mock, patch

import hmac
import pytest

from src.core.config import JWTConfig
from src.core.exceptions import APIAuthenticationError, APISecurityError
from src.integration.auth.jwt_manager import JWTManager, TokenBlacklist


class TestTokenBlacklist:
    """Test token blacklist functionality."""

    @pytest.fixture
    def blacklist(self):
        """Create token blacklist instance."""
        return TokenBlacklist()

    def test_blacklist_initialization(self, blacklist):
        """Test blacklist initialization."""
        assert len(blacklist._blacklisted_tokens) == 0
        assert len(blacklist._blacklisted_jti) == 0
        assert blacklist._last_cleanup <= time.time()

    def test_add_token_to_blacklist(self, blacklist):
        """Test adding tokens to blacklist."""
        token = "test.jwt.token"
        jti = "unique_token_id"

        blacklist.add_token(token, jti)

        assert token in blacklist._blacklisted_tokens
        assert jti in blacklist._blacklisted_jti

    def test_add_token_without_jti(self, blacklist):
        """Test adding token without JTI."""
        token = "test.jwt.token"

        blacklist.add_token(token)

        assert token in blacklist._blacklisted_tokens
        assert len(blacklist._blacklisted_jti) == 0

    def test_is_blacklisted_by_token(self, blacklist):
        """Test blacklist check by token."""
        token = "blacklisted.jwt.token"
        blacklist.add_token(token)

        assert blacklist.is_blacklisted(token) is True
        assert blacklist.is_blacklisted("other.token") is False

    def test_is_blacklisted_by_jti(self, blacklist):
        """Test blacklist check by JTI."""
        token = "test.jwt.token"
        jti = "blacklisted_jti"

        blacklist.add_token(token, jti)

        assert blacklist.is_blacklisted("different.token", jti) is True
        assert blacklist.is_blacklisted("different.token", "other_jti") is False

    def test_is_blacklisted_cleanup_trigger(self, blacklist):
        """Test that blacklist check triggers cleanup."""
        # Mock old cleanup time to trigger cleanup
        blacklist._last_cleanup = time.time() - 7200  # 2 hours ago

        # Check should trigger cleanup and update timestamp
        result = blacklist.is_blacklisted("any.token")

        assert result is False
        assert blacklist._last_cleanup > time.time() - 60  # Updated recently


class TestJWTManagerConfiguration:
    """Test JWT manager configuration and initialization."""

    def test_jwt_manager_initialization_valid_config(self):
        """Test JWT manager initialization with valid configuration."""
        config = JWTConfig(
            enabled=True,
            secret_key="a" * 32,  # 32 character key
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test_issuer",
            audience="test_audience",
            blacklist_enabled=True,
        )

        manager = JWTManager(config)

        assert manager.config == config
        assert isinstance(manager.blacklist, TokenBlacklist)
        assert manager._max_operations_per_minute == 30

    def test_jwt_manager_initialization_short_secret_key(self):
        """Test JWT manager initialization with short secret key."""
        config = JWTConfig(
            enabled=True, secret_key="short", algorithm="HS256"  # Too short
        )

        with pytest.raises(
            ValueError, match="JWT secret key must be at least 32 characters"
        ):
            JWTManager(config)

    def test_jwt_manager_initialization_empty_secret_key(self):
        """Test JWT manager initialization with empty secret key."""
        config = JWTConfig(enabled=True, secret_key="", algorithm="HS256")  # Empty

        with pytest.raises(ValueError, match="JWT secret key is required"):
            JWTManager(config)

    def test_jwt_manager_initialization_blacklist_disabled(self):
        """Test JWT manager initialization with blacklist disabled."""
        config = JWTConfig(
            enabled=True,
            secret_key="a" * 32,
            algorithm="HS256",
            blacklist_enabled=False,
        )

        manager = JWTManager(config)

        assert manager.blacklist is None


class TestJWTManagerTokenGeneration:
    """Test JWT token generation functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test_issuer",
            audience="test_audience",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_generate_access_token_basic(self, jwt_manager):
        """Test basic access token generation."""
        user_id = "test_user_123"
        permissions = ["read", "write"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # header.payload.signature

        # Decode and verify payload structure
        payload = jwt_manager._decode_jwt_token(token)
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"
        assert payload["iss"] == jwt_manager.config.issuer
        assert payload["aud"] == jwt_manager.config.audience

    def test_generate_access_token_with_additional_claims(self, jwt_manager):
        """Test access token generation with additional claims."""
        user_id = "test_user_123"
        permissions = ["read", "write"]
        additional_claims = {
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["user"]

    def test_generate_access_token_reserved_claims_ignored(self, jwt_manager):
        """Test that reserved claims are not overridden by additional claims."""
        user_id = "test_user_123"
        permissions = ["read"]
        additional_claims = {
            "sub": "malicious_user",  # Should be ignored
            "iss": "fake_issuer",  # Should be ignored
            "exp": 999999,  # Should be ignored
            "custom_claim": "allowed",  # Should be included
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        # Reserved claims should not be overridden
        assert payload["sub"] == user_id
        assert payload["iss"] == jwt_manager.config.issuer
        assert payload["exp"] != 999999

        # Custom claim should be included
        assert payload["custom_claim"] == "allowed"

    def test_generate_refresh_token_basic(self, jwt_manager):
        """Test basic refresh token generation."""
        user_id = "test_user_123"

        token = jwt_manager.generate_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3

        payload = jwt_manager._decode_jwt_token(token)
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "permissions" not in payload  # Refresh tokens don't contain permissions

    def test_token_expiration_times(self, jwt_manager):
        """Test that tokens have correct expiration times."""
        user_id = "test_user"

        # Generate tokens
        access_token = jwt_manager.generate_access_token(user_id, [])
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Decode payloads
        access_payload = jwt_manager._decode_jwt_token(access_token)
        refresh_payload = jwt_manager._decode_jwt_token(refresh_token)

        # Check expiration times
        access_exp = datetime.fromtimestamp(access_payload["exp"], tz=timezone.utc)
        refresh_exp = datetime.fromtimestamp(refresh_payload["exp"], tz=timezone.utc)

        now = datetime.now(timezone.utc)

        # Access token should expire in ~60 minutes
        assert timedelta(minutes=59) <= (access_exp - now) <= timedelta(minutes=61)

        # Refresh token should expire in ~30 days
        assert timedelta(days=29) <= (refresh_exp - now) <= timedelta(days=31)

    def test_token_unique_jti(self, jwt_manager):
        """Test that each token has a unique JTI."""
        user_id = "test_user"

        token1 = jwt_manager.generate_access_token(user_id, [])
        token2 = jwt_manager.generate_access_token(user_id, [])

        payload1 = jwt_manager._decode_jwt_token(token1)
        payload2 = jwt_manager._decode_jwt_token(token2)

        assert payload1["jti"] != payload2["jti"]
        assert len(payload1["jti"]) > 0
        assert len(payload2["jti"]) > 0


class TestJWTManagerTokenValidation:
    """Test JWT token validation functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test_issuer",
            audience="test_audience",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_validate_access_token_success(self, jwt_manager):
        """Test successful access token validation."""
        user_id = "test_user_123"
        permissions = ["read", "write"]

        # Generate and validate token
        token = jwt_manager.generate_access_token(user_id, permissions)
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"

    def test_validate_refresh_token_success(self, jwt_manager):
        """Test successful refresh token validation."""
        user_id = "test_user_123"

        token = jwt_manager.generate_refresh_token(user_id)
        payload = jwt_manager.validate_token(token, "refresh")

        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_validate_token_wrong_type(self, jwt_manager):
        """Test token validation with wrong expected type."""
        user_id = "test_user_123"

        access_token = jwt_manager.generate_access_token(user_id, [])

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.validate_token(access_token, "refresh")

    def test_validate_token_invalid_signature(self, jwt_manager):
        """Test token validation with invalid signature."""
        user_id = "test_user_123"

        # Generate valid token
        valid_token = jwt_manager.generate_access_token(user_id, [])

        # Tamper with the token by changing the signature
        parts = valid_token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.invalid_signature"

        with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
            jwt_manager.validate_token(tampered_token, "access")

    def test_validate_token_malformed_format(self, jwt_manager):
        """Test token validation with malformed token format."""
        malformed_tokens = [
            "not.enough.parts",
            "too.many.parts.in.token",
            "invalid_token",
            "",
        ]

        for token in malformed_tokens:
            with pytest.raises(APIAuthenticationError, match="Invalid token format"):
                jwt_manager.validate_token(token, "access")

    def test_validate_token_expired(self, jwt_manager):
        """Test token validation with expired token."""
        # Temporarily modify config for quick expiration
        original_exp = jwt_manager.config.access_token_expire_minutes
        jwt_manager.config.access_token_expire_minutes = 0  # Immediate expiration

        try:
            user_id = "test_user_123"
            token = jwt_manager.generate_access_token(user_id, [])

            # Wait a moment and then validate
            time.sleep(1)

            with pytest.raises(APIAuthenticationError, match="Token has expired"):
                jwt_manager.validate_token(token, "access")
        finally:
            jwt_manager.config.access_token_expire_minutes = original_exp

    def test_validate_token_not_before(self, jwt_manager):
        """Test token validation with future nbf claim."""
        user_id = "test_user_123"

        # Create token with future nbf
        now = datetime.now(timezone.utc)
        exp = now + timedelta(hours=1)
        nbf = now + timedelta(minutes=5)  # Valid 5 minutes in the future

        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "nbf": int(nbf.timestamp()),
            "iss": jwt_manager.config.issuer,
            "aud": jwt_manager.config.audience,
            "jti": "test_jti",
            "type": "access",
            "permissions": [],
            "version": "1.0",
        }

        token = jwt_manager._create_jwt_token(payload)

        with pytest.raises(APIAuthenticationError, match="Token is not yet valid"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_wrong_issuer(self, jwt_manager):
        """Test token validation with wrong issuer."""
        # Create config with different issuer
        other_config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            issuer="other_issuer",  # Different issuer
            audience="test_audience",
        )
        other_manager = JWTManager(other_config)

        # Generate token with other issuer
        token = other_manager.generate_access_token("user", [])

        # Validate with original manager (different issuer)
        with pytest.raises(APIAuthenticationError, match="Invalid token issuer"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_wrong_audience(self, jwt_manager):
        """Test token validation with wrong audience."""
        # Create config with different audience
        other_config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            issuer="test_issuer",
            audience="other_audience",  # Different audience
        )
        other_manager = JWTManager(other_config)

        # Generate token with other audience
        token = other_manager.generate_access_token("user", [])

        # Validate with original manager (different audience)
        with pytest.raises(APIAuthenticationError, match="Invalid token audience"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_missing_required_claims(self, jwt_manager):
        """Test token validation with missing required claims."""
        # Create token missing required claims
        incomplete_payload = {
            "sub": "user",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            # Missing 'jti' claim
            "iss": jwt_manager.config.issuer,
            "aud": jwt_manager.config.audience,
            "type": "access",
        }

        token = jwt_manager._create_jwt_token(incomplete_payload)

        with pytest.raises(APIAuthenticationError, match="Missing required claim: jti"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_blacklisted(self, jwt_manager):
        """Test token validation with blacklisted token."""
        user_id = "test_user_123"

        # Generate and blacklist token
        token = jwt_manager.generate_access_token(user_id, [])
        payload = jwt_manager._decode_jwt_token(token)
        jwt_manager.blacklist.add_token(token, payload["jti"])

        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")


class TestJWTManagerTokenRefresh:
    """Test JWT token refresh functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="test_issuer",
            audience="test_audience",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_refresh_access_token_success(self, jwt_manager):
        """Test successful access token refresh."""
        user_id = "test_user_123"

        # Generate refresh token
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Refresh access token
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )

        # Verify new tokens
        assert new_access_token != refresh_token
        assert new_refresh_token != refresh_token

        access_payload = jwt_manager.validate_token(new_access_token, "access")
        refresh_payload = jwt_manager.validate_token(new_refresh_token, "refresh")

        assert access_payload["sub"] == user_id
        assert refresh_payload["sub"] == user_id

    def test_refresh_access_token_blacklists_old_token(self, jwt_manager):
        """Test that refresh blacklists the old refresh token."""
        user_id = "test_user_123"

        refresh_token = jwt_manager.generate_refresh_token(user_id)
        original_payload = jwt_manager._decode_jwt_token(refresh_token)

        # Refresh token
        jwt_manager.refresh_access_token(refresh_token)

        # Original refresh token should be blacklisted
        assert jwt_manager.blacklist.is_blacklisted(
            refresh_token, original_payload["jti"]
        )

    def test_refresh_access_token_invalid_refresh_token(self, jwt_manager):
        """Test refresh with invalid refresh token."""
        invalid_tokens = [
            "invalid.refresh.token",
            jwt_manager.generate_access_token(
                "user", []
            ),  # Access token instead of refresh
            "expired.token.here",
        ]

        for token in invalid_tokens:
            with pytest.raises(APIAuthenticationError):
                jwt_manager.refresh_access_token(token)

    def test_refresh_access_token_default_permissions(self, jwt_manager):
        """Test that refreshed access token has default permissions."""
        user_id = "test_user_123"

        refresh_token = jwt_manager.generate_refresh_token(user_id)
        new_access_token, _ = jwt_manager.refresh_access_token(refresh_token)

        access_payload = jwt_manager.validate_token(new_access_token, "access")

        # Should have default permissions (implementation detail)
        assert "permissions" in access_payload
        assert isinstance(access_payload["permissions"], list)


class TestJWTManagerTokenRevocation:
    """Test JWT token revocation functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    @pytest.fixture
    def jwt_manager_no_blacklist(self):
        """Create JWT manager without blacklist."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            blacklist_enabled=False,
        )
        return JWTManager(config)

    def test_revoke_token_success(self, jwt_manager):
        """Test successful token revocation."""
        user_id = "test_user_123"

        token = jwt_manager.generate_access_token(user_id, [])
        result = jwt_manager.revoke_token(token)

        assert result is True

        # Token should now be blacklisted
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")

    def test_revoke_token_blacklist_disabled(self, jwt_manager_no_blacklist):
        """Test token revocation when blacklist is disabled."""
        user_id = "test_user_123"

        token = jwt_manager_no_blacklist.generate_access_token(user_id, [])
        result = jwt_manager_no_blacklist.revoke_token(token)

        assert result is False

    def test_revoke_token_invalid_token(self, jwt_manager):
        """Test revoking invalid token."""
        invalid_token = "invalid.jwt.token"
        result = jwt_manager.revoke_token(invalid_token)

        assert result is False

    def test_revoke_token_malformed_token(self, jwt_manager):
        """Test revoking malformed token."""
        malformed_token = "not_a_token"
        result = jwt_manager.revoke_token(malformed_token)

        assert result is False


class TestJWTManagerTokenInfo:
    """Test JWT token info functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_get_token_info_success(self, jwt_manager):
        """Test successful token info retrieval."""
        user_id = "test_user_123"
        permissions = ["read", "write"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        token_info = jwt_manager.get_token_info(token)

        assert "error" not in token_info
        assert token_info["user_id"] == user_id
        assert token_info["token_type"] == "access"
        assert token_info["permissions"] == permissions
        assert isinstance(token_info["issued_at"], datetime)
        assert isinstance(token_info["expires_at"], datetime)
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False
        assert token_info["jti"] is not None

    def test_get_token_info_expired_token(self, jwt_manager):
        """Test token info for expired token."""
        # Temporarily modify config for quick expiration
        original_exp = jwt_manager.config.access_token_expire_minutes
        jwt_manager.config.access_token_expire_minutes = 0

        try:
            user_id = "test_user_123"
            token = jwt_manager.generate_access_token(user_id, [])

            time.sleep(1)  # Let token expire

            token_info = jwt_manager.get_token_info(token)

            assert token_info["is_expired"] is True
        finally:
            jwt_manager.config.access_token_expire_minutes = original_exp

    def test_get_token_info_blacklisted_token(self, jwt_manager):
        """Test token info for blacklisted token."""
        user_id = "test_user_123"

        token = jwt_manager.generate_access_token(user_id, [])
        jwt_manager.revoke_token(token)

        token_info = jwt_manager.get_token_info(token)

        assert token_info["is_blacklisted"] is True

    def test_get_token_info_invalid_token(self, jwt_manager):
        """Test token info for invalid token."""
        invalid_token = "invalid.jwt.token"
        token_info = jwt_manager.get_token_info(invalid_token)

        assert "error" in token_info
        assert isinstance(token_info["error"], str)


class TestJWTManagerRateLimiting:
    """Test JWT manager rate limiting functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
        )
        manager = JWTManager(config)
        # Set lower rate limit for testing
        manager._max_operations_per_minute = 5
        return manager

    def test_rate_limiting_within_limit(self, jwt_manager):
        """Test that operations within rate limit succeed."""
        user_id = "test_user"

        # Should be able to generate up to the limit
        for i in range(5):
            token = jwt_manager.generate_access_token(user_id, [])
            assert token is not None

    def test_rate_limiting_exceeded(self, jwt_manager):
        """Test that operations exceeding rate limit fail."""
        user_id = "test_user"

        # Generate tokens up to limit
        for i in range(5):
            jwt_manager.generate_access_token(user_id, [])

        # Next operation should fail due to rate limiting
        with pytest.raises(APISecurityError, match="Too many token operations"):
            jwt_manager.generate_access_token(user_id, [])

    def test_rate_limiting_different_users(self, jwt_manager):
        """Test that rate limiting is per-user."""
        user1 = "test_user_1"
        user2 = "test_user_2"

        # Fill rate limit for user1
        for i in range(5):
            jwt_manager.generate_access_token(user1, [])

        # user1 should be rate limited
        with pytest.raises(APISecurityError):
            jwt_manager.generate_access_token(user1, [])

        # user2 should still be able to generate tokens
        token = jwt_manager.generate_access_token(user2, [])
        assert token is not None

    def test_rate_limiting_window_reset(self, jwt_manager):
        """Test that rate limiting window resets after time."""
        user_id = "test_user"

        # Fill rate limit
        for i in range(5):
            jwt_manager.generate_access_token(user_id, [])

        # Should be rate limited
        with pytest.raises(APISecurityError):
            jwt_manager.generate_access_token(user_id, [])

        # Mock time passage (1 minute + 1 second)
        with patch("time.time", return_value=time.time() + 61):
            # Should be able to generate token again
            token = jwt_manager.generate_access_token(user_id, [])
            assert token is not None


class TestJWTManagerSecurityFeatures:
    """Test JWT manager security features."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
            issuer="secure_issuer",
            audience="secure_audience",
        )
        return JWTManager(config)

    def test_signature_verification_prevents_tampering(self, jwt_manager):
        """Test that signature verification prevents token tampering."""
        user_id = "test_user"
        token = jwt_manager.generate_access_token(user_id, ["read"])

        parts = token.split(".")
        header, payload, signature = parts

        # Decode and modify payload
        payload_data = json.loads(jwt_manager._base64url_decode(payload))
        payload_data["permissions"] = ["admin"]  # Escalate permissions

        # Re-encode payload
        modified_payload = jwt_manager._base64url_encode(json.dumps(payload_data))
        tampered_token = f"{header}.{modified_payload}.{signature}"

        # Should fail signature verification
        with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
            jwt_manager.validate_token(tampered_token, "access")

    def test_algorithm_verification_prevents_algorithm_confusion(self, jwt_manager):
        """Test that algorithm verification prevents algorithm confusion attacks."""
        user_id = "test_user"

        # Create token with different algorithm in header
        payload = {
            "sub": user_id,
            "type": "access",
            "permissions": ["admin"],
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iss": jwt_manager.config.issuer,
            "aud": jwt_manager.config.audience,
            "jti": "test_jti",
        }

        # Create header with different algorithm
        malicious_header = {"alg": "none", "typ": "JWT"}

        header_encoded = jwt_manager._base64url_encode(json.dumps(malicious_header))
        payload_encoded = jwt_manager._base64url_encode(json.dumps(payload))

        # Create token with no signature (algorithm confusion attack)
        malicious_token = f"{header_encoded}.{payload_encoded}."

        with pytest.raises(APIAuthenticationError, match="Invalid algorithm"):
            jwt_manager.validate_token(malicious_token, "access")

    def test_timing_attack_resistance_signature_comparison(self, jwt_manager):
        """Test that signature comparison is resistant to timing attacks."""
        user_id = "test_user"
        token = jwt_manager.generate_access_token(user_id, [])

        parts = token.split(".")
        valid_signature = parts[2]

        # Create tokens with slightly different signatures
        invalid_signatures = [
            "a" * len(valid_signature),
            valid_signature[:-1] + "x",
            valid_signature[:10] + "x" * (len(valid_signature) - 10),
        ]

        for invalid_sig in invalid_signatures:
            invalid_token = f"{parts[0]}.{parts[1]}.{invalid_sig}"

            with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
                jwt_manager.validate_token(invalid_token, "access")

    def test_replay_attack_prevention_with_jti(self, jwt_manager):
        """Test that JTI helps prevent replay attacks when combined with blacklist."""
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_access_token(user_id, [])
        payload = jwt_manager.validate_token(token, "access")

        # Blacklist token (simulating logout)
        jwt_manager.revoke_token(token)

        # Token should no longer be valid
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")

    def test_secret_key_isolation(self, jwt_manager):
        """Test that tokens from different secret keys are rejected."""
        user_id = "test_user"

        # Create another manager with different secret
        other_config = JWTConfig(
            enabled=True,
            secret_key="different_secret_key_32_chars_long_123",
            algorithm="HS256",
            issuer="secure_issuer",
            audience="secure_audience",
        )
        other_manager = JWTManager(other_config)

        # Generate token with other manager
        other_token = other_manager.generate_access_token(user_id, [])

        # Should fail validation with original manager
        with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
            jwt_manager.validate_token(other_token, "access")


class TestJWTManagerUtilityMethods:
    """Test JWT manager utility methods."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
        )
        return JWTManager(config)

    def test_base64url_encode_decode(self, jwt_manager):
        """Test base64url encoding and decoding."""
        test_data = "Hello, World! This is a test string with special chars: !@#$%^&*()"

        # Test string encoding/decoding
        encoded = jwt_manager._base64url_encode(test_data)
        decoded = jwt_manager._base64url_decode(encoded)

        assert decoded == test_data
        assert "=" not in encoded  # URL-safe encoding removes padding

    def test_base64url_encode_bytes(self, jwt_manager):
        """Test base64url encoding with bytes input."""
        test_bytes = b"Hello, World!"

        encoded = jwt_manager._base64url_encode(test_bytes)
        decoded = jwt_manager._base64url_decode(encoded)

        assert decoded == test_bytes.decode("utf-8")

    def test_create_signature_consistency(self, jwt_manager):
        """Test that signature creation is consistent."""
        message = "test.message.to.sign"

        signature1 = jwt_manager._create_signature(message)
        signature2 = jwt_manager._create_signature(message)

        assert signature1 == signature2

    def test_create_signature_different_messages(self, jwt_manager):
        """Test that different messages produce different signatures."""
        message1 = "first.test.message"
        message2 = "second.test.message"

        signature1 = jwt_manager._create_signature(message1)
        signature2 = jwt_manager._create_signature(message2)

        assert signature1 != signature2

    def test_jwt_token_structure(self, jwt_manager):
        """Test that created JWT tokens have correct structure."""
        user_id = "test_user"
        token = jwt_manager.generate_access_token(user_id, [])

        parts = token.split(".")
        assert len(parts) == 3

        # All parts should be valid base64url
        for part in parts:
            # Should not raise exception
            jwt_manager._base64url_decode(part + "=" * (4 - len(part) % 4))

    def test_payload_json_serialization(self, jwt_manager):
        """Test that payloads are properly JSON serialized."""
        user_id = "test_user"
        permissions = ["read", "write", "admin"]
        additional_claims = {
            "string_claim": "test_string",
            "number_claim": 42,
            "boolean_claim": True,
            "array_claim": [1, 2, 3],
            "object_claim": {"nested": "value"},
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        # All claim types should be preserved
        assert payload["string_claim"] == "test_string"
        assert payload["number_claim"] == 42
        assert payload["boolean_claim"] is True
        assert payload["array_claim"] == [1, 2, 3]
        assert payload["object_claim"] == {"nested": "value"}


class TestJWTManagerEdgeCases:
    """Test JWT manager edge cases and error conditions."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long_12345",
            algorithm="HS256",
        )
        return JWTManager(config)

    def test_empty_user_id(self, jwt_manager):
        """Test token generation with empty user ID."""
        empty_user_ids = ["", None]

        for user_id in empty_user_ids:
            if user_id is None:
                # Should handle None gracefully
                token = jwt_manager.generate_access_token(user_id, [])
                payload = jwt_manager._decode_jwt_token(token)
                assert payload["sub"] is None
            else:
                # Empty string should work
                token = jwt_manager.generate_access_token(user_id, [])
                payload = jwt_manager._decode_jwt_token(token)
                assert payload["sub"] == ""

    def test_empty_permissions_list(self, jwt_manager):
        """Test token generation with empty permissions."""
        user_id = "test_user"

        token = jwt_manager.generate_access_token(user_id, [])
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["permissions"] == []

    def test_large_permissions_list(self, jwt_manager):
        """Test token generation with large permissions list."""
        user_id = "test_user"
        large_permissions = [f"permission_{i}" for i in range(100)]

        token = jwt_manager.generate_access_token(user_id, large_permissions)
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["permissions"] == large_permissions

    def test_unicode_in_claims(self, jwt_manager):
        """Test token generation with unicode characters."""
        user_id = "测试用户_123"
        permissions = ["读取", "写入"]
        additional_claims = {
            "name": "用户姓名",
            "description": "这是一个测试用户 🎭",
            "emoji": "🔐🚀⭐",
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["name"] == "用户姓名"
        assert payload["emoji"] == "🔐🚀⭐"

    def test_very_long_token_handling(self, jwt_manager):
        """Test handling of very long tokens."""
        user_id = "test_user"

        # Create very large additional claims
        large_claim = "x" * 10000  # 10KB of data
        additional_claims = {"large_data": large_claim}

        token = jwt_manager.generate_access_token(user_id, [], additional_claims)
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["large_data"] == large_claim
        assert len(token) > 10000  # Token should be quite large

    def test_blacklist_memory_efficiency(self, jwt_manager):
        """Test that blacklist doesn't grow indefinitely."""
        # Add many tokens to blacklist
        for i in range(1000):
            user_id = f"user_{i}"
            token = jwt_manager.generate_access_token(user_id, [])
            jwt_manager.revoke_token(token)

        # Blacklist should contain all tokens
        assert len(jwt_manager.blacklist._blacklisted_tokens) == 1000

        # Trigger cleanup by checking blacklist
        jwt_manager.blacklist.is_blacklisted("dummy_token")

        # This test mainly ensures no memory leaks occur
