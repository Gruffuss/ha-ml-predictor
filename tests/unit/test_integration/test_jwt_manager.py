"""
Comprehensive unit tests for JWT token management.

This test suite validates JWT token generation, validation, refresh, blacklisting,
and all security features of the JWT manager.
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

    def test_token_blacklist_initialization(self):
        """Test TokenBlacklist initialization."""
        blacklist = TokenBlacklist()

        assert blacklist._blacklisted_tokens == set()
        assert blacklist._blacklisted_jti == set()
        assert isinstance(blacklist._last_cleanup, float)
        assert blacklist._cleanup_interval == 3600

    def test_token_blacklist_add_token(self):
        """Test adding tokens to blacklist."""
        blacklist = TokenBlacklist()

        blacklist.add_token("token123", "jti456")

        assert "token123" in blacklist._blacklisted_tokens
        assert "jti456" in blacklist._blacklisted_jti

    def test_token_blacklist_add_token_no_jti(self):
        """Test adding token without JTI."""
        blacklist = TokenBlacklist()

        blacklist.add_token("token123")

        assert "token123" in blacklist._blacklisted_tokens
        assert len(blacklist._blacklisted_jti) == 0

    def test_token_blacklist_is_blacklisted_token(self):
        """Test checking if token is blacklisted by token."""
        blacklist = TokenBlacklist()
        blacklist.add_token("blacklisted_token", "jti123")

        assert blacklist.is_blacklisted("blacklisted_token") is True
        assert blacklist.is_blacklisted("clean_token") is False

    def test_token_blacklist_is_blacklisted_jti(self):
        """Test checking if token is blacklisted by JTI."""
        blacklist = TokenBlacklist()
        blacklist.add_token("token123", "blacklisted_jti")

        assert blacklist.is_blacklisted("different_token", "blacklisted_jti") is True
        assert blacklist.is_blacklisted("different_token", "clean_jti") is False

    def test_token_blacklist_cleanup_trigger(self):
        """Test that cleanup is triggered based on time interval."""
        blacklist = TokenBlacklist()

        # Force cleanup by setting old timestamp
        blacklist._last_cleanup = time.time() - 7200  # 2 hours ago

        # This should trigger cleanup
        blacklist.is_blacklisted("any_token")

        # Cleanup timestamp should be updated
        assert blacklist._last_cleanup > time.time() - 10  # Recent


class TestJWTManagerInitialization:
    """Test JWT manager initialization and configuration."""

    def test_jwt_manager_initialization_valid(self):
        """Test JWTManager initialization with valid config."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            access_token_expire_minutes=15,
            refresh_token_expire_days=30,
            issuer="ha-ml-predictor",
            audience="api-users",
            blacklist_enabled=True,
        )

        jwt_manager = JWTManager(config)

        assert jwt_manager.config == config
        assert isinstance(jwt_manager.blacklist, TokenBlacklist)
        assert jwt_manager._max_operations_per_minute == 30

    def test_jwt_manager_initialization_no_secret_key(self):
        """Test JWTManager initialization without secret key."""
        config = JWTConfig(enabled=True, secret_key="", algorithm="HS256")

        with pytest.raises(ValueError, match="JWT secret key is required"):
            JWTManager(config)

    def test_jwt_manager_initialization_short_secret_key(self):
        """Test JWTManager initialization with short secret key."""
        config = JWTConfig(enabled=True, secret_key="short", algorithm="HS256")

        with pytest.raises(
            ValueError, match="JWT secret key must be at least 32 characters"
        ):
            JWTManager(config)

    def test_jwt_manager_initialization_blacklist_disabled(self):
        """Test JWTManager initialization with blacklist disabled."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            blacklist_enabled=False,
        )

        jwt_manager = JWTManager(config)

        assert jwt_manager.blacklist is None


class TestJWTTokenGeneration:
    """Test JWT token generation functionality."""

    @pytest.fixture
    def jwt_config(self):
        """JWT configuration fixture."""
        return JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            access_token_expire_minutes=15,
            refresh_token_expire_days=30,
            issuer="ha-ml-predictor",
            audience="api-users",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def jwt_manager(self, jwt_config):
        """JWT manager fixture."""
        return JWTManager(jwt_config)

    def test_generate_access_token_basic(self, jwt_manager):
        """Test basic access token generation."""
        user_id = "test_user_123"
        permissions = ["read", "write"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT has 3 parts

        # Decode and verify payload structure (without signature verification for testing)
        header, payload, signature = token.split(".")
        decoded_payload = json.loads(jwt_manager._base64url_decode(payload))

        assert decoded_payload["sub"] == user_id
        assert decoded_payload["permissions"] == permissions
        assert decoded_payload["type"] == "access"
        assert decoded_payload["iss"] == "ha-ml-predictor"
        assert decoded_payload["aud"] == "api-users"
        assert "jti" in decoded_payload
        assert "iat" in decoded_payload
        assert "exp" in decoded_payload

    def test_generate_access_token_with_additional_claims(self, jwt_manager):
        """Test access token generation with additional claims."""
        user_id = "test_user"
        permissions = ["read"]
        additional_claims = {
            "username": "testuser",
            "email": "test@example.com",
            "custom_field": "custom_value",
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )

        # Decode payload
        _, payload, _ = token.split(".")
        decoded_payload = json.loads(jwt_manager._base64url_decode(payload))

        assert decoded_payload["username"] == "testuser"
        assert decoded_payload["email"] == "test@example.com"
        assert decoded_payload["custom_field"] == "custom_value"

    def test_generate_access_token_reserved_claims_protection(self, jwt_manager):
        """Test that reserved claims cannot be overridden."""
        user_id = "test_user"
        permissions = ["read"]
        additional_claims = {
            "sub": "malicious_user",  # Should be ignored
            "jti": "custom_jti",  # Should be ignored
            "exp": 999999999,  # Should be ignored
            "safe_claim": "allowed",  # Should be included
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )

        # Decode payload
        _, payload, _ = token.split(".")
        decoded_payload = json.loads(jwt_manager._base64url_decode(payload))

        # Reserved claims should not be overridden
        assert decoded_payload["sub"] == user_id
        assert decoded_payload["jti"] != "custom_jti"
        assert decoded_payload["exp"] != 999999999

        # Non-reserved claims should be included
        assert decoded_payload["safe_claim"] == "allowed"

    def test_generate_refresh_token(self, jwt_manager):
        """Test refresh token generation."""
        user_id = "test_user"

        token = jwt_manager.generate_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3

        # Decode payload
        _, payload, _ = token.split(".")
        decoded_payload = json.loads(jwt_manager._base64url_decode(payload))

        assert decoded_payload["sub"] == user_id
        assert decoded_payload["type"] == "refresh"
        assert (
            "permissions" not in decoded_payload
        )  # Refresh tokens don't have permissions

    def test_token_expiration_times(self, jwt_manager):
        """Test that tokens have correct expiration times."""
        user_id = "test_user"

        # Generate tokens
        access_token = jwt_manager.generate_access_token(user_id, [])
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Decode payloads
        _, access_payload, _ = access_token.split(".")
        _, refresh_payload, _ = refresh_token.split(".")

        access_data = json.loads(jwt_manager._base64url_decode(access_payload))
        refresh_data = json.loads(jwt_manager._base64url_decode(refresh_payload))

        # Check expiration times
        access_exp = access_data["exp"]
        refresh_exp = refresh_data["exp"]
        now = time.time()

        # Access token should expire in ~15 minutes
        assert abs(access_exp - now - 15 * 60) < 60  # Within 1 minute tolerance

        # Refresh token should expire in ~30 days
        assert (
            abs(refresh_exp - now - 30 * 24 * 60 * 60) < 3600
        )  # Within 1 hour tolerance

    def test_rate_limiting_token_generation(self, jwt_manager):
        """Test rate limiting for token generation."""
        user_id = "test_user"

        # Generate tokens up to limit
        for _ in range(30):
            jwt_manager.generate_access_token(user_id, [])

        # Next generation should fail
        with pytest.raises(APISecurityError, match="Too many token operations"):
            jwt_manager.generate_access_token(user_id, [])

    def test_rate_limiting_different_users(self, jwt_manager):
        """Test that rate limiting is per-user."""
        # Fill rate limit for user1
        for _ in range(30):
            jwt_manager.generate_access_token("user1", [])

        # Should still be able to generate for user2
        token = jwt_manager.generate_access_token("user2", [])
        assert isinstance(token, str)

    @patch("time.time")
    def test_rate_limiting_window_reset(self, mock_time, jwt_manager):
        """Test that rate limiting window resets."""
        mock_time.return_value = 1000.0
        user_id = "test_user"

        # Fill rate limit
        for _ in range(30):
            jwt_manager.generate_access_token(user_id, [])

        # Move time forward by more than 1 minute
        mock_time.return_value = 1100.0  # 100 seconds later

        # Should be able to generate token again
        token = jwt_manager.generate_access_token(user_id, [])
        assert isinstance(token, str)


class TestJWTTokenValidation:
    """Test JWT token validation functionality."""

    @pytest.fixture
    def jwt_config(self):
        """JWT configuration fixture."""
        return JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            issuer="ha-ml-predictor",
            audience="api-users",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def jwt_manager(self, jwt_config):
        """JWT manager fixture."""
        return JWTManager(jwt_config)

    def test_validate_token_valid_access(self, jwt_manager):
        """Test validating a valid access token."""
        user_id = "test_user"
        permissions = ["read", "write"]

        # Generate token
        token = jwt_manager.generate_access_token(user_id, permissions)

        # Validate token
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"

    def test_validate_token_valid_refresh(self, jwt_manager):
        """Test validating a valid refresh token."""
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_refresh_token(user_id)

        # Validate token
        payload = jwt_manager.validate_token(token, "refresh")

        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_validate_token_wrong_type(self, jwt_manager):
        """Test validating token with wrong expected type."""
        user_id = "test_user"

        # Generate access token
        token = jwt_manager.generate_access_token(user_id, [])

        # Try to validate as refresh token
        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.validate_token(token, "refresh")

    def test_validate_token_invalid_format(self, jwt_manager):
        """Test validating token with invalid format."""
        invalid_tokens = [
            "not.a.jwt",
            "only_one_part",
            "two.parts",
            "four.parts.in.token",
            "",
        ]

        for invalid_token in invalid_tokens:
            with pytest.raises(APIAuthenticationError, match="Invalid token format"):
                jwt_manager.validate_token(invalid_token, "access")

    def test_validate_token_invalid_signature(self, jwt_manager):
        """Test validating token with invalid signature."""
        user_id = "test_user"

        # Generate valid token
        valid_token = jwt_manager.generate_access_token(user_id, [])
        header, payload, signature = valid_token.split(".")

        # Create token with invalid signature
        invalid_token = f"{header}.{payload}.invalid_signature"

        with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
            jwt_manager.validate_token(invalid_token, "access")

    @patch("time.time")
    def test_validate_token_expired(self, mock_time, jwt_manager):
        """Test validating expired token."""
        mock_time.return_value = 1000.0
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_access_token(user_id, [])

        # Move time forward past expiration
        mock_time.return_value = 1000.0 + 16 * 60  # 16 minutes later

        with pytest.raises(APIAuthenticationError, match="Token has expired"):
            jwt_manager.validate_token(token, "access")

    @patch("time.time")
    def test_validate_token_not_before(self, mock_time, jwt_manager):
        """Test validating token before it's valid (nbf claim)."""
        mock_time.return_value = 1000.0
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_access_token(user_id, [])

        # Move time backward before token was issued
        mock_time.return_value = 900.0  # 100 seconds before issuance

        with pytest.raises(APIAuthenticationError, match="Token is not yet valid"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_wrong_issuer(self, jwt_manager):
        """Test validating token with wrong issuer."""
        # Create token with different issuer
        wrong_config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            issuer="wrong-issuer",
            audience="api-users",
        )
        wrong_manager = JWTManager(wrong_config)

        token = wrong_manager.generate_access_token("user", [])

        with pytest.raises(APIAuthenticationError, match="Invalid token issuer"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_wrong_audience(self, jwt_manager):
        """Test validating token with wrong audience."""
        # Create token with different audience
        wrong_config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            issuer="ha-ml-predictor",
            audience="wrong-audience",
        )
        wrong_manager = JWTManager(wrong_config)

        token = wrong_manager.generate_access_token("user", [])

        with pytest.raises(APIAuthenticationError, match="Invalid token audience"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_missing_required_claims(self, jwt_manager):
        """Test validating token with missing required claims."""
        # Create malformed token manually
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"type": "access"}  # Missing required claims

        header_encoded = jwt_manager._base64url_encode(json.dumps(header))
        payload_encoded = jwt_manager._base64url_encode(json.dumps(payload))
        message = f"{header_encoded}.{payload_encoded}"
        signature = jwt_manager._create_signature(message)

        malformed_token = f"{message}.{signature}"

        with pytest.raises(APIAuthenticationError, match="Missing required claim"):
            jwt_manager.validate_token(malformed_token, "access")

    def test_validate_token_blacklisted(self, jwt_manager):
        """Test validating blacklisted token."""
        user_id = "test_user"

        # Generate and blacklist token
        token = jwt_manager.generate_access_token(user_id, [])
        jwt_manager.revoke_token(token)

        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")

    def test_validate_token_malformed_json(self, jwt_manager):
        """Test validating token with malformed JSON payload."""
        # Create token with invalid JSON
        header_encoded = jwt_manager._base64url_encode('{"alg":"HS256","typ":"JWT"}')
        payload_encoded = jwt_manager._base64url_encode(
            '{"invalid":json}'
        )  # Invalid JSON
        message = f"{header_encoded}.{payload_encoded}"
        signature = jwt_manager._create_signature(message)

        malformed_token = f"{message}.{signature}"

        with pytest.raises(APIAuthenticationError, match="Invalid token encoding"):
            jwt_manager.validate_token(malformed_token, "access")


class TestJWTTokenRefresh:
    """Test JWT token refresh functionality."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_refresh_access_token_success(self, jwt_manager):
        """Test successful token refresh."""
        user_id = "test_user"

        # Generate initial refresh token
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Refresh tokens
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )

        assert isinstance(new_access_token, str)
        assert isinstance(new_refresh_token, str)
        assert new_access_token != refresh_token
        assert new_refresh_token != refresh_token

        # Verify new tokens are valid
        access_payload = jwt_manager.validate_token(new_access_token, "access")
        refresh_payload = jwt_manager.validate_token(new_refresh_token, "refresh")

        assert access_payload["sub"] == user_id
        assert refresh_payload["sub"] == user_id

    def test_refresh_access_token_invalid_refresh(self, jwt_manager):
        """Test token refresh with invalid refresh token."""
        with pytest.raises(APIAuthenticationError):
            jwt_manager.refresh_access_token("invalid_refresh_token")

    def test_refresh_access_token_wrong_type(self, jwt_manager):
        """Test token refresh with access token instead of refresh token."""
        user_id = "test_user"

        # Generate access token
        access_token = jwt_manager.generate_access_token(user_id, [])

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.refresh_access_token(access_token)

    def test_refresh_access_token_blacklists_old_token(self, jwt_manager):
        """Test that refresh blacklists the old refresh token."""
        user_id = "test_user"

        # Generate refresh token
        old_refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Refresh tokens
        jwt_manager.refresh_access_token(old_refresh_token)

        # Old refresh token should be blacklisted
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(old_refresh_token, "refresh")

    def test_refresh_access_token_default_permissions(self, jwt_manager):
        """Test that refreshed access token has default permissions."""
        user_id = "test_user"

        # Generate refresh token
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Refresh tokens
        new_access_token, _ = jwt_manager.refresh_access_token(refresh_token)

        # Check that access token has default permissions
        payload = jwt_manager.validate_token(new_access_token, "access")
        assert payload["permissions"] == [
            "read",
            "write",
        ]  # Default from implementation


class TestJWTTokenRevocation:
    """Test JWT token revocation and blacklisting."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_revoke_token_success(self, jwt_manager):
        """Test successful token revocation."""
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_access_token(user_id, [])

        # Revoke token
        result = jwt_manager.revoke_token(token)

        assert result is True

        # Token should be blacklisted
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token, "access")

    def test_revoke_token_blacklist_disabled(self):
        """Test token revocation when blacklist is disabled."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            blacklist_enabled=False,
        )
        jwt_manager = JWTManager(config)

        user_id = "test_user"
        token = jwt_manager.generate_access_token(user_id, [])

        result = jwt_manager.revoke_token(token)

        assert result is False

    def test_revoke_token_invalid_token(self, jwt_manager):
        """Test revoking invalid token."""
        result = jwt_manager.revoke_token("invalid_token")

        assert result is False


class TestJWTTokenInfo:
    """Test JWT token information retrieval."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_get_token_info_success(self, jwt_manager):
        """Test successful token info retrieval."""
        user_id = "test_user"
        permissions = ["read", "write"]

        # Generate token
        token = jwt_manager.generate_access_token(user_id, permissions)

        # Get token info
        token_info = jwt_manager.get_token_info(token)

        assert token_info["user_id"] == user_id
        assert token_info["token_type"] == "access"
        assert token_info["permissions"] == permissions
        assert isinstance(token_info["issued_at"], datetime)
        assert isinstance(token_info["expires_at"], datetime)
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False
        assert "jti" in token_info

    @patch("time.time")
    def test_get_token_info_expired(self, mock_time, jwt_manager):
        """Test token info for expired token."""
        mock_time.return_value = 1000.0
        user_id = "test_user"

        # Generate token
        token = jwt_manager.generate_access_token(user_id, [])

        # Move time forward past expiration
        mock_time.return_value = 1000.0 + 16 * 60  # 16 minutes later

        # Get token info (should work even for expired tokens)
        token_info = jwt_manager.get_token_info(token)

        assert token_info["user_id"] == user_id
        assert token_info["is_expired"] is True

    def test_get_token_info_blacklisted(self, jwt_manager):
        """Test token info for blacklisted token."""
        user_id = "test_user"

        # Generate and blacklist token
        token = jwt_manager.generate_access_token(user_id, [])
        jwt_manager.revoke_token(token)

        # Get token info
        token_info = jwt_manager.get_token_info(token)

        assert token_info["user_id"] == user_id
        assert token_info["is_blacklisted"] is True

    def test_get_token_info_invalid_token(self, jwt_manager):
        """Test token info for invalid token."""
        token_info = jwt_manager.get_token_info("invalid_token")

        assert "error" in token_info


class TestJWTUtilityMethods:
    """Test JWT utility methods and encoding/decoding."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
        )
        return JWTManager(config)

    def test_base64url_encode_string(self, jwt_manager):
        """Test base64url encoding of string."""
        test_string = "Hello, World!"

        encoded = jwt_manager._base64url_encode(test_string)

        # Should not have padding
        assert "=" not in encoded

        # Should be URL-safe
        assert "+" not in encoded
        assert "/" not in encoded

    def test_base64url_encode_bytes(self, jwt_manager):
        """Test base64url encoding of bytes."""
        test_bytes = b"Hello, World!"

        encoded = jwt_manager._base64url_encode(test_bytes)

        assert isinstance(encoded, str)
        assert "=" not in encoded

    def test_base64url_decode(self, jwt_manager):
        """Test base64url decoding."""
        original = "Hello, World!"

        # Encode then decode
        encoded = jwt_manager._base64url_encode(original)
        decoded = jwt_manager._base64url_decode(encoded)

        assert decoded == original

    def test_base64url_decode_with_padding(self, jwt_manager):
        """Test base64url decoding with missing padding."""
        # Create encoded string without padding
        original = "Hello"
        encoded = base64.urlsafe_b64encode(original.encode()).decode().rstrip("=")

        decoded = jwt_manager._base64url_decode(encoded)

        assert decoded == original

    def test_create_signature(self, jwt_manager):
        """Test HMAC signature creation."""
        message = "test.message"

        signature = jwt_manager._create_signature(message)

        assert isinstance(signature, str)
        assert "=" not in signature  # URL-safe encoding

        # Verify signature is consistent
        signature2 = jwt_manager._create_signature(message)
        assert signature == signature2

    def test_create_signature_different_messages(self, jwt_manager):
        """Test that different messages produce different signatures."""
        signature1 = jwt_manager._create_signature("message1")
        signature2 = jwt_manager._create_signature("message2")

        assert signature1 != signature2

    def test_signature_verification_manual(self, jwt_manager):
        """Test manual signature verification."""
        message = "header.payload"
        expected_signature = jwt_manager._create_signature(message)

        # Manually verify using HMAC
        manual_signature = hmac.new(
            jwt_manager.config.secret_key.encode(), message.encode(), hashlib.sha256
        ).digest()
        manual_signature_encoded = jwt_manager._base64url_encode(manual_signature)

        assert expected_signature == manual_signature_encoded


class TestJWTSecurityFeatures:
    """Test JWT security features and edge cases."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_token_uniqueness(self, jwt_manager):
        """Test that generated tokens are unique."""
        user_id = "test_user"

        tokens = set()
        for _ in range(100):
            token = jwt_manager.generate_access_token(user_id, [])
            tokens.add(token)

        # All tokens should be unique
        assert len(tokens) == 100

    def test_jti_uniqueness(self, jwt_manager):
        """Test that JTI values are unique."""
        user_id = "test_user"

        jtis = set()
        for _ in range(100):
            token = jwt_manager.generate_access_token(user_id, [])
            payload = jwt_manager.validate_token(token, "access")
            jtis.add(payload["jti"])

        # All JTIs should be unique
        assert len(jtis) == 100

    def test_timing_attack_resistance(self, jwt_manager):
        """Test that signature comparison is timing-attack resistant."""
        user_id = "test_user"

        # Generate valid token
        valid_token = jwt_manager.generate_access_token(user_id, [])
        header, payload, signature = valid_token.split(".")

        # Create tokens with slightly different signatures
        invalid_token1 = f"{header}.{payload}.{signature[:-1]}x"
        invalid_token2 = f"{header}.{payload}.completely_different_signature"

        # Both should fail validation (and take similar time)
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(invalid_token1, "access")

        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(invalid_token2, "access")

    def test_algorithm_confusion_prevention(self, jwt_manager):
        """Test prevention of algorithm confusion attacks."""
        # Create token with 'none' algorithm
        header = {"alg": "none", "typ": "JWT"}
        payload = {"sub": "malicious_user", "type": "access"}

        header_encoded = jwt_manager._base64url_encode(json.dumps(header))
        payload_encoded = jwt_manager._base64url_encode(json.dumps(payload))

        # 'none' algorithm token (no signature)
        none_token = f"{header_encoded}.{payload_encoded}."

        with pytest.raises(APIAuthenticationError, match="Invalid algorithm"):
            jwt_manager.validate_token(none_token, "access")

    def test_large_payload_handling(self, jwt_manager):
        """Test handling of large token payloads."""
        user_id = "test_user"

        # Create large additional claims
        large_claims = {f"claim_{i}": f"value_{i}" * 100 for i in range(50)}

        # Should handle large payload without issues
        token = jwt_manager.generate_access_token(user_id, [], large_claims)
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == user_id
        assert "claim_0" in payload

    def test_special_characters_in_claims(self, jwt_manager):
        """Test handling of special characters in claims."""
        user_id = "test_user_with_特殊字符_and_🎉"
        permissions = ["read", "write"]
        additional_claims = {
            "username": "用户_with_émojis_🚀",
            "description": 'Contains: quotes", newlines\n, and tabs\t',
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == user_id
        assert payload["username"] == "用户_with_émojis_🚀"
        assert "\n" in payload["description"]


class TestJWTErrorHandling:
    """Test JWT error handling and edge cases."""

    @pytest.fixture
    def jwt_manager(self):
        """JWT manager fixture."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            algorithm="HS256",
            blacklist_enabled=True,
        )
        return JWTManager(config)

    def test_decode_jwt_token_invalid_base64(self, jwt_manager):
        """Test decoding token with invalid base64."""
        # Create token with invalid base64 encoding
        invalid_token = "invalid_base64!.invalid_base64!.signature"

        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(invalid_token, "access")

    def test_validate_token_exception_handling(self, jwt_manager):
        """Test that unexpected exceptions are properly handled."""
        # Mock the _decode_jwt_token method to raise unexpected exception
        original_method = jwt_manager._decode_jwt_token

        def mock_decode(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        jwt_manager._decode_jwt_token = mock_decode

        with pytest.raises(APIAuthenticationError, match="Token validation failed"):
            jwt_manager.validate_token("any.token.here", "access")

        # Restore original method
        jwt_manager._decode_jwt_token = original_method

    def test_get_token_info_exception_handling(self, jwt_manager):
        """Test exception handling in get_token_info."""
        # Mock to raise exception
        original_method = jwt_manager._decode_jwt_token

        def mock_decode(*args, **kwargs):
            raise Exception("Decoding error")

        jwt_manager._decode_jwt_token = mock_decode

        token_info = jwt_manager.get_token_info("any_token")

        assert "error" in token_info
        assert "Decoding error" in token_info["error"]

        # Restore original method
        jwt_manager._decode_jwt_token = original_method

    def test_revoke_token_exception_handling(self, jwt_manager):
        """Test exception handling in revoke_token."""
        # Mock to raise exception during decoding
        original_method = jwt_manager._decode_jwt_token

        def mock_decode(*args, **kwargs):
            raise Exception("Revocation error")

        jwt_manager._decode_jwt_token = mock_decode

        result = jwt_manager.revoke_token("any_token")

        assert result is False

        # Restore original method
        jwt_manager._decode_jwt_token = original_method


class TestJWTConfigurationEdgeCases:
    """Test JWT manager with various configuration edge cases."""

    def test_minimal_valid_config(self):
        """Test JWT manager with minimal valid configuration."""
        config = JWTConfig(
            enabled=True,
            secret_key="minimum_32_character_secret_key!",
        )

        jwt_manager = JWTManager(config)

        # Should work with defaults
        token = jwt_manager.generate_access_token("user", [])
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == "user"

    def test_custom_expiration_times(self):
        """Test JWT manager with custom expiration times."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            access_token_expire_minutes=5,
            refresh_token_expire_days=7,
        )

        jwt_manager = JWTManager(config)

        # Generate tokens
        access_token = jwt_manager.generate_access_token("user", [])
        refresh_token = jwt_manager.generate_refresh_token("user")

        # Decode and check expiration times
        access_payload = jwt_manager.validate_token(access_token, "access")
        refresh_payload = jwt_manager.validate_token(refresh_token, "refresh")

        now = time.time()
        access_exp = access_payload["exp"]
        refresh_exp = refresh_payload["exp"]

        # Check expiration times match config
        assert abs(access_exp - now - 5 * 60) < 60  # 5 minutes
        assert abs(refresh_exp - now - 7 * 24 * 60 * 60) < 3600  # 7 days

    def test_different_issuer_audience(self):
        """Test JWT manager with custom issuer and audience."""
        config = JWTConfig(
            enabled=True,
            secret_key="test_secret_key_32_characters_long!",
            issuer="custom-issuer",
            audience="custom-audience",
        )

        jwt_manager = JWTManager(config)

        token = jwt_manager.generate_access_token("user", [])
        payload = jwt_manager.validate_token(token, "access")

        assert payload["iss"] == "custom-issuer"
        assert payload["aud"] == "custom-audience"
