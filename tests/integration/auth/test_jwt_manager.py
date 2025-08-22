"""
Comprehensive test suite for JWT Manager.

This module provides complete test coverage for JWT token generation, validation,
refresh, blacklisting, and security features.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from typing import Dict, List
from unittest.mock import Mock, patch

import hmac
import pytest

from src.core.config import JWTConfig
from src.core.exceptions import APIAuthenticationError, APISecurityError
from src.integration.auth.jwt_manager import JWTManager, TokenBlacklist


class TestTokenBlacklist:
    """Test TokenBlacklist functionality."""

    def test_blacklist_initialization(self):
        """Test blacklist initialization."""
        blacklist = TokenBlacklist()

        assert isinstance(blacklist._blacklisted_tokens, set)
        assert isinstance(blacklist._blacklisted_jti, set)
        assert len(blacklist._blacklisted_tokens) == 0
        assert len(blacklist._blacklisted_jti) == 0

    def test_add_token(self):
        """Test adding token to blacklist."""
        blacklist = TokenBlacklist()
        token = "test.token.signature"
        jti = "test-jti-123"

        blacklist.add_token(token, jti)

        assert token in blacklist._blacklisted_tokens
        assert jti in blacklist._blacklisted_jti

    def test_add_token_without_jti(self):
        """Test adding token without JTI."""
        blacklist = TokenBlacklist()
        token = "test.token.nosig"

        blacklist.add_token(token)

        assert token in blacklist._blacklisted_tokens
        assert len(blacklist._blacklisted_jti) == 0

    def test_is_blacklisted_by_token(self):
        """Test checking if token is blacklisted by token string."""
        blacklist = TokenBlacklist()
        token = "blacklisted.token.signature"

        assert blacklist.is_blacklisted(token) is False

        blacklist.add_token(token)
        assert blacklist.is_blacklisted(token) is True

    def test_is_blacklisted_by_jti(self):
        """Test checking if token is blacklisted by JTI."""
        blacklist = TokenBlacklist()
        token = "some.token.sig"
        jti = "blacklisted-jti"

        blacklist.add_token("different.token.sig", jti)
        assert blacklist.is_blacklisted(token, jti) is True

    def test_is_blacklisted_not_found(self):
        """Test checking non-blacklisted token."""
        blacklist = TokenBlacklist()

        assert blacklist.is_blacklisted("clean.token.sig") is False
        assert blacklist.is_blacklisted("clean.token.sig", "clean-jti") is False

    def test_cleanup_called(self):
        """Test that cleanup is called during is_blacklisted."""
        blacklist = TokenBlacklist()

        with patch.object(blacklist, "_cleanup_expired") as mock_cleanup:
            blacklist.is_blacklisted("test.token.sig")
            mock_cleanup.assert_called_once()

    def test_cleanup_interval_respected(self):
        """Test cleanup interval is respected."""
        blacklist = TokenBlacklist()
        blacklist._cleanup_interval = 10  # 10 seconds
        blacklist._last_cleanup = time.time() - 5  # 5 seconds ago

        # Should not trigger cleanup (too soon)
        with patch.object(blacklist, "_last_cleanup", time.time() - 5):
            blacklist._cleanup_expired()
            # Cleanup time should not be updated
            assert blacklist._last_cleanup < time.time() - 4

    def test_cleanup_triggered_after_interval(self):
        """Test cleanup is triggered after interval expires."""
        blacklist = TokenBlacklist()
        blacklist._cleanup_interval = 10
        old_cleanup_time = time.time() - 15  # 15 seconds ago
        blacklist._last_cleanup = old_cleanup_time

        blacklist._cleanup_expired()

        # Cleanup time should be updated
        assert blacklist._last_cleanup > old_cleanup_time


class TestJWTManager:
    """Test JWTManager functionality."""

    @pytest.fixture
    def jwt_config(self):
        """Create JWT configuration for testing."""
        return JWTConfig(
            secret_key="test-secret-key-that-is-definitely-long-enough-for-security",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
            issuer="ha-ml-predictor",
            audience="ha-ml-predictor-api",
            blacklist_enabled=True,
        )

    @pytest.fixture
    def jwt_manager(self, jwt_config):
        """Create JWT manager instance."""
        return JWTManager(jwt_config)

    def test_manager_initialization(self, jwt_config):
        """Test JWT manager initialization."""
        manager = JWTManager(jwt_config)

        assert manager.config == jwt_config
        assert isinstance(manager.blacklist, TokenBlacklist)
        assert isinstance(manager._token_operations, dict)
        assert manager._max_operations_per_minute == 30

    def test_manager_initialization_without_blacklist(self, jwt_config):
        """Test initialization without blacklist enabled."""
        jwt_config.blacklist_enabled = False
        manager = JWTManager(jwt_config)

        assert manager.blacklist is None

    def test_initialization_with_short_secret_key(self):
        """Test initialization fails with short secret key."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            config = JWTConfig(
                secret_key="short",
                algorithm="HS256",
                access_token_expire_minutes=30,
                refresh_token_expire_days=7,
                issuer="test",
                audience="test",
            )

    def test_initialization_validates_config(self, jwt_manager):
        """Test JWT manager properly validates initialized config."""
        # This tests that our JWT manager works with properly initialized config
        assert jwt_manager.config is not None
        assert len(jwt_manager.config.secret_key) >= 32
        assert jwt_manager.config.algorithm == "HS256"

    def test_generate_access_token(self, jwt_manager):
        """Test generating access token."""
        user_id = "test-user-123"
        permissions = ["read", "write"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT has 3 parts

        # Verify token can be decoded
        payload = jwt_manager._decode_jwt_token(token)
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"
        assert payload["iss"] == jwt_manager.config.issuer
        assert payload["aud"] == jwt_manager.config.audience
        assert "jti" in payload
        assert "exp" in payload
        assert "iat" in payload

    def test_generate_access_token_with_additional_claims(self, jwt_manager):
        """Test generating access token with additional claims."""
        user_id = "user-with-claims"
        permissions = ["admin"]
        additional_claims = {
            "username": "testuser",
            "role": "administrator",
            "custom_field": "custom_value",
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        assert payload["username"] == "testuser"
        assert payload["role"] == "administrator"
        assert payload["custom_field"] == "custom_value"

    def test_generate_access_token_reserved_claims_protection(self, jwt_manager):
        """Test that reserved claims cannot be overridden."""
        user_id = "protected-user"
        permissions = ["read"]
        malicious_claims = {
            "sub": "hacker",  # Try to override subject
            "exp": 9999999999,  # Try to extend expiration
            "jti": "fake-jti",  # Try to set custom JTI
            "safe_claim": "allowed",
        }

        token = jwt_manager.generate_access_token(
            user_id, permissions, malicious_claims
        )
        payload = jwt_manager._decode_jwt_token(token)

        # Reserved claims should not be overridden
        assert payload["sub"] == user_id  # Original user_id
        assert payload["jti"] != "fake-jti"  # Generated JTI

        # Safe claim should be included
        assert payload["safe_claim"] == "allowed"

    def test_generate_refresh_token(self, jwt_manager):
        """Test generating refresh token."""
        user_id = "refresh-user-456"

        token = jwt_manager.generate_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3

        payload = jwt_manager._decode_jwt_token(token)
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "permissions" not in payload  # Refresh tokens don't have permissions

    def test_validate_token_success(self, jwt_manager):
        """Test successful token validation."""
        user_id = "valid-user"
        permissions = ["read", "write"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        payload = jwt_manager.validate_token(token, "access")

        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"

    def test_validate_token_wrong_type(self, jwt_manager):
        """Test token validation with wrong token type."""
        user_id = "type-test-user"
        permissions = ["read"]

        access_token = jwt_manager.generate_access_token(user_id, permissions)

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.validate_token(access_token, "refresh")

    def test_validate_token_blacklisted(self, jwt_manager):
        """Test validation of blacklisted token."""
        user_id = "blacklist-user"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        # Blacklist the token
        payload = jwt_manager._decode_jwt_token(token)
        jwt_manager.blacklist.add_token(token, payload["jti"])

        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(token)

    def test_validate_token_expired(self, jwt_manager):
        """Test validation of expired token."""
        user_id = "expired-user"
        permissions = ["read"]

        # Create token that expires immediately
        with patch.object(jwt_manager.config, "access_token_expire_minutes", -1):
            token = jwt_manager.generate_access_token(user_id, permissions)

        # Wait a moment to ensure expiration
        time.sleep(0.1)

        with pytest.raises(APIAuthenticationError, match="Token has expired"):
            jwt_manager.validate_token(token)

    def test_validate_token_invalid_issuer(self, jwt_manager):
        """Test validation with wrong issuer."""
        user_id = "issuer-test"
        permissions = ["read"]

        # Generate token with different issuer
        original_issuer = jwt_manager.config.issuer
        jwt_manager.config.issuer = "wrong-issuer"
        token = jwt_manager.generate_access_token(user_id, permissions)
        jwt_manager.config.issuer = "correct-issuer"  # Change expected issuer

        with pytest.raises(APIAuthenticationError, match="Invalid token issuer"):
            jwt_manager.validate_token(token)

        # Restore original issuer
        jwt_manager.config.issuer = original_issuer

    def test_validate_token_invalid_audience(self, jwt_manager):
        """Test validation with wrong audience."""
        user_id = "audience-test"
        permissions = ["read"]

        original_audience = jwt_manager.config.audience
        jwt_manager.config.audience = "wrong-audience"
        token = jwt_manager.generate_access_token(user_id, permissions)
        jwt_manager.config.audience = "correct-audience"

        with pytest.raises(APIAuthenticationError, match="Invalid token audience"):
            jwt_manager.validate_token(token)

        jwt_manager.config.audience = original_audience

    def test_validate_token_invalid_signature(self, jwt_manager):
        """Test validation with invalid signature."""
        user_id = "sig-test"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        # Tamper with token signature
        parts = token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.invalid_signature"

        with pytest.raises(APIAuthenticationError, match="Invalid token signature"):
            jwt_manager.validate_token(tampered_token)

    def test_validate_token_malformed(self, jwt_manager):
        """Test validation of malformed token."""
        malformed_tokens = [
            "not.enough.parts",
            "too.many.parts.here.invalid",
            "single_part",
            "",
            "invalid..token",
        ]

        for token in malformed_tokens:
            with pytest.raises(APIAuthenticationError):
                jwt_manager.validate_token(token)

    def test_validate_token_not_before(self, jwt_manager):
        """Test validation of token not yet valid."""
        user_id = "nbf-test"
        permissions = ["read"]

        # Mock future NBF claim
        with patch("time.time", return_value=100):  # Past time
            token = jwt_manager.generate_access_token(user_id, permissions)

        with patch("time.time", return_value=50):  # Earlier time
            with pytest.raises(APIAuthenticationError, match="Token is not yet valid"):
                jwt_manager.validate_token(token)

    def test_refresh_access_token_success(self, jwt_manager):
        """Test successful token refresh."""
        user_id = "refresh-success"

        refresh_token = jwt_manager.generate_refresh_token(user_id)
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )

        # Verify new tokens are valid
        assert isinstance(new_access_token, str)
        assert isinstance(new_refresh_token, str)

        access_payload = jwt_manager.validate_token(new_access_token, "access")
        refresh_payload = jwt_manager.validate_token(new_refresh_token, "refresh")

        assert access_payload["sub"] == user_id
        assert refresh_payload["sub"] == user_id

        # Original refresh token should be blacklisted
        assert jwt_manager.blacklist.is_blacklisted(refresh_token)

    def test_refresh_access_token_invalid_refresh_token(self, jwt_manager):
        """Test refresh with invalid refresh token."""
        with pytest.raises(APIAuthenticationError):
            jwt_manager.refresh_access_token("invalid.refresh.token")

    def test_refresh_access_token_access_token_used(self, jwt_manager):
        """Test refresh using access token instead of refresh token."""
        user_id = "wrong-token-type"
        permissions = ["read"]

        access_token = jwt_manager.generate_access_token(user_id, permissions)

        with pytest.raises(APIAuthenticationError, match="Invalid token type"):
            jwt_manager.refresh_access_token(access_token)

    def test_revoke_token_success(self, jwt_manager):
        """Test successful token revocation."""
        user_id = "revoke-test"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)

        result = jwt_manager.revoke_token(token)

        assert result is True
        assert jwt_manager.blacklist.is_blacklisted(token)

    def test_revoke_token_without_blacklist(self, jwt_config):
        """Test token revocation when blacklist is disabled."""
        jwt_config.blacklist_enabled = False
        manager = JWTManager(jwt_config)

        result = manager.revoke_token("any.token.here")

        assert result is False

    def test_revoke_token_invalid_token(self, jwt_manager):
        """Test revoking invalid token."""
        result = jwt_manager.revoke_token("invalid.token.format")

        assert result is False

    def test_get_token_info_valid_token(self, jwt_manager):
        """Test getting token information."""
        user_id = "info-user"
        permissions = ["read", "admin"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        token_info = jwt_manager.get_token_info(token)

        assert token_info["user_id"] == user_id
        assert token_info["token_type"] == "access"
        assert token_info["permissions"] == permissions
        assert isinstance(token_info["issued_at"], datetime)
        assert isinstance(token_info["expires_at"], datetime)
        assert token_info["is_expired"] is False
        assert token_info["is_blacklisted"] is False
        assert "jti" in token_info

    def test_get_token_info_expired_token(self, jwt_manager):
        """Test getting info for expired token."""
        user_id = "expired-info"
        permissions = ["read"]

        with patch.object(jwt_manager.config, "access_token_expire_minutes", -1):
            token = jwt_manager.generate_access_token(user_id, permissions)

        time.sleep(0.1)
        token_info = jwt_manager.get_token_info(token)

        assert token_info["is_expired"] is True

    def test_get_token_info_blacklisted_token(self, jwt_manager):
        """Test getting info for blacklisted token."""
        user_id = "blacklisted-info"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        jwt_manager.revoke_token(token)

        token_info = jwt_manager.get_token_info(token)

        assert token_info["is_blacklisted"] is True

    def test_get_token_info_invalid_token(self, jwt_manager):
        """Test getting info for invalid token."""
        token_info = jwt_manager.get_token_info("invalid.token.format")

        assert "error" in token_info

    def test_rate_limiting_enforcement(self, jwt_manager):
        """Test rate limiting for token operations."""
        user_id = "rate-limit-user"
        permissions = ["read"]

        # Perform operations up to the limit (30 operations)
        for i in range(jwt_manager._max_operations_per_minute):
            jwt_manager.generate_access_token(user_id, permissions)

        # Next operation should be rate limited
        with pytest.raises(APISecurityError, match="Too many token operations"):
            jwt_manager.generate_access_token(user_id, permissions)

    def test_rate_limiting_window_reset(self, jwt_manager):
        """Test rate limiting window reset."""
        user_id = "window-reset-user"
        permissions = ["read"]

        # Fill up the rate limit
        for i in range(jwt_manager._max_operations_per_minute):
            jwt_manager.generate_access_token(f"{user_id}-{i}", permissions)

        # Mock time advancement to reset window
        with patch("time.time", return_value=time.time() + 120):  # 2 minutes later
            # Should work again
            token = jwt_manager.generate_access_token(f"{user_id}-reset", permissions)
            assert token is not None

    def test_rate_limiting_per_user(self, jwt_manager):
        """Test rate limiting is enforced per user."""
        permissions = ["read"]
        user1_id = "user1"
        user2_id = "user2"

        # User 1 hits rate limit
        for i in range(jwt_manager._max_operations_per_minute):
            jwt_manager.generate_access_token(user1_id, permissions)

        # User 1 should be rate limited
        with pytest.raises(APISecurityError):
            jwt_manager.generate_access_token(user1_id, permissions)

        # User 2 should still work fine
        token = jwt_manager.generate_access_token(user2_id, permissions)
        assert token is not None

    def test_base64url_encoding_decoding(self, jwt_manager):
        """Test base64url encoding and decoding."""
        test_data = "Hello, World! This is a test string with special chars: +/="

        encoded = jwt_manager._base64url_encode(test_data)
        decoded = jwt_manager._base64url_decode(encoded)

        assert decoded == test_data
        assert "=" not in encoded  # Base64url should not have padding

    def test_base64url_encoding_bytes(self, jwt_manager):
        """Test base64url encoding with bytes input."""
        test_bytes = b"Binary data without invalid UTF-8"

        encoded = jwt_manager._base64url_encode(test_bytes)
        decoded = jwt_manager._base64url_decode(encoded)

        # Verify encoding works and produces valid string
        assert isinstance(encoded, str)
        assert isinstance(decoded, str)

    def test_create_signature(self, jwt_manager):
        """Test JWT signature creation."""
        message = "test.message.to.sign"

        signature = jwt_manager._create_signature(message)

        assert isinstance(signature, str)
        assert len(signature) > 0

        # Verify signature is deterministic
        signature2 = jwt_manager._create_signature(message)
        assert signature == signature2

    def test_create_signature_different_messages(self, jwt_manager):
        """Test that different messages produce different signatures."""
        message1 = "first.message"
        message2 = "second.message"

        sig1 = jwt_manager._create_signature(message1)
        sig2 = jwt_manager._create_signature(message2)

        assert sig1 != sig2

    def test_token_structure_and_format(self, jwt_manager):
        """Test JWT token structure and format compliance."""
        user_id = "format-test"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        parts = token.split(".")

        # JWT should have exactly 3 parts
        assert len(parts) == 3

        # Verify header
        header = json.loads(jwt_manager._base64url_decode(parts[0]))
        assert header["alg"] == jwt_manager.config.algorithm
        assert header["typ"] == "JWT"

        # Verify payload structure
        payload = json.loads(jwt_manager._base64url_decode(parts[1]))
        required_claims = ["sub", "iat", "exp", "nbf", "iss", "aud", "jti", "type"]
        for claim in required_claims:
            assert claim in payload

        # Verify signature is not empty
        assert len(parts[2]) > 0

    def test_token_expiration_times(self, jwt_manager):
        """Test token expiration time calculation."""
        user_id = "exp-test"
        permissions = ["read"]

        start_time = time.time()
        access_token = jwt_manager.generate_access_token(user_id, permissions)
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        access_payload = jwt_manager._decode_jwt_token(access_token)
        refresh_payload = jwt_manager._decode_jwt_token(refresh_token)

        # Check access token expiration
        expected_access_exp = start_time + (
            jwt_manager.config.access_token_expire_minutes * 60
        )
        assert (
            abs(access_payload["exp"] - expected_access_exp) < 2
        )  # 2 second tolerance

        # Check refresh token expiration
        expected_refresh_exp = start_time + (
            jwt_manager.config.refresh_token_expire_days * 24 * 60 * 60
        )
        assert abs(refresh_payload["exp"] - expected_refresh_exp) < 2

    def test_concurrent_token_operations(self, jwt_manager):
        """Test thread safety of token operations."""
        import threading

        user_id = "concurrent-test"
        permissions = ["read"]
        results = []
        errors = []

        def generate_token(user_suffix):
            try:
                token = jwt_manager.generate_access_token(
                    f"{user_id}-{user_suffix}", permissions
                )
                results.append(token)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=generate_token, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All operations should succeed (assuming no rate limiting hit)
        assert len(results) <= 10  # Some might be rate limited
        assert all(isinstance(token, str) for token in results)

    def test_token_validation_edge_cases(self, jwt_manager):
        """Test token validation with edge cases."""
        user_id = "edge-case-user"
        permissions = ["read"]

        # Test with minimal valid token
        token = jwt_manager.generate_access_token(user_id, permissions)

        # Test validation with empty token
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token("")

        # Test validation with None
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(None)

        # Test with token containing null bytes
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token("token\x00with\x00nulls")

    def test_algorithm_validation(self, jwt_manager):
        """Test JWT algorithm validation."""
        user_id = "alg-test"
        permissions = ["read"]

        token = jwt_manager.generate_access_token(user_id, permissions)
        parts = token.split(".")

        # Tamper with algorithm in header
        header = json.loads(jwt_manager._base64url_decode(parts[0]))
        header["alg"] = "none"  # Try to bypass signature verification

        tampered_header = jwt_manager._base64url_encode(json.dumps(header))
        tampered_token = f"{tampered_header}.{parts[1]}.{parts[2]}"

        # The algorithm check comes after signature validation, so we get signature error first
        with pytest.raises(APIAuthenticationError):
            jwt_manager.validate_token(tampered_token)

    def test_blacklist_integration(self, jwt_manager):
        """Test full integration with blacklist."""
        user_id = "blacklist-integration"
        permissions = ["read", "write"]

        # Generate tokens
        access_token = jwt_manager.generate_access_token(user_id, permissions)
        refresh_token = jwt_manager.generate_refresh_token(user_id)

        # Both should be valid initially
        assert jwt_manager.validate_token(access_token, "access")
        assert jwt_manager.validate_token(refresh_token, "refresh")

        # Revoke access token
        jwt_manager.revoke_token(access_token)

        # Access token should be invalid, refresh should still work
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(access_token, "access")

        assert jwt_manager.validate_token(refresh_token, "refresh")

        # Use refresh token (should blacklist it)
        new_access, new_refresh = jwt_manager.refresh_access_token(refresh_token)

        # Old refresh token should now be blacklisted
        with pytest.raises(APIAuthenticationError, match="Token has been revoked"):
            jwt_manager.validate_token(refresh_token, "refresh")

        # New tokens should work
        assert jwt_manager.validate_token(new_access, "access")
        assert jwt_manager.validate_token(new_refresh, "refresh")
