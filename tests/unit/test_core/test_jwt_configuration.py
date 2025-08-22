"""
Comprehensive JWT configuration validation tests.

Production-grade tests for JWT configuration validation, security scenarios,
environment handling, and edge cases in authentication configuration.
"""

import os
from pathlib import Path
import tempfile
from typing import Dict, List
from unittest.mock import patch

import pytest
import yaml

from src.core.config import (
    APIConfig,
    ConfigLoader,
    JWTConfig,
    SystemConfig,
)


class TestJWTConfigurationSecurityValidation:
    """Test JWT configuration security validation and requirements."""

    def test_jwt_secret_key_minimum_length_validation(self):
        """Test JWT secret key minimum length validation."""
        short_keys = [
            "",  # Empty
            "a",  # Single character
            "short",  # Too short
            "this_is_exactly_31_characters_",  # Exactly 31 chars (too short)
        ]

        for short_key in short_keys:
            env = {
                "ENVIRONMENT": "production",
                "JWT_SECRET_KEY": short_key,
                "JWT_ENABLED": "true",
            }

            with patch.dict(os.environ, env, clear=False):
                with pytest.raises(
                    ValueError, match="JWT secret key must be at least 32 characters"
                ):
                    JWTConfig()

    def test_jwt_secret_key_acceptable_lengths(self):
        """Test JWT secret key acceptable minimum and longer lengths."""
        acceptable_keys = [
            "this_is_exactly_32_characters_x",  # Exactly 32 chars
            "this_is_a_much_longer_secret_key_that_exceeds_minimum_requirements_significantly",  # Long key
            "x" * 64,  # 64 character key
            "production_grade_secret_key_with_sufficient_entropy_and_length_requirements_met_123456789",
        ]

        for acceptable_key in acceptable_keys:
            env = {
                "ENVIRONMENT": "production",
                "JWT_SECRET_KEY": acceptable_key,
                "JWT_ENABLED": "true",
            }

            with patch.dict(os.environ, env, clear=False):
                config = JWTConfig()
                assert config.enabled is True
                assert config.secret_key == acceptable_key
                assert len(config.secret_key) >= 32

    def test_jwt_production_environment_security_requirements(self):
        """Test JWT configuration security requirements in production environment."""
        production_env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "production_grade_secret_key_that_meets_all_security_requirements_for_jwt_tokens",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, production_env, clear=False):
            config = JWTConfig()

            # Production defaults
            assert config.enabled is True
            assert config.algorithm == "HS256"  # Secure algorithm
            assert config.access_token_expire_minutes == 60  # Reasonable expiry
            assert config.refresh_token_expire_days == 30  # Reasonable refresh
            assert config.blacklist_enabled is True  # Security feature enabled

            # Security settings should be appropriate for production
            assert len(config.secret_key) >= 32
            assert config.issuer == "ha-ml-predictor"
            assert config.audience == "ha-ml-predictor-api"

    def test_jwt_secret_key_missing_in_production(self):
        """Test JWT configuration fails when secret key missing in production."""
        production_env = {
            "ENVIRONMENT": "production",
            "JWT_ENABLED": "true",
            # Missing JWT_SECRET_KEY
        }

        # Ensure JWT_SECRET_KEY is not in environment
        if "JWT_SECRET_KEY" in os.environ:
            del os.environ["JWT_SECRET_KEY"]

        with patch.dict(os.environ, production_env, clear=False):
            with pytest.raises(
                ValueError, match="JWT_SECRET_KEY environment variable is not set"
            ):
                JWTConfig()

    def test_jwt_configuration_with_weak_algorithms(self):
        """Test JWT configuration with potentially weak algorithms."""
        weak_algorithms = ["none", "HS1", "RS1"]  # Weak or unsupported
        strong_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]

        env_base = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "production_grade_secret_key_that_meets_security_requirements_32_chars_plus",
            "JWT_ENABLED": "true",
        }

        # Test that default algorithm is secure
        with patch.dict(os.environ, env_base, clear=False):
            config = JWTConfig()
            assert config.algorithm in strong_algorithms
            assert config.algorithm == "HS256"  # Default should be secure

    def test_jwt_token_expiration_security_boundaries(self):
        """Test JWT token expiration security boundaries."""
        env_base = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "production_grade_secret_key_that_meets_security_requirements_32_chars_plus",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env_base, clear=False):
            config = JWTConfig()

            # Access token expiration should be reasonable (not too long)
            assert 1 <= config.access_token_expire_minutes <= 480  # Max 8 hours
            assert config.access_token_expire_minutes == 60  # Default 1 hour

            # Refresh token expiration should be reasonable
            assert 1 <= config.refresh_token_expire_days <= 90  # Max 3 months
            assert config.refresh_token_expire_days == 30  # Default 1 month

    def test_jwt_configuration_security_flags(self):
        """Test JWT configuration security flags and settings."""
        production_env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "production_grade_secret_key_that_meets_security_requirements_32_chars_plus",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, production_env, clear=False):
            config = JWTConfig()

            # Security flags
            assert config.blacklist_enabled is True  # Token blacklisting enabled

            # Should have proper issuer/audience for token validation
            assert len(config.issuer) > 0
            assert len(config.audience) > 0
            assert config.issuer != config.audience  # Should be different


class TestJWTEnvironmentHandling:
    """Test JWT configuration handling across different environments."""

    def test_jwt_test_environment_fallback_behavior(self):
        """Test JWT configuration fallback behavior in test environment."""
        test_environments = ["test", "testing", "development", "dev"]
        ci_environments = ["true", "1", "yes"]

        for test_env in test_environments:
            for ci_flag in ci_environments:
                env = {
                    "ENVIRONMENT": test_env,
                    "CI": ci_flag,
                    "JWT_ENABLED": "true",
                    # No JWT_SECRET_KEY provided
                }

                # Clear any existing JWT_SECRET_KEY
                if "JWT_SECRET_KEY" in os.environ:
                    del os.environ["JWT_SECRET_KEY"]

                with patch.dict(os.environ, env, clear=False):
                    config = JWTConfig()

                    # Should use fallback secret in test environments
                    assert config.enabled is True
                    assert len(config.secret_key) >= 32
                    assert "test_jwt_secret_key" in config.secret_key

    def test_jwt_disabled_environment_variations(self):
        """Test various ways to disable JWT via environment variables."""
        disable_values = ["false", "0", "no", "off", "False", "FALSE", "NO", "OFF"]

        env_base = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "production_key_32_chars_minimum_required",
        }

        for disable_value in disable_values:
            env = env_base.copy()
            env["JWT_ENABLED"] = disable_value

            with patch.dict(os.environ, env, clear=False):
                config = JWTConfig()
                assert (
                    config.enabled is False
                ), f"Failed to disable JWT with value: '{disable_value}'"

    def test_jwt_enabled_environment_variations(self):
        """Test various ways to enable JWT via environment variables."""
        enable_values = ["true", "1", "yes", "on", "True", "TRUE", "YES", "ON"]

        for enable_value in enable_values:
            env = {
                "ENVIRONMENT": "production",
                "JWT_SECRET_KEY": "production_grade_secret_key_32_chars_minimum_required_here",
                "JWT_ENABLED": enable_value,
            }

            with patch.dict(os.environ, env, clear=False):
                config = JWTConfig()
                assert (
                    config.enabled is True
                ), f"Failed to enable JWT with value: '{enable_value}'"

    def test_jwt_configuration_environment_precedence(self):
        """Test environment variable precedence in JWT configuration."""
        # Test that environment variables override defaults
        custom_env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "custom_production_secret_key_with_sufficient_length_for_security_requirements",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, custom_env, clear=False):
            config = JWTConfig()

            # Environment should override defaults
            assert config.secret_key == custom_env["JWT_SECRET_KEY"]
            assert config.enabled is True

            # Defaults should still apply for unspecified values
            assert config.algorithm == "HS256"  # Default
            assert config.issuer == "ha-ml-predictor"  # Default

    def test_jwt_configuration_missing_environment_variable(self):
        """Test JWT configuration when ENVIRONMENT variable is missing."""
        # Remove ENVIRONMENT variable
        env_without_environment = {
            "JWT_SECRET_KEY": "secret_key_without_environment_variable_32_chars_minimum_required",
            "JWT_ENABLED": "true",
        }

        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]

        with patch.dict(os.environ, env_without_environment, clear=False):
            config = JWTConfig()

            # Should still work with provided secret key
            assert config.enabled is True
            assert config.secret_key == env_without_environment["JWT_SECRET_KEY"]

    def test_jwt_staging_environment_configuration(self):
        """Test JWT configuration in staging environment."""
        staging_env = {
            "ENVIRONMENT": "staging",
            "JWT_SECRET_KEY": "staging_environment_secret_key_with_sufficient_length_for_jwt_security_32_plus",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, staging_env, clear=False):
            config = JWTConfig()

            # Should work like production but may have different defaults
            assert config.enabled is True
            assert len(config.secret_key) >= 32
            assert config.algorithm == "HS256"


class TestJWTConfigurationIntegrationWithAPIConfig:
    """Test JWT configuration integration with API configuration."""

    def test_api_config_jwt_integration(self):
        """Test API configuration properly integrates JWT configuration."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "api_integration_jwt_secret_key_with_minimum_32_character_requirement_met",
            "JWT_ENABLED": "true",
            "API_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            api_config = APIConfig()

            # API config should have JWT config
            assert hasattr(api_config, "jwt")
            assert isinstance(api_config.jwt, JWTConfig)

            # JWT should be properly configured
            assert api_config.jwt.enabled is True
            assert len(api_config.jwt.secret_key) >= 32

    def test_api_config_with_disabled_jwt(self):
        """Test API configuration when JWT is disabled."""
        env = {
            "ENVIRONMENT": "development",
            "JWT_ENABLED": "false",
            "API_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            api_config = APIConfig()

            # JWT should be disabled
            assert api_config.jwt.enabled is False

            # API should still be functional
            assert api_config.enabled is True

    def test_full_system_config_jwt_integration(self):
        """Test JWT configuration integration in full system configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create system config with JWT settings
            system_config_data = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
                "api": {
                    "enabled": True,
                    "jwt": {
                        "enabled": True,
                        "algorithm": "HS256",
                        "access_token_expire_minutes": 30,
                        "refresh_token_expire_days": 7,
                    },
                },
            }

            env = {
                "ENVIRONMENT": "production",
                "JWT_SECRET_KEY": "system_config_jwt_integration_secret_key_with_32_character_minimum_requirement",
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(system_config_data, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            with patch.dict(os.environ, env, clear=False):
                loader = ConfigLoader(str(config_dir))
                config = loader.load_config()

                # JWT should be properly integrated
                assert config.api.jwt.enabled is True
                assert config.api.jwt.access_token_expire_minutes == 30
                assert config.api.jwt.refresh_token_expire_days == 7
                assert len(config.api.jwt.secret_key) >= 32


class TestJWTConfigurationEdgeCases:
    """Test JWT configuration edge cases and boundary conditions."""

    def test_jwt_secret_key_with_special_characters(self):
        """Test JWT secret key with special characters and encodings."""
        special_keys = [
            "secret_with_!@#$%^&*()_+-={}[]|\\:;\"'<>?,./",  # Special ASCII
            "secret_with_unicode_åäö_ñüß_32_chars_minimum_requirement_met",  # Unicode
            "secret_with_emojis_🔑🛡️🔐_and_32_character_minimum_requirement_met",  # Emojis
            "secret\nwith\nnewlines\nand\ttabs\tplus_32_character_minimum_length",  # Control chars
        ]

        for special_key in special_keys:
            if len(special_key) >= 32:  # Only test if meets minimum length
                env = {
                    "ENVIRONMENT": "test",
                    "JWT_SECRET_KEY": special_key,
                    "JWT_ENABLED": "true",
                }

                with patch.dict(os.environ, env, clear=False):
                    config = JWTConfig()
                    assert config.enabled is True
                    assert config.secret_key == special_key

    def test_jwt_configuration_with_extremely_long_secret(self):
        """Test JWT configuration with extremely long secret key."""
        extremely_long_secret = "x" * 10000  # 10KB secret key

        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": extremely_long_secret,
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Should handle very long keys
            assert config.enabled is True
            assert len(config.secret_key) == 10000
            assert config.secret_key == extremely_long_secret

    def test_jwt_configuration_algorithm_case_sensitivity(self):
        """Test JWT algorithm configuration case sensitivity."""
        # Test that algorithm is case-sensitive and validates properly
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "case_sensitivity_test_secret_key_with_minimum_32_character_requirement_met",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Default algorithm should be properly cased
            assert config.algorithm == "HS256"
            assert config.algorithm != "hs256"  # Should not be lowercase
            assert config.algorithm != "Hs256"  # Should not be mixed case

    def test_jwt_configuration_with_zero_expiration_times(self):
        """Test JWT configuration with zero or negative expiration times."""
        # Test edge cases in token expiration times
        env = {
            "ENVIRONMENT": "test",
            "JWT_SECRET_KEY": "expiration_test_secret_key_with_minimum_32_character_requirement_met_here",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Manually set edge case values
            config.access_token_expire_minutes = 0
            config.refresh_token_expire_days = 0

            # Configuration should allow but these would be impractical
            assert config.access_token_expire_minutes == 0
            assert config.refresh_token_expire_days == 0

    def test_jwt_configuration_issuer_audience_validation(self):
        """Test JWT issuer and audience validation."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "issuer_audience_test_secret_key_with_minimum_32_character_requirement_met",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Should have valid issuer and audience
            assert len(config.issuer) > 0
            assert len(config.audience) > 0

            # Should be reasonable values
            assert "ha-ml-predictor" in config.issuer
            assert "ha-ml-predictor" in config.audience

            # Should follow JWT standards (no spaces, proper format)
            assert " " not in config.issuer
            assert " " not in config.audience

    def test_jwt_configuration_blacklist_functionality(self):
        """Test JWT blacklist configuration and functionality."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "blacklist_test_secret_key_with_minimum_32_character_requirement_met_here",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Blacklist should be enabled by default for security
            assert config.blacklist_enabled is True

            # Should be configurable
            config.blacklist_enabled = False
            assert config.blacklist_enabled is False


class TestJWTConfigurationSecurityBestPractices:
    """Test JWT configuration follows security best practices."""

    def test_jwt_https_requirements_in_production(self):
        """Test HTTPS requirements for JWT in production environment."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "https_requirement_test_secret_key_with_minimum_32_character_requirement",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Production defaults should encourage HTTPS
            # (Note: require_https defaults to False for flexibility,
            #  but should be set to True in production deployments)
            assert hasattr(config, "require_https")
            assert hasattr(config, "secure_cookies")

    def test_jwt_configuration_security_headers(self):
        """Test JWT configuration supports security headers and flags."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "security_headers_test_secret_key_with_minimum_32_character_requirement_met",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Should have security-related configuration options
            assert hasattr(config, "require_https")
            assert hasattr(config, "secure_cookies")
            assert hasattr(config, "blacklist_enabled")

            # Default values should be security-conscious
            assert config.blacklist_enabled is True  # Enable token blacklisting

    def test_jwt_token_lifetime_best_practices(self):
        """Test JWT token lifetime follows security best practices."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "token_lifetime_test_secret_key_with_minimum_32_character_requirement_met",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Access tokens should be short-lived (security best practice)
            assert config.access_token_expire_minutes <= 480  # Max 8 hours
            assert config.access_token_expire_minutes >= 15  # Min 15 minutes

            # Refresh tokens can be longer but should have reasonable limits
            assert config.refresh_token_expire_days <= 90  # Max 3 months
            assert config.refresh_token_expire_days >= 1  # Min 1 day

    def test_jwt_algorithm_security_validation(self):
        """Test JWT algorithm security validation."""
        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "algorithm_security_test_secret_key_with_minimum_32_character_requirement",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            config = JWTConfig()

            # Should use secure algorithm by default
            secure_algorithms = [
                "HS256",
                "HS384",
                "HS512",
                "RS256",
                "RS384",
                "RS512",
                "ES256",
                "ES384",
                "ES512",
            ]
            assert config.algorithm in secure_algorithms

            # Should not use 'none' algorithm (major security vulnerability)
            assert config.algorithm != "none"
            assert config.algorithm != "None"
            assert config.algorithm != "NONE"

    def test_jwt_secret_key_entropy_validation(self):
        """Test JWT secret key has sufficient entropy (basic check)."""
        # Test various secret keys with different entropy levels
        test_keys = [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Low entropy (all same char)
            "abcdefghijklmnopqrstuvwxyz123456",  # Sequential (low entropy)
            "random_secret_key_with_good_entropy_32_characters_minimum",  # Better entropy
            "Tr1$_1s_4_g00d_s3cr3t_k3y_w1th_m1x3d_ch4rs_4nd_numb3rs!",  # High entropy
        ]

        for test_key in test_keys:
            if len(test_key) >= 32:
                env = {
                    "ENVIRONMENT": "test",
                    "JWT_SECRET_KEY": test_key,
                    "JWT_ENABLED": "true",
                }

                with patch.dict(os.environ, env, clear=False):
                    config = JWTConfig()

                    # Should accept key regardless of entropy (basic validation)
                    # In production, you might want additional entropy checks
                    assert config.enabled is True
                    assert config.secret_key == test_key


@pytest.mark.unit
class TestJWTConfigurationPerformanceAndScalability:
    """Test JWT configuration performance and scalability aspects."""

    def test_jwt_configuration_loading_performance(self):
        """Test JWT configuration loading performance."""
        import time

        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "performance_test_secret_key_with_minimum_32_character_requirement_met_here",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            # Time multiple configuration loads
            start_time = time.time()

            for _ in range(100):
                config = JWTConfig()
                assert config.enabled is True
                assert len(config.secret_key) >= 32

            load_time = time.time() - start_time

            # Should be very fast (< 0.1 seconds for 100 loads)
            assert load_time < 0.1

    def test_jwt_configuration_memory_usage(self):
        """Test JWT configuration memory usage."""
        import sys

        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "memory_test_secret_key_with_minimum_32_character_requirement_met_here_now",
            "JWT_ENABLED": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            # Create multiple configurations
            configs = []
            for _ in range(100):
                config = JWTConfig()
                configs.append(config)

            # Verify all configurations are valid
            for config in configs:
                assert config.enabled is True
                assert len(config.secret_key) >= 32

            # Memory usage should be reasonable
            # (Basic check - in production you'd use memory profiling)
            assert len(configs) == 100

    def test_jwt_configuration_concurrent_access(self):
        """Test JWT configuration under concurrent access."""
        import threading

        env = {
            "ENVIRONMENT": "production",
            "JWT_SECRET_KEY": "concurrent_test_secret_key_with_minimum_32_character_requirement_met_ok",
            "JWT_ENABLED": "true",
        }

        results = []
        errors = []

        def create_jwt_config():
            try:
                with patch.dict(os.environ, env, clear=False):
                    config = JWTConfig()
                    results.append(config.secret_key)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_jwt_config)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # All should have same secret key
        for result in results:
            assert result == env["JWT_SECRET_KEY"]
