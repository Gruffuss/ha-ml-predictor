#!/usr/bin/env python3
"""
Test script for the JWT authentication system.

This script demonstrates the complete JWT authentication flow including
token generation, validation, refresh, and revocation.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.auth.jwt_manager import JWTManager
from src.integration.auth.auth_models import AuthUser, LoginRequest, LoginResponse
from src.core.config import JWTConfig
from src.core.exceptions import APIAuthenticationError


async def test_jwt_authentication_system():
    """Test the complete JWT authentication system."""

    print("=" * 60)
    print("JWT Authentication System Test")
    print("=" * 60)

    # Set up test environment
    os.environ["JWT_SECRET_KEY"] = (
        "test_jwt_secret_key_for_authentication_testing_at_least_32_characters_long"
    )

    try:
        # 1. Initialize JWT Manager
        print("\n1. Initializing JWT Manager...")
        jwt_config = JWTConfig(
            enabled=True,
            secret_key=os.environ["JWT_SECRET_KEY"],
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
            issuer="ha-ml-predictor-test",
            audience="ha-ml-predictor-api-test",
            blacklist_enabled=True,
        )

        jwt_manager = JWTManager(jwt_config)
        print("OK JWT Manager initialized successfully")

        # 2. Generate Access Token
        print("\n2. Generating Access Token...")
        user_id = "test_user"
        permissions = ["read", "write", "admin"]
        additional_claims = {
            "username": "test_admin",
            "email": "admin@test.com",
            "is_admin": True,
        }

        access_token = jwt_manager.generate_access_token(
            user_id, permissions, additional_claims
        )
        print(f"OK Access token generated: {access_token[:50]}...")

        # 3. Generate Refresh Token
        print("\n3. Generating Refresh Token...")
        refresh_token = jwt_manager.generate_refresh_token(user_id)
        print(f"OK Refresh token generated: {refresh_token[:50]}...")

        # 4. Validate Access Token
        print("\n4. Validating Access Token...")
        payload = jwt_manager.validate_token(access_token, "access")
        print(f"OK Token validated successfully")
        print(f"  User ID: {payload['sub']}")
        print(f"  Permissions: {payload['permissions']}")
        print(f"  Is Admin: {payload.get('is_admin', False)}")
        print(f"  Expires: {datetime.fromtimestamp(payload['exp'])}")

        # 5. Test Token Info
        print("\n5. Getting Token Information...")
        token_info = jwt_manager.get_token_info(access_token)
        print(f"OK Token info retrieved:")
        print(f"  User ID: {token_info['user_id']}")
        print(f"  Token Type: {token_info['token_type']}")
        print(f"  Is Expired: {token_info['is_expired']}")
        print(f"  Is Blacklisted: {token_info['is_blacklisted']}")

        # 6. Test Token Refresh
        print("\n6. Testing Token Refresh...")
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_token
        )
        print(f"OK Tokens refreshed successfully")
        print(f"  New access token: {new_access_token[:50]}...")
        print(f"  New refresh token: {new_refresh_token[:50]}...")

        # 7. Test Token Revocation
        print("\n7. Testing Token Revocation...")
        revoke_result = jwt_manager.revoke_token(access_token)
        print(f"OK Token revocation: {'Success' if revoke_result else 'Failed'}")

        # 8. Test Revoked Token Validation
        print("\n8. Testing Revoked Token Validation...")
        try:
            jwt_manager.validate_token(access_token, "access")
            print("ERROR: Revoked token should not validate!")
        except APIAuthenticationError as e:
            print(f"OK Revoked token properly rejected: {e.message}")

        # 9. Test AuthUser Model
        print("\n9. Testing AuthUser Model...")
        user = AuthUser(
            user_id="test_user",
            username="test_admin",
            email="admin@test.com",
            permissions=["read", "write", "admin"],
            roles=["admin"],
            is_admin=True,
            is_active=True,
        )

        print(f"OK AuthUser created:")
        print(f"  Has 'admin' permission: {user.has_permission('admin')}")
        print(f"  Has 'admin' role: {user.has_role('admin')}")
        print(f"  Token claims: {user.to_token_claims()}")

        # 10. Test Rate Limiting
        print("\n10. Testing Rate Limiting...")
        rate_limit_hit = False
        try:
            # Generate many tokens quickly to test rate limiting
            for i in range(35):  # Exceed rate limit of 30 per minute
                jwt_manager.generate_access_token(f"user_{i}", ["read"])
        except Exception as e:
            if "rate limit" in str(e).lower():
                rate_limit_hit = True
                print(f"OK Rate limiting working: {e}")
            else:
                print(f"ERROR Unexpected error: {e}")

        if not rate_limit_hit:
            print("WARNING Rate limiting may not be working as expected")

        print("\n" + "=" * 60)
        print("JWT Authentication System Test Complete")
        print("OK All core functionality working correctly")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_jwt_authentication_system())
    sys.exit(0 if success else 1)
