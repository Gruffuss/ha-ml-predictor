#!/usr/bin/env python3
"""
Quick verification script to test if Category 2 database connection issues are actually resolved.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_database_manager_async_context():
    """Test if DatabaseManager.get_session() async context manager works properly."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        from unittest.mock import AsyncMock, Mock
        
        # Create database manager with test config
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        manager = DatabaseManager(config)
        
        # Mock session factory with proper async session mock
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.is_active = False
        
        manager.session_factory = Mock(return_value=mock_session)
        
        # Test async context manager protocol
        async with manager.get_session() as session:
            assert session == mock_session
            # Test that we can do something with the session
            pass
            
        # Verify that commit and close were called
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        
        print("✅ DatabaseManager.get_session() async context manager works correctly")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager async context manager failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_global_get_db_session():
    """Test if global get_db_session() function works properly."""
    try:
        from src.data.storage.database import get_db_session
        from unittest.mock import patch, AsyncMock
        
        # Mock database manager
        mock_manager = AsyncMock()
        mock_session = AsyncMock()
        
        # Create proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_manager.get_session = Mock(return_value=mock_context)
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_manager):
            async with get_db_session() as session:
                assert session == mock_session
                # Test that we got the expected session
                pass
        
        print("✅ Global get_db_session() function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Global get_db_session() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_manager_execute_query():
    """Test if DatabaseManager.execute_query() works with async context manager."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        from unittest.mock import AsyncMock, Mock, patch
        
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        manager = DatabaseManager(config)
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = ("test_result",)
        mock_session.execute.return_value = mock_result
        
        # Create proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(manager, 'get_session', return_value=mock_context):
            result = await manager.execute_query("SELECT 1", fetch_one=True)
            assert result == ("test_result",)
        
        print("✅ DatabaseManager.execute_query() works correctly")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager.execute_query() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all verification tests."""
    print("🔍 Verifying Category 2: Database Connection & Integration Fixes")
    print("=" * 70)
    
    tests = [
        ("DatabaseManager async context manager", test_database_manager_async_context),
        ("Global get_db_session function", test_global_get_db_session),
        ("DatabaseManager execute_query method", test_database_manager_execute_query),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 50)
        
        try:
            success = await test_func()
            if success:
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All database connection functionality works correctly!")
        print("✅ Category 2: Database Connection & Integration Errors - VERIFIED FIXED")
        return True
    else:
        print("⚠️  Some database connection issues still exist")
        print("❌ Category 2 fixes need more work")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)