#!/usr/bin/env python3
"""
Test script to verify the database fixes are working correctly.
This script will test the specific database functionality that was failing.
"""

import asyncio
import sys
import os
import traceback
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_database_manager_async_context():
    """Test DatabaseManager get_session async context manager."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        
        manager = DatabaseManager(config)
        
        # Mock the session factory to avoid real connections
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        
        manager.session_factory = Mock(return_value=mock_session)
        
        # Test the async context manager
        async with manager.get_session() as session:
            assert session == mock_session
            print("✅ DatabaseManager.get_session() context manager works correctly")
        
        # Verify commit was called
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager async context test failed: {e}")
        traceback.print_exc()
        return False

async def test_global_get_db_session():
    """Test global get_db_session function."""
    try:
        from src.data.storage.database import get_db_session, DatabaseManager
        
        # Mock the global database manager
        mock_manager = AsyncMock()
        mock_session = AsyncMock()
        
        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_manager.get_session.return_value = mock_context
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_manager):
            async with get_db_session() as session:
                assert session == mock_session
                print("✅ Global get_db_session() works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Global get_db_session test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_manager_initialization():
    """Test DatabaseManager initialization process."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        
        manager = DatabaseManager(config)
        
        # Mock all the initialization methods
        with (
            patch.object(manager, '_create_engine', new_callable=AsyncMock) as mock_create,
            patch.object(manager, '_setup_session_factory', new_callable=AsyncMock) as mock_setup,
            patch.object(manager, '_verify_connection', new_callable=AsyncMock) as mock_verify,
            patch('asyncio.create_task') as mock_create_task,
        ):
            
            await manager.initialize()
            
            # Verify all initialization steps were called
            mock_create.assert_called_once()
            mock_setup.assert_called_once()
            mock_verify.assert_called_once()
            mock_create_task.assert_called_once()
            
            print("✅ DatabaseManager initialization works correctly")
            return True
            
    except Exception as e:
        print(f"❌ DatabaseManager initialization test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_manager_execute_query():
    """Test DatabaseManager execute_query method."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        
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
        
        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        
        with patch.object(manager, 'get_session', return_value=mock_context):
            result = await manager.execute_query(
                "SELECT 1",
                parameters={"param": "value"},
                fetch_one=True
            )
            
            assert result == ("test_result",)
            print("✅ DatabaseManager execute_query works correctly")
            return True
            
    except Exception as e:
        print(f"❌ DatabaseManager execute_query test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_health_check():
    """Test DatabaseManager health_check method."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        
        manager = DatabaseManager(config)
        
        # Mock session and results
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        
        # Mock engine and pool
        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 5
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        manager.engine.pool = mock_pool
        
        with patch.object(manager, 'get_session', return_value=mock_context):
            health = await manager.health_check()
            
            assert health['status'] == 'healthy'
            assert 'performance_metrics' in health
            print("✅ DatabaseManager health_check works correctly")
            return True
            
    except Exception as e:
        print(f"❌ DatabaseManager health_check test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all database fix tests."""
    print("🔧 Testing database fixes...\n")
    
    tests = [
        ("DatabaseManager async context", test_database_manager_async_context()),
        ("Global get_db_session", test_global_get_db_session()),
        ("DatabaseManager initialization", test_database_manager_initialization()),
        ("DatabaseManager execute_query", test_database_manager_execute_query()),
        ("DatabaseManager health_check", test_database_health_check()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*60}")
    print("📊 DATABASE FIXES TEST SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All database fixes are working correctly!")
        print("\nNext step: Update the tracking file to mark Category 2 as truly completed")
    else:
        print("⚠️  Some fixes still need work")
        
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)