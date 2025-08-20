#!/usr/bin/env python3
"""
Quick diagnostic script to identify the specific database errors.
"""

import asyncio
import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_database_imports():
    """Test if database modules can be imported."""
    try:
        from src.data.storage.database import (
            DatabaseManager, 
            get_database_manager, 
            get_db_session
        )
        print("✅ Database modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        traceback.print_exc()
        return False

async def test_async_context_manager():
    """Test async context manager protocol."""
    try:
        from src.data.storage.database import get_db_session
        from unittest.mock import AsyncMock, patch
        
        # Mock the database manager to avoid actual connections
        mock_manager = AsyncMock()
        mock_session = AsyncMock()
        
        # Create proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_manager.get_session.return_value = mock_context
        
        with patch('src.data.storage.database.get_database_manager', return_value=mock_manager):
            async with get_db_session() as session:
                print("✅ get_db_session async context manager works")
                return True
    except Exception as e:
        print(f"❌ Async context manager test failed: {e}")
        traceback.print_exc()
        return False

async def test_database_manager_creation():
    """Test DatabaseManager creation."""
    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig
        
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        
        manager = DatabaseManager(config)
        print("✅ DatabaseManager created successfully")
        return True
    except Exception as e:
        print(f"❌ DatabaseManager creation failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all diagnostic tests."""
    print("🔍 Running database diagnostics...\n")
    
    tests = [
        ("Import Test", test_database_imports()),
        ("DatabaseManager Creation", test_database_manager_creation()),
        ("Async Context Manager", test_async_context_manager()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("📊 DIAGNOSTIC SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All diagnostics passed!")
    else:
        print("⚠️  Some diagnostics failed - issues need to be fixed")

if __name__ == "__main__":
    asyncio.run(main())