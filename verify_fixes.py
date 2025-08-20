#!/usr/bin/env python3
"""
Script to verify that Category 2 database connection errors are actually fixed.
This will run the specific tests that were failing and confirm they pass.
"""

import asyncio
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_specific_tests():
    """Run the specific database tests that were failing."""
    test_commands = [
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_initialize_success -v -x',
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_verify_connection_success -v -x', 
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_execute_query_success -v -x',
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_healthy -v -x',
        'python -m pytest tests/unit/test_data/test_database.py::TestGlobalDatabaseFunctions::test_get_db_session -v -x',
    ]
    
    results = []
    
    for i, cmd in enumerate(test_commands):
        print(f"\n{'='*60}")
        print(f"Running test {i+1}/{len(test_commands)}")
        print(f"Command: {cmd}")
        print('='*60)
        
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=120,
                cwd=os.path.dirname(__file__)
            )
            
            success = result.returncode == 0
            results.append((cmd, success, result.stdout, result.stderr))
            
            if success:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT")
            results.append((cmd, False, "", "Test timed out"))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append((cmd, False, "", str(e)))
    
    return results

def analyze_results(results):
    """Analyze test results and provide summary."""
    total_tests = len(results)
    passed_tests = sum(1 for _, success, _, _ in results if success)
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print('='*60)
    
    for cmd, success, stdout, stderr in results:
        test_name = cmd.split("::")[-1]
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        
        if not success and stderr:
            print(f"   Error: {stderr[:200]}...")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All database connection tests are now passing!")
        print("✨ Category 2: Database Connection & Integration Errors - FIXED!")
        return True
    else:
        print("⚠️  Some tests are still failing - more fixes needed")
        return False

async def verify_manual_tests():
    """Run manual verification tests."""
    try:
        print("\n🔧 Running manual verification tests...")
        
        # Test basic imports
        from src.data.storage.database import DatabaseManager, get_db_session
        from src.core.config import DatabaseConfig
        print("✅ Database modules import successfully")
        
        # Test DatabaseManager creation
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10
        )
        manager = DatabaseManager(config)
        print("✅ DatabaseManager creates successfully")
        
        # Test async context manager protocol
        async def test_context_manager():
            from unittest.mock import AsyncMock, Mock
            
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.close = AsyncMock()
            
            manager.session_factory = Mock(return_value=mock_session)
            
            async with manager.get_session() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
            
        await test_context_manager()
        print("✅ DatabaseManager.get_session() async context manager works")
        
        # Test global function
        async def test_global_function():
            from unittest.mock import patch, AsyncMock
            
            mock_manager = AsyncMock()
            mock_session = AsyncMock()
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_manager.get_session.return_value = mock_context
            
            with patch('src.data.storage.database.get_database_manager', return_value=mock_manager):
                async with get_db_session() as session:
                    assert session == mock_session
        
        await test_global_function()
        print("✅ Global get_db_session() function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    print("🔍 Verifying Category 2: Database Connection & Integration Error Fixes")
    print("="*80)
    
    # Run manual verification first
    manual_success = asyncio.run(verify_manual_tests())
    
    if not manual_success:
        print("\n❌ Manual verification failed - basic functionality broken")
        return False
    
    # Run the actual pytest tests
    print("\n🧪 Running pytest database tests...")
    results = run_specific_tests()
    
    # Analyze results
    all_passed = analyze_results(results)
    
    if all_passed:
        print("\n🎯 CONCLUSION: All database connection errors have been fixed!")
        print("✅ Category 2 can now be marked as COMPLETED")
        
        # Update the tracking file
        update_tracking_file()
        
        return True
    else:
        print("\n⚠️  CONCLUSION: Some database tests are still failing")
        print("❌ Category 2 fixes are incomplete")
        return False

def update_tracking_file():
    """Update the new_error_categories.md file to reflect completed status."""
    try:
        tracking_file = Path(__file__).parent / "new_error_categories.md"
        
        if tracking_file.exists():
            content = tracking_file.read_text()
            
            # Update the status line for Category 2
            updated_content = content.replace(
                "### Category 2: Database Connection & Integration Errors\n**Status**: ✅ COMPLETED - Implemented proper mocking for unit tests",
                "### Category 2: Database Connection & Integration Errors\n**Status**: ✅ COMPLETED - Fixed async context manager issues and proper database mocking"
            )
            
            # Also update the resolution section
            updated_content = updated_content.replace(
                "**Resolution Implemented**:\n1. **Fixed async context manager protocol violations** in `src/data/storage/database.py`:",
                "**Resolution Implemented** (VERIFIED):\n1. **Fixed async context manager protocol violations** in `src/data/storage/database.py`:"
            )
            
            tracking_file.write_text(updated_content)
            print(f"📝 Updated tracking file: {tracking_file}")
        else:
            print("⚠️  Tracking file not found")
            
    except Exception as e:
        print(f"⚠️  Failed to update tracking file: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)