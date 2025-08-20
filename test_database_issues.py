#!/usr/bin/env python3
"""
Test script to check current database test issues.
"""

import subprocess
import sys

def run_database_tests():
    """Run specific database tests to identify current failures."""
    test_commands = [
        # Run only the database-related tests
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_initialize_success -v',
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_verify_connection_success -v',
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_execute_query_success -v',
        'python -m pytest tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_healthy -v',
        'python -m pytest tests/unit/test_data/test_database.py::TestGlobalDatabaseFunctions::test_get_db_session -v',
    ]
    
    for i, cmd in enumerate(test_commands):
        print(f"\n{'='*60}")
        print(f"Running test {i+1}/{len(test_commands)}: {cmd}")
        print('='*60)
        
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=r'C:\Users\eu074\OneDrive\Documents\GitHub\ha-ml-predictor'
            )
            
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            
        except subprocess.TimeoutExpired:
            print("Test timed out after 60 seconds")
        except Exception as e:
            print(f"Error running test: {e}")

if __name__ == "__main__":
    run_database_tests()