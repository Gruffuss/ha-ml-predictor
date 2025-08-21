#!/usr/bin/env python3
"""Run specific Category 7 assertion tests to verify fixes."""

import subprocess
import sys

def run_test(test_pattern, description):
    """Run specific test and return result."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_pattern, 
            "-xvs", "--tb=short"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✓ PASSED: {description}")
            return True
        else:
            print(f"✗ FAILED: {description}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {description} - {str(e)}")
        return False

def main():
    """Run all Category 7 assertion tests."""
    print("=== Category 7: Assertion/Expectation Fixes ===\n")
    
    tests = [
        # Test optimizer model type fixes  
        ("tests/unit/test_adaptation/test_optimizer.py::TestModelOptimizer", "Optimizer model type fixes"),
        
        # Test priority queue consistency
        ("tests/unit/test_adaptation/test_retrainer.py::TestRetrainingRequestManagement::test_retraining_queue_priority_ordering", "Priority queue consistency"),
        
        # Test exception error codes
        ("tests/unit/test_core/test_exceptions.py::TestExceptionIntegration::test_error_code_uniqueness", "Exception error code format"),
        
        # Test config database expectations
        ("tests/unit/test_core/test_config.py::TestDatabaseConfig", "Database config expectations"),
        
        # Test health check mocks
        ("tests/test_end_to_end_validation.py -k health", "Health check mock fixes"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_pattern, description in tests:
        if run_test(test_pattern, description):
            passed += 1
        print()  # Add blank line
    
    print(f"=== RESULTS: {passed}/{total} Category 7 assertion tests passing ===")
    
    if passed == total:
        print("🎉 ALL Category 7 assertion fixes COMPLETE!")
        return 0
    else:
        print("❌ Some Category 7 assertion tests still failing")
        return 1

if __name__ == "__main__":
    sys.exit(main())