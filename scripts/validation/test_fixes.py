#!/usr/bin/env python3
"""
Test runner script to validate the Sprint 1 test fixes.
"""

import sys
import subprocess
import os


def run_test(test_name):
    """Run a specific test and return result."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"tests/test_sprint1_validation.py::{test_name}",
        "-v",
        "--tb=short",
        "--no-header",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running test: {e}")
        return False


def main():
    """Run the three failing tests."""
    failing_tests = [
        "test_sprint1_database_system",
        "test_sprint1_model_relationships",
        "test_sprint1_end_to_end_workflow",
    ]

    results = {}

    for test in failing_tests:
        results[test] = run_test(test)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
