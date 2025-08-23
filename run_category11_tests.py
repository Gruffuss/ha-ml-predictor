#!/usr/bin/env python3
"""
Test runner for Category 11: Fixture Dependency Errors
"""

import subprocess
import sys
import os

def run_category11_tests():
    """Run tests that should have fixture dependency issues."""

    # Test specific methods mentioned in Category 11
    test_patterns = [
        # Exact tests mentioned
        "tests/unit/test_features/test_contextual.py::TestContextualFeatureExtractorEdgeCases::test_no_room_states",
        "tests/unit/test_adaptation/test_retrainer.py::TestRetrainingNeedEvaluation::test_cooldown_period_enforcement",

        # Run a broader set to catch any fixture issues
        "tests/unit/test_features/test_temporal.py -k 'lookback'",
        "tests/unit/test_features/test_contextual.py -k 'target_time'",
        "tests/unit/test_adaptation/test_retrainer.py -k 'fixture'",
    ]

    print("=== Category 11: Fixture Dependency Error Tests ===\n")

    failed_tests = []

    for test_pattern in test_patterns:
        print(f"Running: {test_pattern}")
        print("-" * 50)

        # Split command for pytest args
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + test_pattern.split()

        result = subprocess.run(
            cmd,
            cwd=".",
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ PASSED: {test_pattern}")
        else:
            print(f"❌ FAILED: {test_pattern}")
            failed_tests.append(test_pattern)

            # Show relevant error info
            if "fixture" in result.stdout.lower() or "fixture" in result.stderr.lower():
                print("\n🔍 FIXTURE ERROR DETECTED:")
                print(result.stdout[-800:] if len(result.stdout) > 800 else result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr[-400:] if len(result.stderr) > 400 else result.stderr)
            else:
                print("No fixture errors detected in output.")

        print("\n" + "="*60 + "\n")

    if failed_tests:
        print(f"\n❌ FAILED TESTS: {len(failed_tests)}")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print("\n✅ ALL FIXTURE DEPENDENCY TESTS PASSED!")

    return len(failed_tests)

if __name__ == "__main__":
    sys.exit(run_category11_tests())
