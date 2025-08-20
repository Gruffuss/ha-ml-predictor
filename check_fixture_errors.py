#!/usr/bin/env python3
"""
Quick script to check fixture dependency errors in Category 11.
"""

import subprocess
import sys

def run_specific_tests():
    """Run the specific tests that have fixture dependency errors."""
    
    tests_to_check = [
        "tests/unit/test_features/test_contextual.py::TestContextualFeatureExtractorEdgeCases::test_no_room_states",
        "tests/unit/test_adaptation/test_retrainer.py::TestRetrainingNeedEvaluation::test_cooldown_period_enforcement",
    ]
    
    print("Checking fixture dependency errors for Category 11...")
    
    for test in tests_to_check:
        print(f"\n{'='*60}")
        print(f"Testing: {test}")
        print('='*60)
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-xvs", test],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED: {test}")
        else:
            print(f"❌ FAILED: {test}")
            print("STDOUT:")
            print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            print("STDERR:")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

if __name__ == "__main__":
    run_specific_tests()