#!/usr/bin/env python3
"""
Run a specific database test to verify Category 2 fixes.
"""

import subprocess
import sys
import os

# Change to the project directory
os.chdir(os.path.dirname(__file__))

# Run a specific database test that was previously failing
cmd = [
    sys.executable, "-m", "pytest",
    "tests/unit/test_data/test_database.py::TestDatabaseManager::test_execute_query_success",
    "-v", "-s", "--tb=short"
]

print("Running database test to verify Category 2 fixes...")
print(f"Command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print("\n" + "="*60)
    print("TEST OUTPUT:")
    print("="*60)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    if result.returncode == 0:
        print("\n✅ Database test PASSED - Category 2 is working!")
    else:
        print("\n❌ Database test FAILED - Category 2 needs more work")

except subprocess.TimeoutExpired:
    print("❌ Test timed out")
except Exception as e:
    print(f"❌ Error running test: {e}")
