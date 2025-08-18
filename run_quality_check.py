#!/usr/bin/env python3
"""Run quality checks on the feature test files."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and return its result."""
    print(f"\n🔍 Running {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"💥 {description} error: {e}")
        return False

def main():
    """Run quality pipeline on feature tests."""
    print("🚀 Running quality pipeline on feature test files...")
    
    # Target files
    files = [
        "tests/unit/test_features/test_contextual.py",
        "tests/unit/test_features/test_temporal.py", 
        "tests/unit/test_features/test_engineering.py"
    ]
    
    file_args = " ".join(files)
    
    commands = [
        (f"black --check --diff --line-length 88 {file_args}", "Black formatting check"),
        (f"isort --check-only --diff --profile black {file_args}", "isort import sorting check"),
        (f"flake8 {file_args} --max-line-length=140 --extend-ignore=E203,W503,E501,W291,W293,E402,C901", "Flake8 linting"),
        (f"mypy {file_args} --config-file=mypy.ini", "MyPy type checking"),
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    if all_passed:
        print("\n🎉 All quality checks passed!")
        return 0
    else:
        print("\n💔 Some quality checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())