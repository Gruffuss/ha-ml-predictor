#!/usr/bin/env python3
"""
Code Quality Pipeline Runner

Runs the mandatory code quality pipeline after any agent code changes:
1. Black code formatting
2. isort import sorting
3. flake8 linting
4. mypy type checking

This script ensures zero errors across all four tools.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 Running {description}...")
    print(f"Command: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ {description} passed")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def main():
    """Run the complete quality pipeline."""
    print("🚀 Starting Code Quality Pipeline")

    # Change to project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print(f"Working directory: {project_root}")

    # Define quality checks
    quality_checks = [
        (
            "black --check --diff --line-length 88 src/ tests/ scripts/",
            "Black code formatting check"
        ),
        (
            "isort --check-only --diff --profile black src/ tests/ scripts/",
            "isort import sorting check"
        ),
        (
            "flake8 src/ tests/ scripts/ --max-line-length=140 --extend-ignore=E203,W503,E501,W291,W293,E402,C901",
            "flake8 linting check"
        ),
        (
            "mypy src/ --config-file=mypy.ini",
            "mypy type checking"
        )
    ]

    all_passed = True
    results = []

    # Run each quality check
    for command, description in quality_checks:
        success = run_command(command, description)
        results.append((description, success))
        if not success:
            all_passed = False

    # Print summary
    print("\n" + "="*60)
    print("📊 Code Quality Pipeline Results")
    print("="*60)

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")

    print("="*60)

    if all_passed:
        print("🎉 All quality checks passed! Code is ready for commit.")
        return 0
    else:
        print("🔧 Some quality checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
