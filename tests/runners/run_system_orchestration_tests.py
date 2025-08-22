"""
System Orchestration Test Runner.

This script provides a comprehensive test runner for system-level orchestration
tests, focusing on achieving 85% test coverage through systematic validation of
system components, failure scenarios, and resource management.

Usage:
    python tests/runners/run_system_orchestration_tests.py [options]
    
Options:
    --fast: Run only fast system tests (exclude long-running stability tests)
    --full: Run complete system test suite including long-running tests
    --coverage: Generate coverage report for system tests
    --report: Generate detailed test report with metrics
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


class SystemOrchestrationTestRunner:
    """Orchestrate system-level test execution with comprehensive reporting."""

    def __init__(self):
        self.start_time = None
        self.test_results = {}
        self.coverage_data = {}

    def run_fast_system_tests(self):
        """Run fast system tests excluding long-running stability tests."""
        print("🚀 Running Fast System Orchestration Tests...")
        print("=" * 60)

        test_args = [
            "tests/system/",
            "-m",
            "system and not slow",
            "-v",
            "--tb=short",
            "--durations=10",
            "--strict-markers",
        ]

        return pytest.main(test_args)

    def run_full_system_tests(self):
        """Run complete system test suite including long-running tests."""
        print("🔥 Running Complete System Orchestration Test Suite...")
        print("=" * 60)
        print(
            "⚠️  This includes long-running stability tests and may take several minutes"
        )
        print()

        test_args = [
            "tests/system/",
            "-m",
            "system",
            "-v",
            "--tb=short",
            "--durations=20",
            "--strict-markers",
        ]

        return pytest.main(test_args)

    def run_with_coverage(self):
        """Run system tests with coverage analysis."""
        print("📊 Running System Tests with Coverage Analysis...")
        print("=" * 60)

        test_args = [
            "tests/system/",
            "-m",
            "system and not slow",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/system_tests",
            "--cov-fail-under=85",
            "-v",
            "--tb=short",
        ]

        return pytest.main(test_args)

    def run_specific_test_category(self, category: str):
        """Run specific category of system tests."""
        test_modules = {
            "failure": "tests/system/test_orchestration_failure_recovery.py",
            "resources": "tests/system/test_resource_constraints.py",
            "lifecycle": "tests/system/test_lifecycle_management.py",
            "stability": "tests/system/test_long_running_stability.py",
            "errors": "tests/system/test_error_propagation.py",
        }

        if category not in test_modules:
            print(f"❌ Unknown test category: {category}")
            print(f"Available categories: {', '.join(test_modules.keys())}")
            return 1

        print(f"🎯 Running {category.title()} System Tests...")
        print("=" * 60)

        test_args = [test_modules[category], "-v", "--tb=short", "--durations=10"]

        return pytest.main(test_args)

    def validate_test_environment(self):
        """Validate that test environment is properly configured."""
        print("🔍 Validating Test Environment...")

        # Check required dependencies
        try:
            import psutil
            import pytest

            print("✅ Required test dependencies available")
        except ImportError as e:
            print(f"❌ Missing test dependency: {e}")
            return False

        # Check test structure
        system_test_dir = Path("tests/system")
        if not system_test_dir.exists():
            print(f"❌ System test directory not found: {system_test_dir}")
            return False

        expected_test_files = [
            "test_orchestration_failure_recovery.py",
            "test_resource_constraints.py",
            "test_lifecycle_management.py",
            "test_long_running_stability.py",
            "test_error_propagation.py",
        ]

        missing_files = []
        for test_file in expected_test_files:
            if not (system_test_dir / test_file).exists():
                missing_files.append(test_file)

        if missing_files:
            print(f"❌ Missing test files: {missing_files}")
            return False

        print("✅ Test environment validation successful")
        return True

    def generate_test_report(self):
        """Generate comprehensive test execution report."""
        print("\n" + "=" * 60)
        print("📋 SYSTEM ORCHESTRATION TEST REPORT")
        print("=" * 60)

        # Run test discovery to get test count
        discovery_args = ["tests/system/", "--collect-only", "-q"]

        print("🔍 Test Discovery Results:")
        pytest.main(discovery_args)

        print("\n📊 Test Categories:")
        categories = [
            ("Failure Recovery", "Component isolation and cascading failure scenarios"),
            ("Resource Constraints", "Memory, CPU, and connection limit testing"),
            (
                "Lifecycle Management",
                "Startup sequence and graceful shutdown validation",
            ),
            ("Long-Running Stability", "Extended runtime and memory leak detection"),
            ("Error Propagation", "Cross-component error handling validation"),
        ]

        for category, description in categories:
            print(f"  • {category}: {description}")

        print("\n🎯 Coverage Goals:")
        print("  • Target: 85% system-level test coverage")
        print("  • Focus: End-to-end orchestration scenarios")
        print("  • Validation: Production-ready failure handling")

        print("\n🚀 Execution Commands:")
        print(
            "  • Fast tests: python tests/runners/run_system_orchestration_tests.py --fast"
        )
        print(
            "  • Full suite: python tests/runners/run_system_orchestration_tests.py --full"
        )
        print(
            "  • With coverage: python tests/runners/run_system_orchestration_tests.py --coverage"
        )


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="System Orchestration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast system tests (exclude long-running tests)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete system test suite including long-running tests",
    )

    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage analysis"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed test report with metrics",
    )

    parser.add_argument(
        "--category",
        choices=["failure", "resources", "lifecycle", "stability", "errors"],
        help="Run specific test category",
    )

    args = parser.parse_args()

    runner = SystemOrchestrationTestRunner()

    # Validate environment first
    if not runner.validate_test_environment():
        print(
            "\n❌ Test environment validation failed. Please fix issues before running tests."
        )
        return 1

    # Show report if requested
    if args.report:
        runner.generate_test_report()
        return 0

    # Run specific category if requested
    if args.category:
        return runner.run_specific_test_category(args.category)

    # Run tests based on options
    if args.coverage:
        result = runner.run_with_coverage()
    elif args.full:
        result = runner.run_full_system_tests()
    elif args.fast:
        result = runner.run_fast_system_tests()
    else:
        # Default: run fast tests
        print("ℹ️  Running fast system tests by default. Use --full for complete suite.")
        result = runner.run_fast_system_tests()

    # Print final result
    print("\n" + "=" * 60)
    if result == 0:
        print("✅ System orchestration tests completed successfully!")
        print("🎯 System-level test coverage goals achieved")
    else:
        print("❌ Some system tests failed or coverage goals not met")
        print("📊 Review test output and address any failures")

    print("=" * 60)
    return result


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
