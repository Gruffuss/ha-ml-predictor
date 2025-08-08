#!/usr/bin/env python3
"""
Sprint 5 Test Runner

This script provides a convenient way to run Sprint 5 integration and end-to-end tests
with various options for coverage, filtering, and reporting.

Usage:
    python run_sprint5_tests.py [options]

Options:
    --integration-only    Run only integration tests
    --e2e-only           Run only end-to-end tests
    --with-coverage      Run tests with coverage reporting
    --verbose           Run tests with verbose output
    --fast              Skip slow tests
    --performance       Run only performance tests
    --help              Show this help message
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Sprint5TestRunner:
    """Test runner for Sprint 5 integration tests."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"

        # Test files
        self.integration_tests = "test_sprint5_integration.py"
        self.e2e_tests = "test_end_to_end_validation.py"
        self.fixtures = "test_sprint5_fixtures.py"

        # Ensure we're in the right directory
        os.chdir(self.project_root)

    def check_dependencies(self) -> bool:
        """Check if required testing dependencies are installed."""
        try:
            import pytest
            import pytest_asyncio

            return True
        except ImportError as e:
            logger.error(f"Missing testing dependencies: {e}")
            logger.info(
                "Please install with: pip install pytest pytest-asyncio pytest-mock"
            )
            return False

    def run_integration_tests(
        self, verbose: bool = False, coverage: bool = False, fast: bool = False
    ) -> int:
        """Run Sprint 5 integration tests."""
        logger.info("Running Sprint 5 integration tests...")

        cmd = ["python", "-m", "pytest", f"tests/{self.integration_tests}"]

        if verbose:
            cmd.append("-v")

        if fast:
            cmd.extend(["-m", "not slow"])

        if coverage:
            cmd.extend(
                [
                    "--cov=src/integration",
                    "--cov=src/adaptation",
                    "--cov-report=html:coverage_reports/sprint5_integration",
                    "--cov-report=term-missing",
                ]
            )

        cmd.extend(["--tb=short", "-x"])  # Stop on first failure

        return subprocess.call(cmd)

    def run_e2e_tests(
        self, verbose: bool = False, coverage: bool = False, fast: bool = False
    ) -> int:
        """Run end-to-end validation tests."""
        logger.info("Running Sprint 5 end-to-end tests...")

        cmd = ["python", "-m", "pytest", f"tests/{self.e2e_tests}"]

        if verbose:
            cmd.append("-v")

        if fast:
            cmd.extend(["-m", "not slow"])

        if coverage:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=html:coverage_reports/sprint5_e2e",
                    "--cov-report=term-missing",
                ]
            )

        cmd.extend(["--tb=short"])

        return subprocess.call(cmd)

    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance and benchmarking tests."""
        logger.info("Running Sprint 5 performance tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            f"tests/{self.integration_tests}",
            f"tests/{self.e2e_tests}",
            "-m",
            "performance",
            "--tb=short",
        ]

        if verbose:
            cmd.append("-v")

        return subprocess.call(cmd)

    def run_all_tests(
        self, verbose: bool = False, coverage: bool = False, fast: bool = False
    ) -> int:
        """Run all Sprint 5 tests."""
        logger.info("Running all Sprint 5 tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            f"tests/{self.integration_tests}",
            f"tests/{self.e2e_tests}",
        ]

        if verbose:
            cmd.append("-v")

        if fast:
            cmd.extend(["-m", "not slow"])

        if coverage:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=html:coverage_reports/sprint5_complete",
                    "--cov-report=term-missing",
                    "--cov-fail-under=80",
                ]
            )

        cmd.extend(["--tb=short"])

        return subprocess.call(cmd)

    def validate_test_files(self) -> bool:
        """Validate that test files exist and are syntactically correct."""
        logger.info("Validating Sprint 5 test files...")

        test_files = [
            self.test_dir / self.integration_tests,
            self.test_dir / self.e2e_tests,
            self.test_dir / self.fixtures,
        ]

        for test_file in test_files:
            if not test_file.exists():
                logger.error(f"Test file not found: {test_file}")
                return False

            # Check syntax
            try:
                subprocess.check_call(
                    ["python", "-m", "py_compile", str(test_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"✓ {test_file.name} syntax OK")
            except subprocess.CalledProcessError:
                logger.error(f"✗ {test_file.name} has syntax errors")
                return False

        return True

    def create_coverage_reports_dir(self):
        """Create coverage reports directory if it doesn't exist."""
        coverage_dir = self.project_root / "coverage_reports"
        coverage_dir.mkdir(exist_ok=True)
        logger.info(f"Coverage reports will be saved to: {coverage_dir}")

    def run_quick_smoke_test(self) -> int:
        """Run a quick smoke test to verify basic functionality."""
        logger.info("Running Sprint 5 smoke test...")

        cmd = [
            "python",
            "-m",
            "pytest",
            f"tests/{self.integration_tests}::TestTrackingIntegrationManager::test_integration_manager_initialization",
            f"tests/{self.e2e_tests}::TestCompleteSystemWorkflow::test_sensor_event_to_prediction_workflow",
            "-v",
            "--tb=short",
        ]

        return subprocess.call(cmd)

    def generate_test_report(self) -> int:
        """Generate comprehensive test report."""
        logger.info("Generating Sprint 5 test report...")

        cmd = [
            "python",
            "-m",
            "pytest",
            f"tests/{self.integration_tests}",
            f"tests/{self.e2e_tests}",
            "--html=test_reports/sprint5_report.html",
            "--self-contained-html",
            "--cov=src",
            "--cov-report=html:coverage_reports/sprint5_complete",
            "-v",
        ]

        # Create reports directory
        reports_dir = self.project_root / "test_reports"
        reports_dir.mkdir(exist_ok=True)

        return subprocess.call(cmd)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Sprint 5 Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )

    parser.add_argument(
        "--e2e-only", action="store_true", help="Run only end-to-end tests"
    )

    parser.add_argument(
        "--performance-only", action="store_true", help="Run only performance tests"
    )

    parser.add_argument(
        "--with-coverage", action="store_true", help="Run tests with coverage reporting"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Run tests with verbose output"
    )

    parser.add_argument("--fast", action="store_true", help="Skip slow tests")

    parser.add_argument(
        "--smoke-test", action="store_true", help="Run quick smoke test only"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate test files, don't run tests",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive test report"
    )

    args = parser.parse_args()

    # Find project root
    current_dir = Path.cwd()
    if not (current_dir / "src").exists():
        logger.error("Please run this script from the project root directory")
        return 1

    runner = Sprint5TestRunner(current_dir)

    # Check dependencies
    if not runner.check_dependencies():
        return 1

    # Validate test files
    if not runner.validate_test_files():
        return 1

    if args.validate_only:
        logger.info("Test files validation completed successfully")
        return 0

    # Create coverage directory if needed
    if args.with_coverage or args.report:
        runner.create_coverage_reports_dir()

    # Determine what to run
    exit_code = 0

    if args.smoke_test:
        exit_code = runner.run_quick_smoke_test()
    elif args.report:
        exit_code = runner.generate_test_report()
    elif args.integration_only:
        exit_code = runner.run_integration_tests(
            verbose=args.verbose, coverage=args.with_coverage, fast=args.fast
        )
    elif args.e2e_only:
        exit_code = runner.run_e2e_tests(
            verbose=args.verbose, coverage=args.with_coverage, fast=args.fast
        )
    elif args.performance_only:
        exit_code = runner.run_performance_tests(verbose=args.verbose)
    else:
        # Run all tests
        exit_code = runner.run_all_tests(
            verbose=args.verbose, coverage=args.with_coverage, fast=args.fast
        )

    if exit_code == 0:
        logger.info("✓ All Sprint 5 tests completed successfully!")
    else:
        logger.error("✗ Some Sprint 5 tests failed")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
