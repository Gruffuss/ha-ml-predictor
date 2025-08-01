#!/usr/bin/env python3
"""
Test runner script for the occupancy prediction system.

This script provides an easy way to run different categories of tests
with appropriate configuration and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run tests for occupancy prediction system")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "smoke", "database", "ha_client", "slow"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of parallel workers (requires pytest-xdist)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-fast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--keep-running",
        "-k",
        help="Only run tests matching given substring expression"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test selection based on type
    if args.test_type == "all":
        cmd.append("tests/")
    elif args.test_type == "unit":
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration", "tests/integration/"])
    elif args.test_type == "smoke":
        cmd.extend(["-m", "smoke"])
    elif args.test_type == "database":
        cmd.extend(["-m", "database"])
    elif args.test_type == "ha_client":
        cmd.extend(["-m", "ha_client"])
    elif args.test_type == "slow":
        cmd.extend(["-m", "slow"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=70"
        ])
    
    # Add parallel execution if requested
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    
    # Add fail-fast
    if args.fail_fast:
        cmd.append("-x")
    
    # Add test selection filter
    if args.keep_running:
        cmd.extend(["-k", args.keep_running])
    
    # Add HTML report
    if args.html_report:
        cmd.extend(["--html=test-report.html", "--self-contained-html"])
    
    # Run the tests
    description = f"{args.test_type} tests"
    if args.coverage:
        description += " with coverage"
    
    success = run_command(cmd, description)
    
    if success:
        print(f"\n🎉 All {args.test_type} tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
        
        if args.html_report:
            print("\n📝 Test report generated: test-report.html")
            
    else:
        print(f"\n💥 Some {args.test_type} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent.parent
    if script_dir.name == "ha-ml-predictor":
        import os
        os.chdir(script_dir)
    
    main()