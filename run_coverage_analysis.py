#!/usr/bin/env python3
"""
Coverage analysis script to check test coverage progress.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_coverage_analysis():
    """Run coverage analysis on the test suite."""
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🔍 Running comprehensive test coverage analysis...")
    print("=" * 60)
    
    try:
        # Run tests with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-config=.coveragerc",
            "-v",
            "tests/"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n✅ Coverage analysis completed successfully!")
            print("📊 Check htmlcov/index.html for detailed coverage report")
        else:
            print("\n❌ Coverage analysis failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running coverage analysis: {e}")
        return False

def run_specific_module_coverage():
    """Run coverage on specific high-value modules."""
    
    modules_to_test = [
        "tests/unit/test_models/test_xgboost_comprehensive.py",
        "tests/unit/test_models/test_lstm_comprehensive.py", 
        "tests/unit/test_data/test_database_comprehensive.py",
        "tests/unit/test_core/test_exceptions_comprehensive.py",
        "tests/unit/test_integration/test_mqtt_publisher_comprehensive.py",
        "tests/unit/test_features/test_temporal_comprehensive.py"
    ]
    
    print("🎯 Running coverage on newly created comprehensive test modules...")
    print("=" * 70)
    
    for module in modules_to_test:
        if os.path.exists(module):
            print(f"\n📝 Testing {module}...")
            
            cmd = [
                sys.executable, "-m", "pytest",
                module,
                "--cov=src",
                "--cov-report=term",
                "-v"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"✅ {module} - PASSED")
                else:
                    print(f"❌ {module} - FAILED")
                    print(f"Error: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {module} - TIMEOUT")
            except Exception as e:
                print(f"❌ {module} - ERROR: {e}")
        else:
            print(f"⚠️  {module} - NOT FOUND")

if __name__ == "__main__":
    print("🚀 Starting Test Coverage Analysis")
    print("=" * 50)
    
    # First run specific modules
    run_specific_module_coverage()
    
    print("\n" + "=" * 70)
    print("📊 Running full coverage analysis...")
    
    # Then run full analysis
    success = run_coverage_analysis()
    
    if success:
        print("\n🎉 Coverage analysis complete! Check the results above.")
        print("🎯 Target: 85% coverage for deployment readiness")
    else:
        print("\n⚠️  Coverage analysis had issues. Check error messages above.")
    
    print("\n💡 Key files created for comprehensive coverage:")
    print("   - tests/unit/test_models/test_xgboost_comprehensive.py")
    print("   - tests/unit/test_models/test_lstm_comprehensive.py")
    print("   - tests/unit/test_data/test_database_comprehensive.py")
    print("   - tests/unit/test_core/test_exceptions_comprehensive.py")
    print("   - tests/unit/test_integration/test_mqtt_publisher_comprehensive.py")
    print("   - tests/unit/test_features/test_temporal_comprehensive.py")