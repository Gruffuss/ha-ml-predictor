#!/usr/bin/env python3
"""
Diagnostics script to run pytest and identify issues.
"""

import sys
import os
from pathlib import Path
import traceback

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if key modules can be imported."""
    print("=== TESTING IMPORTS ===")
    
    modules_to_test = [
        "src.core.config",
        "src.core.constants", 
        "src.core.exceptions",
        "src.models.base.predictor",
        "src.models.base.lstm_predictor",
        "src.models.base.hmm_predictor",
        "src.models.base.xgboost_predictor",
        "src.models.base.gp_predictor",
        "src.data.storage.models",
        "src.data.ingestion.ha_client",
        "pytest",
        "pandas",
        "numpy",
        "sklearn",
    ]
    
    import_results = {}
    for module in modules_to_test:
        try:
            __import__(module)
            import_results[module] = "✅ SUCCESS"
            print(f"✅ {module}")
        except Exception as e:
            import_results[module] = f"❌ ERROR: {str(e)}"
            print(f"❌ {module}: {str(e)}")
    
    return import_results

def check_test_structure():
    """Check test directory structure."""
    print("\n=== TESTING DIRECTORY STRUCTURE ===")
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ tests/ directory not found")
        return False
        
    unit_dir = test_dir / "unit"
    if not unit_dir.exists():
        print("❌ tests/unit/ directory not found")
        return False
        
    print("✅ Test directory structure exists")
    
    # Find test files
    test_files = list(test_dir.rglob("test_*.py"))
    print(f"Found {len(test_files)} test files:")
    for f in test_files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(test_files) > 10:
        print(f"  ... and {len(test_files) - 10} more")
    
    return True

def run_simple_pytest():
    """Try to run pytest on a simple test."""
    print("\n=== RUNNING SIMPLE PYTEST ===")
    
    try:
        import pytest
        
        # Try to run pytest on conftest.py first to check setup
        conftest_path = Path("tests/conftest.py")
        if conftest_path.exists():
            print("✅ conftest.py found")
        else:
            print("❌ conftest.py not found")
            return False
            
        # Try to collect tests
        print("Attempting to collect tests...")
        
        # Run pytest with collection only
        exit_code = pytest.main([
            "tests/",
            "--collect-only",
            "-q"
        ])
        
        print(f"Collection exit code: {exit_code}")
        return exit_code == 0
        
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        traceback.print_exc()
        return False

def check_specific_test_file():
    """Check a specific test file for obvious issues."""
    print("\n=== CHECKING SPECIFIC TEST FILE ===")
    
    test_file = Path("tests/unit/test_core/test_config.py")
    if not test_file.exists():
        print(f"❌ {test_file} not found")
        return False
        
    try:
        # Try to import the test file
        sys.path.insert(0, str(test_file.parent))
        spec = __import__(test_file.stem)
        print(f"✅ Successfully imported {test_file}")
        return True
    except Exception as e:
        print(f"❌ Error importing {test_file}: {e}")
        traceback.print_exc()
        return False

def check_environment():
    """Check environment setup."""
    print("\n=== CHECKING ENVIRONMENT ===")
    
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check if required environment variables are set
    env_vars = [
        "JWT_SECRET_KEY",
        "ENVIRONMENT", 
        "TESTING"
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"⚠️  {var} not set")

def run_minimal_test():
    """Try to run one minimal test."""
    print("\n=== RUNNING MINIMAL TEST ===")
    
    try:
        import pytest
        
        # Run just one simple test
        exit_code = pytest.main([
            "tests/unit/test_core/test_config.py::TestHomeAssistantConfig::test_default_values",
            "-v",
            "-s"
        ])
        
        print(f"Minimal test exit code: {exit_code}")
        return exit_code == 0
        
    except Exception as e:
        print(f"❌ Error running minimal test: {e}")
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function."""
    print("PYTEST DIAGNOSTICS")
    print("==================")
    
    results = {}
    
    # Run diagnostics
    results['imports'] = test_imports()
    results['structure'] = check_test_structure()
    
    check_environment()
    
    results['collection'] = run_simple_pytest()
    results['specific_file'] = check_specific_test_file()
    results['minimal_test'] = run_minimal_test()
    
    # Summary
    print("\n=== SUMMARY ===")
    
    if results['structure'] and results['collection']:
        print("✅ Basic pytest setup is working")
    else:
        print("❌ Basic pytest setup has issues")
    
    # Count import failures
    import_failures = [k for k, v in results['imports'].items() if "ERROR" in v]
    if import_failures:
        print(f"❌ {len(import_failures)} import failures:")
        for failure in import_failures[:5]:  # Show first 5
            print(f"  - {failure}")
    else:
        print("✅ All imports successful")
    
    print("\nTo run full test suite manually:")
    print("python -m pytest tests/ -v")

if __name__ == "__main__":
    main()