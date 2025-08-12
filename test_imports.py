#!/usr/bin/env python3
"""
Test script to validate all critical imports work correctly.
This helps identify import issues before running the full test suite.
"""

def test_core_imports():
    """Test core module imports."""
    try:
        from src.core.config import ConfigLoader, SystemConfig
        from src.core.environment import EnvironmentManager, Environment
        from src.core.config_validator import ConfigurationValidator, ValidationResult
        from src.core.backup_manager import BackupManager
        print("PASS: Core imports successful")
        return True
    except ImportError as e:
        print(f"FAIL: Core import failed: {e}")
        return False

def test_adaptation_imports():
    """Test adaptation module imports."""
    try:
        from src.adaptation import (
            PredictionValidator, AccuracyMetrics, ValidationStatus,
            ConceptDriftDetector, DriftMetrics, DriftSeverity,
            AccuracyTracker, AccuracyAlert,
            AdaptiveRetrainer, RetrainingRequest, RetrainingStatus,
            ModelOptimizer, OptimizationConfig, OptimizationResult,
            TrackingManager, TrackingConfig
        )
        print("PASS: Adaptation imports successful")
        return True
    except ImportError as e:
        print(f"FAIL: Adaptation import failed: {e}")
        return False

def test_models_imports():
    """Test models module imports."""
    try:
        from src.models.base import BasePredictor, PredictionResult, TrainingResult
        print("PASS: Models imports successful") 
        return True
    except ImportError as e:
        print(f"FAIL: Models import failed: {e}")
        return False

def test_cryptography_import():
    """Test cryptography import."""
    try:
        from cryptography.fernet import Fernet
        print("PASS: Cryptography import successful")
        return True
    except ImportError as e:
        print(f"FAIL: Cryptography import failed: {e}")
        return False

def test_pytest_plugins():
    """Test pytest plugin imports."""
    try:
        import pytest_html
        import pytest_timeout
        import pytest_xdist
        print("PASS: Pytest plugins imports successful")
        return True
    except ImportError as e:
        print(f"FAIL: Pytest plugins import failed: {e}")
        return False

def main():
    """Run all import tests."""
    print("Testing critical imports...")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_adaptation_imports, 
        test_models_imports,
        test_cryptography_import,
        test_pytest_plugins,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Import tests: {passed}/{total} passed")
    
    if passed == total:
        print("SUCCESS: All imports working correctly!")
        return 0
    else:
        print("WARNING: Some imports failing - check dependencies")
        return 1

if __name__ == "__main__":
    exit(main())