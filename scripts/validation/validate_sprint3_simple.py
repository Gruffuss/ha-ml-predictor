#!/usr/bin/env python3
"""
Sprint 3 Simple Validation Script

Minimal validation that checks file structure and basic Sprint 3 components
without requiring external dependencies.
"""

import sys
import os
from pathlib import Path


def check_file_structure():
    """Check that all expected Sprint 3 files exist."""
    print("Checking Sprint 3 file structure...")

    base_path = Path(__file__).parent

    # Required files for Sprint 3
    required_files = {
        "Base Predictor Interface": "src/models/base/predictor.py",
        "LSTM Predictor": "src/models/base/lstm_predictor.py",
        "XGBoost Predictor": "src/models/base/xgboost_predictor.py",
        "HMM Predictor": "src/models/base/hmm_predictor.py",
        "Ensemble Model": "src/models/ensemble.py",
        "Models Package": "src/models/__init__.py",
        "Base Package": "src/models/base/__init__.py",
        "Constants": "src/core/constants.py",
        "Exceptions": "src/core/exceptions.py",
    }

    missing_files = []
    present_files = []

    for name, file_path in required_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            present_files.append((name, file_path))
            print(f"  [PASS] {name}: {file_path}")
        else:
            missing_files.append((name, file_path))
            print(f"  [FAIL] {name}: {file_path} - NOT FOUND")

    return len(missing_files) == 0, present_files, missing_files


def check_file_contents():
    """Check basic content structure of key files."""
    print("\nChecking Sprint 3 file contents...")

    base_path = Path(__file__).parent

    checks = {
        "BasePredictor class": ("src/models/base/predictor.py", "class BasePredictor"),
        "PredictionResult class": (
            "src/models/base/predictor.py",
            "class PredictionResult",
        ),
        "TrainingResult class": (
            "src/models/base/predictor.py",
            "class TrainingResult",
        ),
        "LSTMPredictor class": (
            "src/models/base/lstm_predictor.py",
            "class LSTMPredictor",
        ),
        "XGBoostPredictor class": (
            "src/models/base/xgboost_predictor.py",
            "class XGBoostPredictor",
        ),
        "HMMPredictor class": (
            "src/models/base/hmm_predictor.py",
            "class HMMPredictor",
        ),
        "OccupancyEnsemble class": (
            "src/models/ensemble.py",
            "class OccupancyEnsemble",
        ),
        "ModelType enum": ("src/core/constants.py", "class ModelType"),
        "Abstract methods": ("src/models/base/predictor.py", "@abstractmethod"),
        "Train method": ("src/models/base/predictor.py", "async def train"),
        "Predict method": ("src/models/base/predictor.py", "async def predict"),
    }

    passed_checks = []
    failed_checks = []

    for check_name, (file_path, expected_content) in checks.items():
        full_path = base_path / file_path

        try:
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                if expected_content in content:
                    passed_checks.append(check_name)
                    print(f"  [PASS] {check_name}")
                else:
                    failed_checks.append(check_name)
                    print(f"  [FAIL] {check_name} - Expected content not found")
            else:
                failed_checks.append(check_name)
                print(f"  [FAIL] {check_name} - File not found")
        except Exception as e:
            failed_checks.append(check_name)
            print(f"  [FAIL] {check_name} - Error reading file: {e}")

    return len(failed_checks) == 0, passed_checks, failed_checks


def check_class_inheritance():
    """Check that predictors inherit from BasePredictor."""
    print("\nChecking class inheritance...")

    base_path = Path(__file__).parent

    inheritance_checks = [
        ("LSTMPredictor", "src/models/base/lstm_predictor.py", "BasePredictor"),
        ("XGBoostPredictor", "src/models/base/xgboost_predictor.py", "BasePredictor"),
        ("HMMPredictor", "src/models/base/hmm_predictor.py", "BasePredictor"),
        ("OccupancyEnsemble", "src/models/ensemble.py", "BasePredictor"),
    ]

    passed_inheritance = []
    failed_inheritance = []

    for class_name, file_path, base_class in inheritance_checks:
        full_path = base_path / file_path

        try:
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                # Look for class definition with inheritance
                if f"class {class_name}({base_class})" in content:
                    passed_inheritance.append(class_name)
                    print(f"  [PASS] {class_name} inherits from {base_class}")
                else:
                    failed_inheritance.append(class_name)
                    print(
                        f"  [FAIL] {class_name} does not properly inherit from {base_class}"
                    )
            else:
                failed_inheritance.append(class_name)
                print(f"  [FAIL] {class_name} - File not found")
        except Exception as e:
            failed_inheritance.append(class_name)
            print(f"  [FAIL] {class_name} - Error reading file: {e}")

    return len(failed_inheritance) == 0, passed_inheritance, failed_inheritance


def check_required_methods():
    """Check that required abstract methods are implemented."""
    print("\nChecking required method implementations...")

    base_path = Path(__file__).parent

    # Methods that should be in BasePredictor
    base_methods = [
        "async def train",
        "async def predict",
        "def get_feature_importance",
        "def save_model",
        "def load_model",
        "def get_model_info",
        "def validate_features",
    ]

    method_checks = []
    for method in base_methods:
        predictor_file = base_path / "src/models/base/predictor.py"
        if predictor_file.exists():
            content = predictor_file.read_text(encoding="utf-8")
            if method in content:
                method_checks.append((method, True))
                print(f"  [PASS] {method} found in BasePredictor")
            else:
                method_checks.append((method, False))
                print(f"  [FAIL] {method} not found in BasePredictor")
        else:
            method_checks.append((method, False))
            print(f"  [FAIL] BasePredictor file not found")

    passed_methods = sum(1 for _, passed in method_checks if passed)
    return passed_methods == len(base_methods), method_checks


def check_dataclass_structure():
    """Check that PredictionResult and TrainingResult are properly defined."""
    print("\nChecking dataclass structures...")

    base_path = Path(__file__).parent
    predictor_file = base_path / "src/models/base/predictor.py"

    dataclass_checks = []

    if predictor_file.exists():
        content = predictor_file.read_text(encoding="utf-8")

        # Check for PredictionResult
        if "@dataclass" in content and "class PredictionResult" in content:
            dataclass_checks.append(("PredictionResult dataclass", True))
            print("  [PASS] PredictionResult dataclass found")
        else:
            dataclass_checks.append(("PredictionResult dataclass", False))
            print("  [FAIL] PredictionResult dataclass not found")

        # Check for TrainingResult
        if "@dataclass" in content and "class TrainingResult" in content:
            dataclass_checks.append(("TrainingResult dataclass", True))
            print("  [PASS] TrainingResult dataclass found")
        else:
            dataclass_checks.append(("TrainingResult dataclass", False))
            print("  [FAIL] TrainingResult dataclass not found")

        # Check for to_dict methods
        if "def to_dict(self)" in content:
            dataclass_checks.append(("to_dict methods", True))
            print("  [PASS] to_dict methods found")
        else:
            dataclass_checks.append(("to_dict methods", False))
            print("  [FAIL] to_dict methods not found")
    else:
        dataclass_checks.append(("File exists", False))
        print("  [FAIL] predictor.py file not found")

    passed_dataclass = sum(1 for _, passed in dataclass_checks if passed)
    return passed_dataclass == len(dataclass_checks), dataclass_checks


def check_ensemble_architecture():
    """Check ensemble-specific implementation details."""
    print("\nChecking ensemble architecture...")

    base_path = Path(__file__).parent
    ensemble_file = base_path / "src/models/ensemble.py"

    ensemble_checks = []

    if ensemble_file.exists():
        content = ensemble_file.read_text(encoding="utf-8")

        # Key ensemble components
        ensemble_features = [
            ("base_models attribute", "base_models"),
            ("model_weights attribute", "model_weights"),
            ("LSTM base model", "LSTMPredictor"),
            ("XGBoost base model", "XGBoostPredictor"),
            ("HMM base model", "HMMPredictor"),
            ("Ensemble info method", "get_ensemble_info"),
            ("Prediction combination", "_combine_predictions"),
        ]

        for feature_name, search_text in ensemble_features:
            if search_text in content:
                ensemble_checks.append((feature_name, True))
                print(f"  [PASS] {feature_name} found")
            else:
                ensemble_checks.append((feature_name, False))
                print(f"  [FAIL] {feature_name} not found")
    else:
        ensemble_checks.append(("File exists", False))
        print("  [FAIL] ensemble.py file not found")

    passed_ensemble = sum(1 for _, passed in ensemble_checks if passed)
    return passed_ensemble == len(ensemble_checks), ensemble_checks


def main():
    """Run Sprint 3 validation checks."""
    print("=" * 60)
    print("Sprint 3 Model Development Validation (Simple)")
    print("=" * 60)
    print("This validation checks Sprint 3 file structure and basic")
    print("implementation without requiring external dependencies.")
    print("-" * 60)

    all_checks = []

    # Run all validation checks
    checks = [
        ("File Structure", check_file_structure),
        ("File Contents", check_file_contents),
        ("Class Inheritance", check_class_inheritance),
        ("Required Methods", check_required_methods),
        ("Dataclass Structure", check_dataclass_structure),
        ("Ensemble Architecture", check_ensemble_architecture),
    ]

    total_passed = 0
    total_failed = 0

    for check_name, check_func in checks:
        try:
            passed, *details = check_func()
            if passed:
                total_passed += 1
                print(f"\n[PASS] {check_name} validation completed successfully")
            else:
                total_failed += 1
                print(f"\n[FAIL] {check_name} validation failed")
            all_checks.append((check_name, passed, details))
        except Exception as e:
            total_failed += 1
            print(f"\n[FAIL] {check_name} validation failed with exception: {e}")
            all_checks.append((check_name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("Sprint 3 Validation Summary:")
    print(f"  [PASS] Passed: {total_passed}")
    print(f"  [FAIL] Failed: {total_failed}")
    print(f"  Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")

    if total_failed == 0:
        print("\n[SUCCESS] All Sprint 3 structural validations PASSED!")
        print("Sprint 3 Model Development appears to be properly implemented.")
        print("\nNote: This validation checks structure and interfaces only.")
        print("To test actual functionality, install requirements.txt and run:")
        print("  pip install -r requirements.txt")
        print("  python validate_sprint3.py")
        return True
    else:
        print(f"\n[ERROR] {total_failed} Sprint 3 validations FAILED!")
        print("Review the failed checks above and fix before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
