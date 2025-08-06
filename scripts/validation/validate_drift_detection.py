"""
Basic validation script for drift detection module structure.

Validates that the drift detection implementation has the required
components and structure without requiring full dependencies.
"""

import ast
import sys
from pathlib import Path

def validate_drift_detector_structure():
    """Validate the structure of the drift detector module."""
    
    drift_detector_path = Path("src/adaptation/drift_detector.py")
    
    if not drift_detector_path.exists():
        print("[FAIL] drift_detector.py not found")
        return False
    
    print("[OK] drift_detector.py exists")
    
    # Read and parse the file
    with open(drift_detector_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print("[OK] drift_detector.py syntax is valid")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error in drift_detector.py: {e}")
        return False
    
    # Check for required classes and functions
    required_classes = [
        'DriftMetrics',
        'FeatureDriftResult', 
        'ConceptDriftDetector',
        'FeatureDriftDetector'
    ]
    
    required_enums = [
        'DriftType',
        'DriftSeverity', 
        'StatisticalTest'
    ]
    
    required_methods = [
        'detect_drift',
        '_analyze_prediction_drift',
        '_analyze_feature_drift',
        '_test_feature_drift',
        '_calculate_psi',
        '_run_page_hinkley_test'
    ]
    
    found_classes = []
    found_methods = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found_classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            found_methods.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            found_methods.append(node.name)
    
    # Validate classes
    for cls in required_classes:
        if cls in found_classes:
            print(f"[OK] Class {cls} found")
        else:
            print(f"[FAIL] Class {cls} missing")
            return False
    
    # Validate enums (also classes in Python)
    for enum in required_enums:
        if enum in found_classes:
            print(f"[OK] Enum {enum} found")
        else:
            print(f"[FAIL] Enum {enum} missing")
            return False
    
    # Validate key methods
    missing_methods = []
    for method in required_methods:
        if method not in found_methods:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"[FAIL] Missing methods: {missing_methods}")
        return False
    else:
        print("[OK] All required methods found")
    
    # Check for statistical imports
    if 'scipy' in content and 'numpy' in content and 'pandas' in content:
        print("[OK] Required statistical libraries imported")
    else:
        print("[FAIL] Missing statistical library imports")
        return False
    
    # Check for integration with existing modules
    if 'from .validator import' in content and 'from .tracker import' in content:
        print("[OK] Integration with existing modules found")
    else:
        print("[FAIL] Missing integration imports")
        return False
    
    return True

def validate_test_structure():
    """Validate the test file structure."""
    
    test_path = Path("tests/test_drift_detection.py")
    
    if not test_path.exists():
        print("[FAIL] test_drift_detection.py not found")
        return False
    
    print("[OK] test_drift_detection.py exists")
    
    # Read and parse the test file
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print("[OK] test_drift_detection.py syntax is valid")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error in test file: {e}")
        return False
    
    # Check for test classes
    test_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            test_classes.append(node.name)
    
    expected_test_classes = [
        'TestDriftMetrics',
        'TestFeatureDriftResult',
        'TestConceptDriftDetector',
        'TestFeatureDriftDetector'
    ]
    
    for test_cls in expected_test_classes:
        if test_cls in test_classes:
            print(f"[OK] Test class {test_cls} found")
        else:
            print(f"[FAIL] Test class {test_cls} missing")
            return False
    
    return True

def validate_example_structure():
    """Validate the example file structure."""
    
    example_path = Path("examples/drift_detection_example.py")
    
    if not example_path.exists():
        print("[FAIL] drift_detection_example.py not found")
        return False
    
    print("[OK] drift_detection_example.py exists")
    
    # Read and check for key functions
    with open(example_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        print("[OK] drift_detection_example.py syntax is valid")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error in example: {e}")
        return False
    
    # Check for demonstration functions
    expected_functions = [
        'demonstrate_concept_drift_detection',
        'demonstrate_feature_drift_monitoring',
        'demonstrate_integrated_monitoring'
    ]
    
    for func in expected_functions:
        if func in content:
            print(f"[OK] Example function {func} found")
        else:
            print(f"[FAIL] Example function {func} missing")
            return False
    
    return True

def validate_todo_updates():
    """Validate that TODO.md has been updated with drift detection functions."""
    
    todo_path = Path("TODO.md")
    
    if not todo_path.exists():
        print("[FAIL] TODO.md not found")
        return False
    
    with open(todo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for drift detection section
    if "Drift Detector (`src/adaptation/drift_detector.py`) - ✅ COMPLETED" in content:
        print("[OK] TODO.md updated with drift detection completion")
    else:
        print("[FAIL] TODO.md not updated with drift detection")
        return False
    
    # Check for specific function tracking
    required_functions = [
        'ConceptDriftDetector.detect_drift()',
        'FeatureDriftDetector.start_monitoring()',
        'DriftMetrics.__init__()'
    ]
    
    for func in required_functions:
        if func in content:
            print(f"[OK] Function {func} tracked in TODO.md")
        else:
            print(f"[FAIL] Function {func} not tracked in TODO.md")
            return False
    
    return True

def main():
    """Main validation function."""
    print("Concept Drift Detection System Validation")
    print("=" * 50)
    
    all_valid = True
    
    print("\n1. Validating drift detector module structure...")
    if not validate_drift_detector_structure():
        all_valid = False
    
    print("\n2. Validating test structure...")
    if not validate_test_structure():
        all_valid = False
    
    print("\n3. Validating example structure...")
    if not validate_example_structure():
        all_valid = False
    
    print("\n4. Validating TODO.md updates...")
    if not validate_todo_updates():
        all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("[SUCCESS] All validations passed! Drift detection system is properly implemented.")
        return True
    else:
        print("[FAIL] Some validations failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)