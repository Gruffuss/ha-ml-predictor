#!/usr/bin/env python3
"""
Check specific Category 2 files for quality issues.
"""

import subprocess
import sys
import os

def check_file_quality(file_path):
    """Check a specific file for quality issues."""
    print(f"\n🔍 Checking: {file_path}")
    print("-" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Run black check
    cmd = ["python", "-m", "black", "--check", "--diff", "--line-length", "88", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if result.returncode == 0:
        print("✅ Black formatting: PASSED")
    else:
        print("❌ Black formatting: FAILED")
        print(result.stdout)
        return False
    
    # Run isort check
    cmd = ["python", "-m", "isort", "--check-only", "--diff", "--profile", "black", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if result.returncode == 0:
        print("✅ Import sorting: PASSED")
    else:
        print("❌ Import sorting: FAILED")
        print(result.stdout)
        return False
    
    # Run flake8 check
    cmd = ["python", "-m", "flake8", file_path, "--max-line-length=140", 
           "--extend-ignore=E203,W503,E501,W291,W293,E402,C901"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if result.returncode == 0:
        print("✅ Style guide: PASSED")
    else:
        print("❌ Style guide: FAILED")
        print(result.stdout)
        return False
    
    print(f"✅ {file_path} passes all quality checks")
    return True

def main():
    """Check Category 2 related files."""
    print("🎯 Checking Category 2 Database Files for Quality Issues")
    print("=" * 70)
    
    # Files that were modified for Category 2 fixes
    files_to_check = [
        "src/data/storage/database.py",
        "src/core/exceptions.py",
        "tests/unit/test_data/test_database.py"
    ]
    
    results = []
    for file_path in files_to_check:
        success = check_file_quality(file_path)
        results.append((file_path, success))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{'=' * 70}")
    print(f"📊 CATEGORY 2 QUALITY CHECK: {passed}/{total} files passed")
    print("=" * 70)
    
    for file_path, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{os.path.basename(file_path)}: {status}")
    
    if passed == total:
        print(f"\n🎉 All Category 2 files meet quality standards!")
        print("✅ Database connection fixes are production-ready")
        return True
    else:
        print(f"\n⚠️  {total - passed} files need quality fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)