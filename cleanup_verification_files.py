#!/usr/bin/env python3
"""
Clean up temporary verification files.
"""

import os

files_to_remove = [
    "test_category2_verification.py",
    "run_db_test.py",
    "run_quality_pipeline.py",
    "check_specific_files.py",
    "cleanup_verification_files.py"
]

for file_path in files_to_remove:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✅ Removed {file_path}")
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
    else:
        print(f"ℹ️  File not found: {file_path}")

print("\n🧹 Cleanup complete")
