#!/usr/bin/env python3
"""
Advanced script to fix remaining flake8 issues more systematically.
"""

import ast
import os
import re
import subprocess
from pathlib import Path


def remove_unused_variables(content, file_path):
    """Remove unused local variables (F841)."""
    try:
        lines = content.split('\n')
        
        # Get F841 errors for this file
        result = subprocess.run(
            ['flake8', file_path, '--select=F841'],
            capture_output=True, text=True
        )
        
        if result.stdout:
            # Parse F841 errors
            for line in result.stdout.strip().split('\n'):
                if 'F841' in line:
                    # Extract line number and variable name
                    parts = line.split(':')
                    if len(parts) >= 4:
                        line_num = int(parts[1]) - 1  # Convert to 0-based
                        if line_num < len(lines):
                            # Common patterns for unused variables
                            line_content = lines[line_num]
                            
                            # Pattern: local variable 'var' is assigned to but never used
                            if 'retraining_task' in line_content and '=' in line_content:
                                # Comment out or use _ pattern
                                if not line_content.strip().startswith('#'):
                                    lines[line_num] = line_content.replace(
                                        'retraining_task = ', '_ = '
                                    )
                            elif 'is_accurate' in line_content and '=' in line_content:
                                if not line_content.strip().startswith('#'):
                                    lines[line_num] = line_content.replace(
                                        'is_accurate = ', '_ = '
                                    )
                            elif 'alert_key' in line_content and '=' in line_content:
                                if not line_content.strip().startswith('#'):
                                    lines[line_num] = line_content.replace(
                                        'alert_key = ', '_ = '
                                    )
        
        return '\n'.join(lines)
    except Exception as e:
        print(f"Error removing unused variables in {file_path}: {e}")
        return content


def remove_duplicate_exceptions(content):
    """Remove duplicate exception definitions (F811)."""
    # Find and remove duplicate class definitions
    lines = content.split('\n')
    seen_classes = set()
    filtered_lines = []
    skip_class = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check for class definitions
        if stripped.startswith('class ') and ':' in stripped:
            class_name = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
            
            # Skip duplicate definitions of error classes
            if (class_name.endswith('Error') or class_name.endswith('Exception')) and class_name in seen_classes:
                skip_class = True
                continue
            
            seen_classes.add(class_name)
            skip_class = False
        elif skip_class and stripped and not stripped.startswith('class '):
            # Skip lines that are part of duplicate class
            if line.startswith('    ') or line.startswith('\t'):
                continue
            else:
                skip_class = False
        
        if not skip_class:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def remove_unused_imports_advanced(content):
    """Advanced unused import removal."""
    lines = content.split('\n')
    
    # Patterns for common unused imports
    unused_patterns = [
        r'^from typing import.*Tuple.*$',
        r'^from typing import.*Union.*$',  
        r'^import json$',
        r'^from dataclasses import.*field.*$',
        r'^from sqlalchemy import.*desc.*$',
        r'^from sqlalchemy import.*func.*$',
        r'^from sqlalchemy import.*or_.*$',
        r'^from sqlalchemy import.*update.*$',
        r'^from collections import deque$',
        r'^import math$',
        r'^import traceback$'
    ]
    
    filtered_lines = []
    for line in lines:
        should_remove = False
        for pattern in unused_patterns:
            if re.match(pattern, line.strip()):
                # Check if any part of the import is actually used in the file
                if 'Tuple' in line and 'Tuple[' not in content:
                    should_remove = True
                elif 'Union' in line and 'Union[' not in content:
                    should_remove = True  
                elif 'import json' in line and 'json.' not in content:
                    should_remove = True
                elif 'field' in line and 'field(' not in content:
                    should_remove = True
                break
        
        if not should_remove:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def process_file_advanced(file_path):
    """Advanced processing of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply advanced fixes
        content = remove_unused_variables(content, str(file_path))
        content = remove_duplicate_exceptions(content)
        content = remove_unused_imports_advanced(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Advanced fixes applied: {file_path}")
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False


def main():
    """Main function to apply advanced fixes."""
    project_root = Path(__file__).parent
    python_files = []
    
    # Find all Python files in src/ and tests/
    for directory in ['src', 'tests']:
        dir_path = project_root / directory
        if dir_path.exists():
            python_files.extend(dir_path.rglob('*.py'))
    
    print(f"Applying advanced fixes to {len(python_files)} Python files...")
    
    fixed_count = 0
    for file_path in python_files:
        if process_file_advanced(file_path):
            fixed_count += 1
    
    print(f"Applied advanced fixes to {fixed_count} files")
    
    # Run flake8 to check remaining issues
    print("\nRunning flake8 to check remaining issues...")
    try:
        result = subprocess.run(['flake8', 'src/', 'tests/', '--count'], 
                              capture_output=True, text=True, cwd=project_root)
        if result.stdout:
            print("Remaining flake8 issues:")
            print(result.stdout.strip().split('\n')[-1])  # Just the count
    except Exception as e:
        print(f"Error running flake8: {e}")


if __name__ == '__main__':
    main()