#!/usr/bin/env python3
"""
Script to systematically fix common flake8 issues in the codebase.
"""

import os
import re
import subprocess
from pathlib import Path


def fix_boolean_comparisons(content):
    """Fix E712 - comparison to False should be 'if cond is False:' or 'if not cond:'"""
    # Fix == False comparisons
    content = re.sub(r'\b(\w+)\s*==\s*False\b', r'\1 is False', content)
    # Fix == True comparisons
    content = re.sub(r'\b(\w+)\s*==\s*True\b', r'\1 is True', content)
    # Fix != False comparisons
    content = re.sub(r'\b(\w+)\s*!=\s*False\b', r'\1 is not False', content)
    # Fix != True comparisons
    content = re.sub(r'\b(\w+)\s*!=\s*True\b', r'\1 is not True', content)
    return content


def fix_fstring_placeholders(content):
    """Fix F541 - f-string is missing placeholders"""
    # Find f-strings without placeholders and convert to regular strings
    content = re.sub(r'f"([^"{]*)"(?![^"]*{)', r'"\1"', content)
    content = re.sub(r"f'([^'{]*)'(?![^']*{)", r"'\1'", content)
    return content


def fix_trailing_whitespace(content):
    """Fix W291 - trailing whitespace and W293 - blank line contains whitespace"""
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]
    return '\n'.join(fixed_lines)


def fix_whitespace_before_colon(content):
    """Fix E203 - whitespace before ':'"""
    # Fix whitespace before colon in slices and dictionary definitions
    content = re.sub(r'\s+:', ':', content)
    return content


def fix_modulo_operator(content):
    """Fix E228 - missing whitespace around modulo operator"""
    # Add whitespace around % operator
    content = re.sub(r'(\w+)%(\w+)', r'\1 % \2', content)
    return content


def process_file(file_path):
    """Process a single Python file to fix flake8 issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_boolean_comparisons(content)
        content = fix_fstring_placeholders(content)
        content = fix_trailing_whitespace(content)
        content = fix_whitespace_before_colon(content)
        content = fix_modulo_operator(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False


def main():
    """Main function to process all Python files."""
    project_root = Path(__file__).parent
    python_files = []
    
    # Find all Python files in src/ and tests/
    for directory in ['src', 'tests']:
        dir_path = project_root / directory
        if dir_path.exists():
            python_files.extend(dir_path.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files to process...")
    
    fixed_count = 0
    for file_path in python_files:
        if process_file(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")
    
    # Run flake8 to check remaining issues
    print("\nRunning flake8 to check remaining issues...")
    try:
        result = subprocess.run(['flake8', 'src/', 'tests/', '--count', '--statistics'], 
                              capture_output=True, text=True, cwd=project_root)
        if result.stdout:
            print("Remaining flake8 issues:")
            print(result.stdout)
        if result.stderr:
            print("Flake8 stderr:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running flake8: {e}")


if __name__ == '__main__':
    main()