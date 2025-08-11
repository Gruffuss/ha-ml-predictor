#!/usr/bin/env python3
"""Quick mypy check script to count errors and provide summary."""

import subprocess

def run_mypy():
    """Run mypy and count errors."""
    try:
        result = subprocess.run([
            'python', '-m', 'mypy', 'src/', 
            '--ignore-missing-imports', 
            '--no-strict-optional'
        ], capture_output=True, text=True, timeout=300)
        
        error_lines = [line for line in result.stdout.split('\n') if 'error:' in line]
        
        print(f"Total mypy errors found: {len(error_lines)}")
        print(f"Return code: {result.returncode}")
        
        # Categorize errors
        categories = {}
        for line in error_lines:
            if 'Missing positional argument' in line:
                categories.setdefault('Missing arguments', 0)
                categories['Missing arguments'] += 1
            elif 'Incompatible types' in line:
                categories.setdefault('Type mismatches', 0)
                categories['Type mismatches'] += 1
            elif 'has no attribute' in line:
                categories.setdefault('Missing attributes', 0)
                categories['Missing attributes'] += 1
            elif 'Need type annotation' in line:
                categories.setdefault('Missing annotations', 0)
                categories['Missing annotations'] += 1
            else:
                categories.setdefault('Other', 0)
                categories['Other'] += 1
        
        print("\nError categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
        
        # Show first 20 errors
        print("\nFirst 20 errors:")
        for line in error_lines[:20]:
            print(f"  {line}")
            
        return len(error_lines), result.returncode
        
    except subprocess.TimeoutExpired:
        print("Mypy check timed out after 5 minutes")
        return -1, -1
    except Exception as e:
        print(f"Error running mypy: {e}")
        return -1, -1

if __name__ == "__main__":
    error_count, return_code = run_mypy()
    print(f"\nFinal result: {error_count} errors, exit code: {return_code}")