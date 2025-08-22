#!/usr/bin/env python3
"""
Script to run pytest and generate comprehensive error report.
"""

import subprocess
import sys
from pathlib import Path
import datetime
import re
from collections import defaultdict

def run_pytest():
    """Run pytest and capture output."""
    cmd = [
        sys.executable, 
        "-m", 
        "pytest", 
        "tests/", 
        "-v", 
        "--tb=short", 
        "--color=no",
        "--durations=10",
        "--showlocals"
    ]
    
    print("Running pytest command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    
    return result.returncode, result.stdout, result.stderr

def parse_test_results(stdout, stderr):
    """Parse pytest output to extract test results."""
    results = {
        'passed': [],
        'failed': [],
        'skipped': [],
        'errors': [],
        'summary': {},
        'durations': [],
        'warnings': []
    }
    
    # Parse test results from stdout
    lines = stdout.split('\n')
    
    for i, line in enumerate(lines):
        # Parse individual test results
        if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line or 'ERROR' in line):
            if 'PASSED' in line:
                results['passed'].append(line.strip())
            elif 'FAILED' in line:
                results['failed'].append(line.strip())
            elif 'SKIPPED' in line:
                results['skipped'].append(line.strip())
            elif 'ERROR' in line:
                results['errors'].append(line.strip())
        
        # Parse summary line
        if 'failed' in line and 'passed' in line and '=' in line:
            results['summary']['summary_line'] = line.strip()
        
        # Parse durations
        if 'slowest durations' in line.lower():
            # Look for next lines with durations
            for j in range(i+1, min(i+15, len(lines))):
                if re.match(r'^\d+\.\d+s', lines[j]):
                    results['durations'].append(lines[j].strip())
        
        # Parse warnings
        if 'warning' in line.lower() and ('pytest' in line.lower() or 'deprecation' in line.lower()):
            results['warnings'].append(line.strip())
    
    return results

def extract_failure_details(stdout, stderr):
    """Extract detailed failure information."""
    failure_details = []
    
    # Split output into sections
    lines = stdout.split('\n')
    
    current_failure = None
    collecting_failure = False
    
    for line in lines:
        if line.startswith('FAILURES'):
            collecting_failure = True
            continue
        
        if collecting_failure:
            if line.startswith('='):
                if current_failure:
                    failure_details.append(current_failure)
                if 'FAILED' in line and '::' in line:
                    current_failure = {
                        'test_name': line.strip('= '),
                        'traceback': [],
                        'assertion': '',
                        'locals': []
                    }
                elif not any(x in line for x in ['FAILED', 'ERROR', 'short test summary']):
                    collecting_failure = False
            elif current_failure:
                if line.strip():
                    if 'AssertionError' in line or 'Error:' in line or 'Exception:' in line:
                        current_failure['assertion'] = line.strip()
                    elif '>' in line and 'assert' in line:
                        current_failure['assertion'] = line.strip()
                    elif 'self =' in line or 'cls =' in line:
                        current_failure['locals'].append(line.strip())
                    else:
                        current_failure['traceback'].append(line.strip())
    
    if current_failure:
        failure_details.append(current_failure)
    
    return failure_details

def categorize_errors(failure_details):
    """Categorize errors by type."""
    categories = defaultdict(list)
    
    for failure in failure_details:
        test_name = failure.get('test_name', '')
        assertion = failure.get('assertion', '')
        traceback = ' '.join(failure.get('traceback', []))
        
        # Categorize based on error patterns
        if 'ImportError' in assertion or 'ModuleNotFoundError' in assertion:
            categories['Import Errors'].append(failure)
        elif 'AttributeError' in assertion:
            categories['Attribute Errors'].append(failure)
        elif 'TypeError' in assertion:
            categories['Type Errors'].append(failure)
        elif 'ValueError' in assertion:
            categories['Value Errors'].append(failure)
        elif 'AssertionError' in assertion:
            categories['Assertion Failures'].append(failure)
        elif 'fixture' in traceback.lower():
            categories['Fixture Errors'].append(failure)
        elif 'config' in test_name.lower():
            categories['Configuration Errors'].append(failure)
        elif 'model' in test_name.lower():
            categories['Model Errors'].append(failure)
        else:
            categories['Other Errors'].append(failure)
    
    return categories

def generate_recommendations(categories):
    """Generate recommendations based on error categories."""
    recommendations = []
    
    for category, failures in categories.items():
        if category == 'Import Errors':
            recommendations.append(
                f"• {category} ({len(failures)} tests): Check if all required dependencies are installed. "
                "Run 'pip install -r requirements.txt' to ensure all packages are available."
            )
        elif category == 'Configuration Errors':
            recommendations.append(
                f"• {category} ({len(failures)} tests): Verify configuration files exist and have correct format. "
                "Check config.yaml and rooms.yaml in the config/ directory."
            )
        elif category == 'Model Errors':
            recommendations.append(
                f"• {category} ({len(failures)} tests): Review model parameter configurations and "
                "ensure model initialization parameters are compatible."
            )
        elif category == 'Fixture Errors':
            recommendations.append(
                f"• {category} ({len(failures)} tests): Check test fixture definitions and ensure "
                "all fixture dependencies are properly set up."
            )
        elif category == 'Assertion Failures':
            recommendations.append(
                f"• {category} ({len(failures)} tests): Review test expectations vs actual behavior. "
                "These may indicate functional issues in the code being tested."
            )
        else:
            recommendations.append(
                f"• {category} ({len(failures)} tests): Review individual error messages for specific fixes."
            )
    
    return recommendations

def create_error_report(return_code, stdout, stderr, results, failure_details, categories):
    """Create comprehensive error report."""
    report = []
    
    # Header
    report.append("# PYTEST ERROR REPORT")
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Working Directory: {Path.cwd()}")
    report.append("")
    
    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("")
    
    total_tests = len(results['passed']) + len(results['failed']) + len(results['skipped']) + len(results['errors'])
    report.append(f"**Total Tests:** {total_tests}")
    report.append(f"**Passed:** {len(results['passed'])}")
    report.append(f"**Failed:** {len(results['failed'])}")
    report.append(f"**Errors:** {len(results['errors'])}")
    report.append(f"**Skipped:** {len(results['skipped'])}")
    report.append("")
    
    if return_code == 0:
        report.append("🟢 **Status:** ALL TESTS PASSED")
    else:
        report.append("🔴 **Status:** TESTS FAILED")
    
    report.append("")
    
    # Summary Statistics
    if results.get('summary', {}).get('summary_line'):
        report.append(f"**pytest Summary:** {results['summary']['summary_line']}")
        report.append("")
    
    # Error Categories
    if categories:
        report.append("## ERROR ANALYSIS BY CATEGORY")
        report.append("")
        
        for category, failures in categories.items():
            report.append(f"### {category} ({len(failures)} tests)")
            report.append("")
            for failure in failures:
                test_name = failure.get('test_name', 'Unknown')
                report.append(f"- **{test_name}**")
                if failure.get('assertion'):
                    report.append(f"  - Error: `{failure['assertion']}`")
                report.append("")
    
    # Recommendations
    if categories:
        report.append("## RECOMMENDATIONS")
        report.append("")
        recommendations = generate_recommendations(categories)
        for rec in recommendations:
            report.append(rec)
        report.append("")
    
    # Detailed Failure Analysis
    if failure_details:
        report.append("## DETAILED FAILURE ANALYSIS")
        report.append("")
        
        for i, failure in enumerate(failure_details, 1):
            report.append(f"### Failure #{i}: {failure.get('test_name', 'Unknown')}")
            report.append("")
            
            if failure.get('assertion'):
                report.append("**Error Message:**")
                report.append(f"```")
                report.append(failure['assertion'])
                report.append("```")
                report.append("")
            
            if failure.get('traceback'):
                report.append("**Traceback:**")
                report.append("```")
                for line in failure['traceback'][-10:]:  # Last 10 lines
                    if line.strip():
                        report.append(line)
                report.append("```")
                report.append("")
            
            if failure.get('locals'):
                report.append("**Local Variables:**")
                report.append("```")
                for local in failure['locals'][:5]:  # First 5 locals
                    report.append(local)
                report.append("```")
                report.append("")
    
    # Performance Analysis
    if results.get('durations'):
        report.append("## PERFORMANCE ANALYSIS")
        report.append("")
        report.append("**Slowest Tests:**")
        for duration in results['durations'][:10]:
            report.append(f"- {duration}")
        report.append("")
    
    # Warnings
    if results.get('warnings'):
        report.append("## WARNINGS")
        report.append("")
        for warning in results['warnings'][:10]:
            report.append(f"- {warning}")
        report.append("")
    
    # Full pytest Output
    report.append("## FULL PYTEST OUTPUT")
    report.append("")
    report.append("### STDOUT")
    report.append("```")
    # Limit output to reasonable size
    stdout_lines = stdout.split('\n')
    if len(stdout_lines) > 200:
        report.append(f"... (showing last 200 lines of {len(stdout_lines)} total)")
        report.extend(stdout_lines[-200:])
    else:
        report.append(stdout)
    report.append("```")
    report.append("")
    
    if stderr.strip():
        report.append("### STDERR")
        report.append("```")
        stderr_lines = stderr.split('\n')
        if len(stderr_lines) > 50:
            report.append(f"... (showing last 50 lines of {len(stderr_lines)} total)")
            report.extend(stderr_lines[-50:])
        else:
            report.append(stderr)
        report.append("```")
        report.append("")
    
    # Next Steps
    report.append("## NEXT STEPS")
    report.append("")
    report.append("1. **Address Import Errors**: Ensure all dependencies are installed")
    report.append("2. **Fix Configuration Issues**: Verify config files are present and valid")
    report.append("3. **Resolve Model Parameter Issues**: Review model initialization parameters")
    report.append("4. **Update Test Fixtures**: Fix any broken test setup code")
    report.append("5. **Run Specific Tests**: Use `pytest tests/path/to/specific_test.py -v` for targeted debugging")
    report.append("")
    
    return '\n'.join(report)

def main():
    """Main function to run tests and generate report."""
    print("Starting pytest execution...")
    
    return_code, stdout, stderr = run_pytest()
    
    print(f"pytest finished with return code: {return_code}")
    
    # Parse results
    results = parse_test_results(stdout, stderr)
    failure_details = extract_failure_details(stdout, stderr)
    categories = categorize_errors(failure_details)
    
    # Generate report
    report = create_error_report(return_code, stdout, stderr, results, failure_details, categories)
    
    # Write report to file
    report_file = Path("PYTEST_ERROR_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"ERROR REPORT GENERATED: {report_file.absolute()}")
    print(f"{'='*60}")
    
    # Print summary to console
    total_tests = len(results['passed']) + len(results['failed']) + len(results['skipped']) + len(results['errors'])
    print(f"Total tests: {total_tests}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Skipped: {len(results['skipped'])}")
    
    if categories:
        print(f"\nError categories found:")
        for category, failures in categories.items():
            print(f"  - {category}: {len(failures)} tests")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())