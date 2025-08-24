#!/usr/bin/env python3
"""
Script to fix configuration system test failures by adding proper JWT mocking.
"""

import re

def fix_api_tests():
    """Fix all API configuration tests by adding JWT_ENABLED: false mocking."""
    
    file_path = "tests/unit/core_system/test_configuration_system.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix patterns for APIConfig tests that need JWT disabled
    api_test_patterns = [
        (
            r'(def test_api_config_[^(]+\([^)]+\):\s*"""[^"]*"""\s*mock_getenv\.side_effect = lambda key, default[^:]*: \{\s*)',
            r'\1            "JWT_ENABLED": "false",  # Disable JWT for this test\n'
        ),
        (
            r'(mock_getenv\.side_effect = lambda key, default[^:]*: \{[^}]*)\}\.get\(key, str\(default\) if default is not None else ""\)',
            r'\1            "JWT_ENABLED": "false",  # Disable JWT for this test\n        }.get(key, str(default) if default is not None else "")'
        )
    ]
    
    # Apply fixes
    for pattern, replacement in api_test_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
    
    # Fix specific test methods that need JWT disabled but don't follow the pattern
    specific_fixes = [
        # test_api_config_default_cors_origins_splitting
        (
            r'mock_getenv\.side_effect = lambda key, default: \{\s*\}\.get\(key, str\(default\) if default is not None else "\*"\)',
            '''mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "false",  # Disable JWT for this test
        }.get(key, str(default) if default is not None else "*")'''
        ),
        # Large room configuration test
        (
            r'(with patch\(\'pathlib\.Path\.exists\', return_value=True\), \\\s*patch\(\'yaml\.safe_load\', side_effect=\[sample_config, large_rooms_config\]\):)',
            r'''with patch('os.getenv') as mock_getenv, \\
             patch('pathlib.Path.exists', return_value=True), \\
             patch('builtins.open', mock_open()), \\
             patch('yaml.safe_load', side_effect=[sample_config, large_rooms_config]):
            
            mock_getenv.side_effect = lambda key, default="": {
                "JWT_ENABLED": "false",  # Disable JWT for this test
            }.get(key, default)'''
        )
    ]
    
    for pattern, replacement in specific_fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Add @patch('os.getenv') decorators where needed for system config tests
    system_config_patterns = [
        r'(def test_system_config_get_[^(]+\([^)]*\):)',
        r'(def test_room_config_get_all_entity_ids_list_structure\([^)]*\):)',
    ]
    
    for pattern in system_config_patterns:
        matches = re.finditer(pattern, content)
        for match in reversed(list(matches)):  # Reverse to avoid index shifting
            method_def = match.group(1)
            # Check if it already has @patch('os.getenv')
            before_method = content[:match.start()]
            if "@patch('os.getenv')" not in before_method[-200:]:
                # Find the class method start
                lines_before = before_method.split('\n')
                method_line_idx = len(lines_before) - 1
                
                # Insert the decorator
                new_content_lines = content.split('\n')
                new_content_lines.insert(method_line_idx, "    @patch('os.getenv')")
                content = '\n'.join(new_content_lines)
                
                # Also need to add mock_getenv parameter and setup
                content = content.replace(
                    method_def,
                    method_def.replace('(self', '(self, mock_getenv')
                )
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed API configuration tests")

if __name__ == "__main__":
    fix_api_tests()