# Flake8 Linting Report

**Generated:** August 2025  
**Configuration:** max-line-length=140, extend-ignore=E203,W503,E501  
**Total Errors:** 317

## Error Distribution by Directory

| Directory | Errors |
|-----------|---------|
| src/ | 170 |
| tests/ | 33 |
| scripts/ | 76 |
| examples/ | 38 |

## Error Types

| Error Code | Count | Description |
|------------|--------|-------------|
| F401 | 126 | Module imported but unused |
| E402 | 47 | Module level import not at top of file |
| E712 | 42 | Comparison to True/False should use 'is' |
| F541 | 31 | f-string is missing placeholders |
| W291 | 12 | Trailing whitespace |
| W293 | 10 | Blank line contains whitespace |
| F841 | 10 | Local variable assigned but never used |
| E999 | 9 | SyntaxError or IndentationError |
| F811 | 7 | Redefined while unused |
| E722 | 7 | Do not use bare 'except' |
| F402 | 1 | Import shadowed by loop variable |

## Configuration (.flake8)

```ini
[flake8]
max-line-length = 140
extend-ignore = E203,W503,E501
exclude = .git,__pycache__,venv,.venv,build,dist,*.egg-info,.pytest_cache,logs,performance_reports,*.md,*.txt,*.json,*.yaml,*.yml,*.dockerfile,Dockerfile*,docker-compose*
per-file-ignores = tests/*:F401,F811,F841,examples/*:F401,F841,scripts/*:F401,F841
```

## Priority Issues

**High Priority (9 errors):**
- E999: SyntaxError or IndentationError

**Medium Priority (225 errors):**
- F401: Unused imports (126)
- E402: Import placement (47) 
- E712: Boolean comparisons (42)
- F841: Unused variables (10)

**Low Priority (83 errors):**
- F541: Empty f-strings (31)
- W291: Trailing whitespace (12)
- W293: Blank line whitespace (10)
- F811: Redefined names (7)
- E722: Bare except (7)
- F402: Import shadowing (1)