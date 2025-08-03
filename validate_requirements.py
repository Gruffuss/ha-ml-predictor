#!/usr/bin/env python3
"""
Validate that all required dependencies are installed and compatible.
Run this after installing requirements.txt to verify everything works.
"""

import sys
import importlib
import traceback

def test_import(module_name, package_name=None):
    """Test importing a module and return status."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name or module_name}: {e}")
        return False

def main():
    """Test all critical dependencies."""
    print("🔍 Validating requirements.txt dependencies...\n")
    
    # Core async dependencies
    print("📡 Core async dependencies:")
    results = []
    results.append(test_import("asyncio_mqtt", "asyncio-mqtt"))
    results.append(test_import("aiofiles"))
    results.append(test_import("websockets"))
    
    # Home Assistant integration
    print("\n🏠 Home Assistant integration:")
    results.append(test_import("homeassistant_api", "homeassistant-api"))
    results.append(test_import("aiohttp"))
    results.append(test_import("websocket", "websocket-client"))
    
    # Database
    print("\n🗄️  Database:")
    results.append(test_import("psycopg2", "psycopg2-binary"))
    results.append(test_import("asyncpg"))
    results.append(test_import("sqlalchemy"))
    results.append(test_import("alembic"))
    
    # Machine Learning
    print("\n🤖 Machine Learning:")
    results.append(test_import("sklearn", "scikit-learn"))
    results.append(test_import("xgboost"))
    results.append(test_import("numpy"))
    results.append(test_import("pandas"))
    results.append(test_import("scipy"))
    
    # Feature Engineering
    print("\n⚙️  Feature Engineering:")
    results.append(test_import("tsfresh"))
    results.append(test_import("pytz"))
    
    # API and Integration
    print("\n🌐 API and Integration:")
    results.append(test_import("fastapi"))
    results.append(test_import("uvicorn"))
    results.append(test_import("paho.mqtt", "paho-mqtt"))
    results.append(test_import("pydantic"))
    
    # Configuration and Logging
    print("\n📋 Configuration and Logging:")
    results.append(test_import("yaml", "pyyaml"))
    results.append(test_import("dotenv", "python-dotenv"))
    results.append(test_import("structlog"))
    
    # Development and Testing
    print("\n🧪 Development and Testing:")
    results.append(test_import("pytest"))
    results.append(test_import("pytest_asyncio", "pytest-asyncio"))
    results.append(test_import("pytest_mock", "pytest-mock"))
    results.append(test_import("pytest_cov", "pytest-cov"))
    results.append(test_import("black"))
    results.append(test_import("flake8"))
    results.append(test_import("mypy"))
    
    # Utilities
    print("\n🔧 Utilities:")
    results.append(test_import("click"))
    results.append(test_import("dateutil", "python-dateutil"))
    
    # Test critical compatibility
    print("\n🔬 Testing critical compatibility:")
    try:
        import pytest
        import pytest_asyncio
        print(f"✅ pytest ({pytest.__version__}) + pytest-asyncio ({pytest_asyncio.__version__}) compatibility: OK")
        results.append(True)
    except Exception as e:
        print(f"❌ pytest + pytest-asyncio compatibility: {e}")
        results.append(False)
    
    # Summary
    total = len(results)
    passed = sum(results)
    print(f"\n📊 Summary: {passed}/{total} dependencies validated")
    
    if passed == total:
        print("🎉 All dependencies are working correctly!")
        return 0
    else:
        print(f"⚠️  {total - passed} dependencies have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())