#!/usr/bin/env python3
"""
Manual test to verify the fixes work without pytest.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_database_manager_initialization():
    """Test that DatabaseManager can be initialized properly."""
    print("Testing DatabaseManager initialization...")

    try:
        from src.data.storage.database import DatabaseManager
        from src.core.config import DatabaseConfig

        # Create test config
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://test:test@localhost:5432/test",
            pool_size=1,
            max_overflow=0,
        )

        # Create manager
        manager = DatabaseManager(config)

        # Check that it's not initialized yet
        assert not manager.is_initialized
        print("✓ Manager created successfully")

        # Check that get_session raises proper error before initialization
        try:
            async with manager.get_session() as session:
                pass
            print("✗ Should have raised RuntimeError")
            return False
        except RuntimeError as e:
            if "Database manager not initialized" in str(e):
                print("✓ Proper error raised before initialization")
            else:
                print(f"✗ Wrong error: {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_model_imports():
    """Test that all models can be imported."""
    print("Testing model imports...")

    try:
        from src.data.storage.models import SensorEvent, RoomState, Prediction

        # Create test instances
        event = SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.utcnow(),
        )

        room_state = RoomState(
            room_id="test_room",
            timestamp=datetime.utcnow(),
            is_occupied=True,
            occupancy_confidence=0.9,
        )

        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.utcnow(),
            predicted_transition_time=datetime.utcnow() + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )

        print("✓ All models imported and instantiated successfully")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_config_system():
    """Test config system works."""
    print("Testing config system...")

    try:
        from src.core.config import SystemConfig, DatabaseConfig, HomeAssistantConfig

        # Test basic config creation
        ha_config = HomeAssistantConfig(url="http://test", token="test")
        db_config = DatabaseConfig(connection_string="postgresql://test")

        print("✓ Config classes work")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    """Run all manual tests."""
    print("Running manual verification tests...")
    print("=" * 50)

    tests = [
        test_config_system,
        test_model_imports,
        test_database_manager_initialization,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()

    print("=" * 50)
    print("SUMMARY:")

    all_passed = all(results)
    if all_passed:
        print("✅ All manual tests PASSED!")
        print("The fixes appear to be working correctly.")
    else:
        print("❌ Some manual tests FAILED!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    asyncio.run(main())
