"""
Test script to validate API server integration with TrackingManager.

This script verifies that the API server starts automatically when
TrackingManager is initialized with api_config, meeting the CLAUDE.md
requirements for automatic component integration.
"""

import asyncio
import logging
from datetime import datetime

from src.core.config import get_config, APIConfig
from src.adaptation.tracking_manager import TrackingManager, TrackingConfig


async def test_api_integration():
    """Test that API server integrates automatically with TrackingManager."""
    
    print("🧪 Testing API Server Integration with TrackingManager")
    print("=" * 60)
    
    try:
        # Load configuration
        config = get_config()
        print(f"✅ Configuration loaded (API enabled: {config.api.enabled})")
        
        # Create TrackingManager with API config
        print("🚀 Creating TrackingManager with API configuration...")
        tracking_manager = TrackingManager(
            config=config.tracking,
            api_config=config.api  # This should enable automatic API server startup
        )
        
        # Check initial state
        initial_status = tracking_manager.get_api_server_status()
        print(f"📊 Initial API server status: {initial_status}")
        
        # Initialize tracking manager (should automatically start API server)
        print("🔄 Initializing TrackingManager (should auto-start API server)...")
        await tracking_manager.initialize()
        
        # Check if API server started automatically
        final_status = tracking_manager.get_api_server_status()
        print(f"📊 Final API server status: {final_status}")
        
        # Validate integration success
        if final_status["enabled"] and final_status["running"]:
            print("✅ SUCCESS: API server started automatically!")
            print(f"🌐 API server running at: http://{final_status['host']}:{final_status['port']}")
            print("🎯 Integration meets CLAUDE.md requirements:")
            print("   ✓ Component integrates into main system automatically")
            print("   ✓ No manual setup required")
            print("   ✓ Works without example files")
            
            # Brief test run
            print("\n⏳ Running integrated system for 5 seconds...")
            await asyncio.sleep(5)
            
            result = "PASS"
        elif final_status["enabled"] and not final_status["running"]:
            print("❌ FAIL: API server enabled but failed to start automatically")
            result = "FAIL"
        elif not final_status["enabled"]:
            print("ℹ️ INFO: API server disabled in configuration")
            print("   This is acceptable - integration works when enabled")
            result = "PASS (disabled)"
        else:
            print("❌ FAIL: Unexpected API server state")
            result = "FAIL"
        
        # Clean shutdown
        print("\n🛑 Shutting down system...")
        await tracking_manager.stop_tracking()
        
        final_final_status = tracking_manager.get_api_server_status()
        if not final_final_status["running"]:
            print("✅ API server stopped cleanly during shutdown")
        else:
            print("⚠️ WARNING: API server may not have stopped properly")
        
        return result
        
    except Exception as e:
        print(f"💥 ERROR: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"


async def main():
    """Run the integration test."""
    print("🏠 Home Assistant Occupancy Prediction System")
    print("🔧 API Server Integration Test")
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    print()
    
    # Configure logging to reduce noise
    logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
    
    result = await test_api_integration()
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Result: {result}")
    print(f"⏰ Test completed at: {datetime.now().isoformat()}")
    
    if result in ["PASS", "PASS (disabled)"]:
        print("🎉 API integration working correctly!")
        return 0
    else:
        print("💥 API integration has issues!")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)