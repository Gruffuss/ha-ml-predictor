#!/usr/bin/env python3
"""
Production monitoring system startup script.

This script initializes and starts the comprehensive monitoring system
for the Home Assistant ML Predictor, including metrics collection,
alerting, and health monitoring.
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys

import signal

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.alerts import NotificationConfig, get_alert_manager
from src.utils.logger import get_logger_manager
from src.utils.metrics import get_metrics_manager
from src.utils.monitoring_integration import get_monitoring_integration


class MonitoringSystemStarter:
    """Manages startup and shutdown of the monitoring system."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path
        self.logger = None
        self.monitoring_integration = None
        self.metrics_manager = None
        self.alert_manager = None
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize monitoring components."""
        print("🔍 Initializing Home Assistant ML Predictor Monitoring System...")

        # Setup logging first
        logger_manager = get_logger_manager()
        self.logger = logger_manager.get_logger("monitoring_startup")

        self.logger.info("Starting monitoring system initialization")

        # Initialize monitoring components
        self.monitoring_integration = get_monitoring_integration()
        self.metrics_manager = get_metrics_manager()
        self.alert_manager = get_alert_manager()

        self.logger.info("Monitoring components initialized")
        print("✅ Monitoring components initialized")

    async def start_monitoring(self):
        """Start all monitoring services."""
        print("🚀 Starting monitoring services...")

        try:
            # Start metrics collection
            self.logger.info("Starting metrics collection")
            self.metrics_manager.start_background_collection(update_interval=30)
            print("✅ Metrics collection started")

            # Start system monitoring
            self.logger.info("Starting system monitoring")
            await self.monitoring_integration.start_monitoring(
                health_check_interval=300,  # 5 minutes
                performance_summary_interval=900,  # 15 minutes
            )
            print("✅ System monitoring started")

            # Log successful startup
            await self.alert_manager.trigger_alert(
                rule_name="monitoring_system_started",
                title="Monitoring System Online",
                message="Home Assistant ML Predictor monitoring system started successfully",
                component="monitoring_system",
                context={
                    "startup_time": asyncio.get_event_loop().time(),
                    "metrics_enabled": True,
                    "health_checks_enabled": True,
                },
            )

            self.logger.info("Monitoring system startup complete")
            print("🎉 Monitoring system started successfully!")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            print(f"❌ Failed to start monitoring system: {e}")
            raise

    async def run(self):
        """Main monitoring loop."""
        print("\n📊 Monitoring System Status:")
        print("  • Metrics endpoint: http://localhost:8000/metrics")
        print("  • Health checks: Every 5 minutes")
        print("  • Performance summaries: Every 15 minutes")
        print("  • Log files: logs/ directory")
        print("\nPress Ctrl+C to stop monitoring system\n")

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except KeyboardInterrupt:
            print("\n⚠️  Received shutdown signal...")
            self.logger.info("Received shutdown signal")

    async def shutdown(self):
        """Shutdown monitoring services."""
        print("🛑 Shutting down monitoring system...")
        self.logger.info("Starting monitoring system shutdown")

        try:
            # Stop monitoring services
            await self.monitoring_integration.stop_monitoring()
            print("✅ System monitoring stopped")

            # Stop metrics collection
            self.metrics_manager.stop_background_collection()
            print("✅ Metrics collection stopped")

            # Log shutdown
            await self.alert_manager.trigger_alert(
                rule_name="monitoring_system_stopped",
                title="Monitoring System Offline",
                message="Home Assistant ML Predictor monitoring system stopped",
                component="monitoring_system",
                context={"shutdown_time": asyncio.get_event_loop().time()},
            )

            self.logger.info("Monitoring system shutdown complete")
            print("✅ Monitoring system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            print(f"⚠️  Error during shutdown: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start Home Assistant ML Predictor monitoring system"
    )
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=300,
        help="Health check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=30,
        help="Metrics collection interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Create monitoring starter
    starter = MonitoringSystemStarter(config_path=args.config)

    try:
        # Initialize
        await starter.initialize()

        # Setup signal handlers
        starter.setup_signal_handlers()

        # Start monitoring
        await starter.start_monitoring()

        # Run monitoring loop
        await starter.run()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Monitoring system failed: {e}")
        if starter.logger:
            starter.logger.error(f"Monitoring system failed: {e}", exc_info=True)
        return 1

    finally:
        # Shutdown
        try:
            await starter.shutdown()
        except Exception as e:
            print(f"⚠️  Shutdown error: {e}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
