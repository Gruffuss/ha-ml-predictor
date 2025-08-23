#!/usr/bin/env python3
"""
Example script demonstrating database usage for the occupancy prediction system.

This script shows how to:
1. Initialize the database manager
2. Insert sensor events
3. Query recent data
4. Perform health checks
5. Clean up connections
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.storage import (
    RoomState,
    SensorEvent,
    close_database_manager,
    get_database_manager,
    get_db_session,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_sensor_data_operations():
    """Demonstrate basic sensor data operations."""
    logger.info("=== Sensor Data Operations Example ===")

    try:
        # Get database session
        async with get_db_session() as session:

            # Insert sample sensor event
            sensor_event = SensorEvent(
                timestamp=datetime.utcnow(),
                room_id="living_room",
                sensor_id="motion_sensor_1",
                sensor_type="motion",
                state="on",
                previous_state="off",
                is_human_triggered=True,
                confidence_score=0.95,
                attributes={"brightness": 300, "temperature": 22.5},
            )

            session.add(sensor_event)
            await session.flush()  # Get the ID

            logger.info(f"Inserted sensor event with ID: {sensor_event.id}")

            # Query recent events
            recent_events = await SensorEvent.get_recent_events(
                session=session,
                room_id="living_room",
                hours=24,
                sensor_types=["motion"],
            )

            logger.info(
                f"Found {len(recent_events)} recent motion events in living room"
            )

            # Query state changes
            state_changes = await SensorEvent.get_state_changes(
                session=session,
                room_id="living_room",
                start_time=datetime.utcnow() - timedelta(hours=1),
            )

            logger.info(f"Found {len(state_changes)} state changes in last hour")

    except Exception as e:
        logger.error(f"Sensor data operations failed: {e}")
        raise


async def example_room_state_operations():
    """Demonstrate room state tracking."""
    logger.info("=== Room State Operations Example ===")

    try:
        async with get_db_session() as session:

            # Create room state entry
            room_state = RoomState(
                room_id="living_room",
                timestamp=datetime.utcnow(),
                is_occupied=True,
                occupancy_confidence=0.89,
                occupant_type="human",
                occupant_count=1,
                state_duration=300,  # 5 minutes
                transition_trigger="motion_sensor_1",
                certainty_factors={
                    "motion_detected": 0.9,
                    "door_activity": 0.1,
                    "time_of_day": 0.7,
                },
            )

            session.add(room_state)
            await session.flush()

            logger.info(f"Created room state entry with ID: {room_state.id}")

            # Get current state
            current_state = await RoomState.get_current_state(
                session=session, room_id="living_room"
            )

            if current_state:
                logger.info(
                    f"Current room state: occupied={current_state.is_occupied}, "
                    f"confidence={current_state.occupancy_confidence}"
                )

            # Get occupancy history
            history = await RoomState.get_occupancy_history(
                session=session, room_id="living_room", hours=24
            )

            logger.info(f"Found {len(history)} occupancy history entries")

    except Exception as e:
        logger.error(f"Room state operations failed: {e}")
        raise


async def example_database_health_check():
    """Demonstrate database health monitoring."""
    logger.info("=== Database Health Check Example ===")

    try:
        # Get database manager
        db_manager = await get_database_manager()

        # Perform health check
        health_status = await db_manager.health_check()

        logger.info(f"Database status: {health_status['status']}")
        logger.info(
            f"Response time: {health_status['performance_metrics'].get('response_time_ms', 'N/A')}ms"
        )
        logger.info(f"TimescaleDB: {health_status.get('timescale_status', 'unknown')}")

        # Get connection statistics
        conn_stats = db_manager.get_connection_stats()
        logger.info(f"Total connections: {conn_stats['total_connections']}")
        logger.info(f"Failed connections: {conn_stats['failed_connections']}")

        if health_status["errors"]:
            logger.warning(f"Health check errors: {health_status['errors']}")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise


async def example_bulk_operations():
    """Demonstrate bulk data operations for performance."""
    logger.info("=== Bulk Operations Example ===")

    try:
        async with get_db_session() as session:

            # Prepare bulk sensor events
            events = []
            base_time = datetime.utcnow() - timedelta(minutes=30)

            for i in range(10):
                event = SensorEvent(
                    timestamp=base_time + timedelta(minutes=i * 3),
                    room_id="kitchen",
                    sensor_id=f"sensor_{i % 3}",
                    sensor_type="motion" if i % 2 == 0 else "door",
                    state="on" if i % 2 == 0 else "open",
                    previous_state="off" if i % 2 == 0 else "closed",
                    is_human_triggered=True,
                    confidence_score=0.8 + (i * 0.02),
                )
                events.append(event)

            # Bulk insert
            session.add_all(events)
            await session.flush()

            logger.info(f"Bulk inserted {len(events)} sensor events")

            # Verify insertion
            recent_kitchen_events = await SensorEvent.get_recent_events(
                session=session, room_id="kitchen", hours=1
            )

            logger.info(f"Verified: {len(recent_kitchen_events)} events in kitchen")

    except Exception as e:
        logger.error(f"Bulk operations failed: {e}")
        raise


async def example_advanced_queries():
    """Demonstrate advanced query patterns."""
    logger.info("=== Advanced Queries Example ===")

    try:
        async with get_db_session() as session:

            # Get transition sequences for pattern analysis
            sequences = await SensorEvent.get_transition_sequences(
                session=session,
                room_id="living_room",
                lookback_hours=24,
                min_sequence_length=2,
            )

            logger.info(f"Found {len(sequences)} sensor transition sequences")

            for i, sequence in enumerate(sequences[:3]):  # Show first 3
                logger.info(
                    f"Sequence {i+1}: {len(sequence)} events, "
                    f"span: {sequence[-1].timestamp - sequence[0].timestamp}"
                )

            # Raw SQL query example
            db_manager = await get_database_manager()

            # Get sensor activity summary
            result = await db_manager.execute_query(
                """
                SELECT
                    room_id,
                    sensor_type,
                    COUNT(*) as event_count,
                    COUNT(DISTINCT sensor_id) as sensor_count,
                    MAX(timestamp) as last_activity
                FROM sensor_events
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY room_id, sensor_type
                ORDER BY event_count DESC
                """,
                fetch_all=True,
            )

            if result:
                logger.info("Sensor activity summary (last 24 hours):")
                for row in result:
                    logger.info(
                        f"  {row[0]} - {row[1]}: {row[2]} events, "
                        f"{row[3]} sensors, last: {row[4]}"
                    )

    except Exception as e:
        logger.error(f"Advanced queries failed: {e}")
        raise


async def main():
    """Main example function."""
    logger.info("Starting database usage examples...")

    try:
        # Run all examples
        await example_sensor_data_operations()
        await example_room_state_operations()
        await example_database_health_check()
        await example_bulk_operations()
        await example_advanced_queries()

        logger.info("All database examples completed successfully!")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        return 1
    finally:
        # Always clean up connections
        await close_database_manager()
        logger.info("Database connections closed")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
