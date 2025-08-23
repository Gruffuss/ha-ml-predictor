"""Integration tests for database operations and data persistence.

Covers database integration with system components, data consistency,
and transaction management across the application.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import sqlite3
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text, select, func
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import tempfile
import os

from src.data.storage.database import DatabaseManager
from src.data.storage.models import SensorEvent, RoomState, Base
from src.core.config import DatabaseConfig
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError
)


class TestDatabaseSystemIntegration:
    """Test database integration with system components."""
    
    @pytest.fixture
    def test_db_manager(self):
        """Create a mock database manager for integration testing."""
        mock_manager = MagicMock()
        mock_manager.get_session = MagicMock()
        
        # Mock session context manager
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.get = AsyncMock()
        
        # Mock the async context manager
        async def mock_session_context():
            return mock_session
        
        mock_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock health check
        mock_manager.health_check = AsyncMock(return_value={
            "status": "healthy",
            "connection_pool": {"active": 1, "size": 10},
            "last_check": datetime.now(timezone.utc)
        })
        
        return mock_manager

    @pytest.fixture
    def sample_sensor_events(self):
        """Create sample sensor events for testing."""
        now = datetime.now(timezone.utc)
        return [
            SensorEvent(
                room_id="living_room",
                sensor_id="binary_sensor.living_room_motion",
                sensor_type="motion",
                state="on",
                previous_state="off",
                timestamp=now - timedelta(minutes=30),
                attributes={"device_class": "motion"},
                is_human_triggered=True
            ),
            SensorEvent(
                room_id="living_room",
                sensor_id="binary_sensor.living_room_motion",
                sensor_type="motion",
                state="off",
                previous_state="on",
                timestamp=now - timedelta(minutes=15),
                attributes={"device_class": "motion"},
                is_human_triggered=True
            ),
            SensorEvent(
                room_id="kitchen",
                sensor_id="binary_sensor.kitchen_motion",
                sensor_type="motion",
                state="on",
                previous_state="off",
                timestamp=now - timedelta(minutes=10),
                attributes={"device_class": "motion"},
                is_human_triggered=True
            )
        ]

    @pytest.mark.asyncio
    async def test_orm_integration(self, test_db_manager, sample_sensor_events):
        """Test SQLAlchemy ORM integration with database operations."""
        db_manager = await test_db_manager.__anext__()
        async with db_manager.get_session() as session:
            # Test bulk creation
            session.add_all(sample_sensor_events)
            await session.commit()
            
            # Test querying
            result = await session.execute(
                select(SensorEvent).where(SensorEvent.room_id == "living_room")
            )
            events = result.scalars().all()
            
            assert len(events) == 2
            assert all(event.room_id == "living_room" for event in events)

    @pytest.mark.asyncio
    async def test_transaction_management(self, test_db_manager, sample_sensor_events):
        """Test transaction management and rollback functionality."""
        # Test successful transaction
        async with test_db_manager.get_session() as session:
            session.add(sample_sensor_events[0])
            await session.commit()
            
            # Verify event was saved
            result = await session.execute(
                select(func.count()).select_from(SensorEvent)
            )
            count = result.scalar()
            assert count == 1
        
        # Test transaction rollback
        try:
            async with test_db_manager.get_session() as session:
                session.add(sample_sensor_events[1])
                # Simulate error before commit
                raise Exception("Test error")
        except Exception:
            pass  # Expected
        
        # Verify rollback worked - should still only have 1 event
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(func.count()).select_from(SensorEvent)
            )
            count = result.scalar()
            assert count == 1

    @pytest.mark.asyncio
    async def test_connection_pooling(self, test_db_manager):
        """Test database connection pooling functionality."""
        # Test multiple concurrent sessions
        async def create_session_task():
            async with test_db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar()
        
        # Run multiple concurrent tasks
        tasks = [create_session_task() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result == 1 for result in results)

    @pytest.mark.asyncio
    async def test_health_check_integration(self, test_db_manager):
        """Test database health check integration."""
        health_status = await test_db_manager.health_check()
        
        assert health_status["status"] == "healthy"
        assert "connection_pool" in health_status
        assert "last_check" in health_status


class TestDataPersistenceIntegration:
    """Test data persistence integration across system components."""
    
    @pytest_asyncio.fixture
    async def test_db_manager(self):
        """Create a test database manager."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        config = DatabaseConfig(
            connection_string=f"sqlite+aiosqlite:///{temp_db.name}"
        )
        
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        
        async with db_manager.get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield db_manager
        
        await db_manager.close()
        os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_event_persistence_lifecycle(self, test_db_manager):
        """Test complete event persistence lifecycle."""
        now = datetime.now(timezone.utc)
        
        # Create event
        event = SensorEvent(
            room_id="test_room",
            sensor_id="test_sensor",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=now,
            attributes={"test": "data"},
            is_human_triggered=True
        )
        
        # Save event
        async with test_db_manager.get_session() as session:
            session.add(event)
            await session.commit()
            event_id = event.id
        
        # Retrieve event
        async with test_db_manager.get_session() as session:
            retrieved_event = await session.get(SensorEvent, event_id)
            
            assert retrieved_event is not None
            assert retrieved_event.room_id == "test_room"
            assert retrieved_event.sensor_id == "test_sensor"
            assert retrieved_event.attributes["test"] == "data"
        
        # Update event
        async with test_db_manager.get_session() as session:
            event_to_update = await session.get(SensorEvent, event_id)
            event_to_update.state = "off"
            await session.commit()
        
        # Verify update
        async with test_db_manager.get_session() as session:
            updated_event = await session.get(SensorEvent, event_id)
            assert updated_event.state == "off"

    @pytest.mark.asyncio
    async def test_room_state_persistence(self, test_db_manager):
        """Test room state persistence integration."""
        now = datetime.now(timezone.utc)
        
        room_state = RoomState(
            room_id="living_room",
            current_state="occupied",
            confidence=0.95,
            last_changed=now,
            last_sensor_trigger=now - timedelta(minutes=5),
            attributes={"prediction_source": "lstm_model"}
        )
        
        # Save room state
        async with test_db_manager.get_session() as session:
            session.add(room_state)
            await session.commit()
        
        # Query room state
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(RoomState).where(RoomState.room_id == "living_room")
            )
            retrieved_state = result.scalar_one()
            
            assert retrieved_state.current_state == "occupied"
            assert retrieved_state.confidence == 0.95

    @pytest.mark.asyncio
    async def test_bulk_data_persistence(self, test_db_manager):
        """Test bulk data operations and persistence."""
        # Create multiple events
        events = []
        now = datetime.now(timezone.utc)
        
        for i in range(100):
            event = SensorEvent(
                room_id=f"room_{i % 5}",  # 5 different rooms
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=now - timedelta(minutes=i),
                attributes={"batch_id": 1},
                is_human_triggered=True
            )
            events.append(event)
        
        # Bulk insert
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
        
        # Verify all events were saved
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(func.count()).select_from(SensorEvent)
            )
            count = result.scalar()
            assert count == 100
            
            # Test querying by room
            result = await session.execute(
                select(SensorEvent).where(SensorEvent.room_id == "room_0")
            )
            room_events = result.scalars().all()
            assert len(room_events) == 20  # 100 events / 5 rooms


class TestTimescaleDBIntegration:
    """Test TimescaleDB-specific integration (simulated with SQLite)."""
    
    @pytest_asyncio.fixture
    async def test_db_manager(self):
        """Create test database manager."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        config = DatabaseConfig(
            connection_string=f"sqlite+aiosqlite:///{temp_db.name}"
        )
        
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        
        async with db_manager.get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield db_manager
        
        await db_manager.close()
        os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_time_series_operations(self, test_db_manager):
        """Test time-series style operations (simulated for SQLite)."""
        now = datetime.now(timezone.utc)
        
        # Create time series data
        events = []
        for i in range(24):  # 24 hours of hourly data
            event = SensorEvent(
                room_id="living_room",
                sensor_id="motion_sensor",
                sensor_type="motion",
                state="on" if i % 3 == 0 else "off",
                previous_state="off" if i % 3 == 0 else "on",
                timestamp=now - timedelta(hours=i),
                attributes={"hour": i},
                is_human_triggered=True
            )
            events.append(event)
        
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
            
            # Test time-based queries
            six_hours_ago = now - timedelta(hours=6)
            result = await session.execute(
                select(SensorEvent)
                .where(SensorEvent.timestamp >= six_hours_ago)
                .order_by(SensorEvent.timestamp.desc())
            )
            recent_events = result.scalars().all()
            
            assert len(recent_events) == 6
            assert all(event.timestamp >= six_hours_ago for event in recent_events)

    @pytest.mark.asyncio
    async def test_time_series_aggregation(self, test_db_manager):
        """Test time series aggregation queries."""
        now = datetime.now(timezone.utc)
        
        # Create events with varying states
        events = []
        for i in range(20):
            event = SensorEvent(
                room_id="test_room",
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on" if i < 10 else "off",
                timestamp=now - timedelta(minutes=i * 10),
                is_human_triggered=True
            )
            events.append(event)
        
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
            
            # Test aggregation by state
            result = await session.execute(
                select(SensorEvent.state, func.count())
                .group_by(SensorEvent.state)
            )
            state_counts = dict(result.all())
            
            assert state_counts["on"] == 10
            assert state_counts["off"] == 10

    @pytest.mark.asyncio
    async def test_data_retention_simulation(self, test_db_manager):
        """Test data retention policies (simulated)."""
        now = datetime.now(timezone.utc)
        
        # Create old and new data
        old_events = []
        new_events = []
        
        # Old data (older than 30 days)
        for i in range(5):
            event = SensorEvent(
                room_id="test_room",
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                timestamp=now - timedelta(days=35 + i),
                is_human_triggered=True
            )
            old_events.append(event)
        
        # New data (within 30 days)
        for i in range(5):
            event = SensorEvent(
                room_id="test_room",
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                timestamp=now - timedelta(days=i),
                is_human_triggered=True
            )
            new_events.append(event)
        
        async with test_db_manager.get_session() as session:
            session.add_all(old_events + new_events)
            await session.commit()
            
            # Simulate retention policy (delete data older than 30 days)
            cutoff_date = now - timedelta(days=30)
            
            # Count data before cleanup
            total_count_result = await session.execute(
                select(func.count()).select_from(SensorEvent)
            )
            total_count = total_count_result.scalar()
            assert total_count == 10
            
            # Count new data (would be retained)
            new_count_result = await session.execute(
                select(func.count()).select_from(SensorEvent)
                .where(SensorEvent.timestamp >= cutoff_date)
            )
            new_count = new_count_result.scalar()
            assert new_count == 5  # Only new data


class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    @pytest_asyncio.fixture
    async def test_db_manager(self):
        """Create test database manager."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        config = DatabaseConfig(
            connection_string=f"sqlite+aiosqlite:///{temp_db.name}"
        )
        
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        
        async with db_manager.get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield db_manager
        
        await db_manager.close()
        os.unlink(temp_db.name)

    @pytest.mark.asyncio
    async def test_query_performance(self, test_db_manager):
        """Test query performance with moderate data set."""
        import time
        
        # Create test data
        now = datetime.now(timezone.utc)
        events = []
        
        for i in range(1000):  # 1000 events
            event = SensorEvent(
                room_id=f"room_{i % 10}",
                sensor_id=f"sensor_{i % 50}",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                timestamp=now - timedelta(minutes=i),
                is_human_triggered=True
            )
            events.append(event)
        
        # Bulk insert
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
        
        # Test query performance
        start_time = time.time()
        
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(SensorEvent)
                .where(SensorEvent.room_id == "room_1")
                .order_by(SensorEvent.timestamp.desc())
                .limit(50)
            )
            events = result.scalars().all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert len(events) == 50
        assert query_time < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, test_db_manager):
        """Test bulk operations performance."""
        import time
        
        # Test bulk insert performance
        events = []
        now = datetime.now(timezone.utc)
        
        for i in range(500):
            event = SensorEvent(
                room_id=f"room_{i % 5}",
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                timestamp=now - timedelta(seconds=i),
                is_human_triggered=True
            )
            events.append(event)
        
        start_time = time.time()
        
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Verify all events were inserted
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(func.count()).select_from(SensorEvent)
            )
            count = result.scalar()
            assert count == 500
        
        # Performance assertion - should handle 500 inserts quickly
        assert insert_time < 5.0  # Within 5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, test_db_manager):
        """Test concurrent database access performance."""
        async def concurrent_query_task(room_id: str):
            async with test_db_manager.get_session() as session:
                result = await session.execute(
                    select(func.count()).select_from(SensorEvent)
                    .where(SensorEvent.room_id == room_id)
                )
                return result.scalar()
        
        # First create some test data
        events = []
        now = datetime.now(timezone.utc)
        
        for i in range(100):
            event = SensorEvent(
                room_id=f"room_{i % 10}",
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                timestamp=now - timedelta(minutes=i),
                is_human_triggered=True
            )
            events.append(event)
        
        async with test_db_manager.get_session() as session:
            session.add_all(events)
            await session.commit()
        
        # Test concurrent access
        import time
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = concurrent_query_task(f"room_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # All queries should succeed
        assert len(results) == 10
        assert all(isinstance(count, int) for count in results)
        
        # Should complete within reasonable time
        assert concurrent_time < 2.0  # Within 2 seconds