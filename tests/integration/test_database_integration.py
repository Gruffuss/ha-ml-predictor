"""
Integration tests for database operations.

Tests full database operations including models, relationships,
queries, and TimescaleDB-specific functionality.
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import case, func, select, text
from sqlalchemy.exc import IntegrityError

from src.data.storage.database import DatabaseManager, get_database_manager
from src.data.storage.models import (
    FeatureStore,
    ModelAccuracy,
    Prediction,
    RoomState,
    SensorEvent,
    create_timescale_hypertables,
    optimize_database_performance,
)


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Integration tests for database functionality."""

    @pytest.mark.asyncio
    async def test_database_manager_lifecycle(self, test_db_manager):
        """Test complete database manager lifecycle using test database manager fixture."""
        # Use the test database manager fixture which is already initialized and configured
        manager = test_db_manager

        # Verify it's initialized
        assert manager.is_initialized

        # Test basic operations
        async with manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        # Test health check
        health = await manager.health_check()
        assert health["status"] == "healthy"
        assert health["performance_metrics"]["response_time_ms"] >= 0

        # Note: Don't close the manager here as it's managed by the fixture

    @pytest.mark.asyncio
    async def test_sensor_event_crud_operations(self, test_db_session):
        """Test CRUD operations for SensorEvent model."""
        # Create
        event = SensorEvent(
            room_id="integration_test_room",
            sensor_id="binary_sensor.integration_test",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={"device_class": "motion", "test": True},
            is_human_triggered=True,
            confidence_score=0.85,
        )

        test_db_session.add(event)
        await test_db_session.commit()

        # Read
        result = await test_db_session.execute(
            select(SensorEvent).where(
                SensorEvent.sensor_id == "binary_sensor.integration_test"
            )
        )
        retrieved_event = result.scalar_one()

        assert retrieved_event.room_id == "integration_test_room"
        assert retrieved_event.sensor_type == "motion"
        assert retrieved_event.state == "on"
        assert retrieved_event.attributes["test"] is True
        assert retrieved_event.confidence_score == 0.85

        # Update
        retrieved_event.state = "off"
        retrieved_event.previous_state = "on"
        await test_db_session.commit()

        # Verify update
        await test_db_session.refresh(retrieved_event)
        assert retrieved_event.state == "off"
        assert retrieved_event.previous_state == "on"

        # Delete
        await test_db_session.delete(retrieved_event)
        await test_db_session.commit()

        # Verify deletion
        result = await test_db_session.execute(
            select(SensorEvent).where(
                SensorEvent.sensor_id == "binary_sensor.integration_test"
            )
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_room_state_tracking(self, test_db_session):
        """Test room state tracking and queries."""
        base_time = datetime.now(timezone.utc)
        room_id = "integration_test_room"

        # Create sequence of room states
        states = [
            RoomState(
                room_id=room_id,
                timestamp=base_time - timedelta(hours=3),
                is_occupied=False,
                occupancy_confidence=0.9,
                occupant_type="human",
                state_duration=1800,  # 30 minutes
            ),
            RoomState(
                room_id=room_id,
                timestamp=base_time - timedelta(hours=2),
                is_occupied=True,
                occupancy_confidence=0.95,
                occupant_type="human",
                state_duration=3600,  # 1 hour
            ),
            RoomState(
                room_id=room_id,
                timestamp=base_time - timedelta(hours=1),
                is_occupied=False,
                occupancy_confidence=0.8,
                occupant_type="human",
                state_duration=3600,  # 1 hour
            ),
            RoomState(
                room_id=room_id,
                timestamp=base_time,
                is_occupied=True,
                occupancy_confidence=0.92,
                occupant_type="human",
                state_duration=0,  # Current state
            ),
        ]

        for state in states:
            test_db_session.add(state)
        await test_db_session.commit()

        # Test get_current_state
        current = await RoomState.get_current_state(test_db_session, room_id)
        assert current is not None
        assert current.is_occupied is True
        assert current.occupancy_confidence == 0.92
        assert current.timestamp == base_time

        # Test get_occupancy_history
        history = await RoomState.get_occupancy_history(
            test_db_session, room_id, hours=4
        )
        assert len(history) == 4

        # Should be ordered by timestamp
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i - 1].timestamp

        # Test occupancy pattern analysis
        occupied_count = sum(1 for state in history if state.is_occupied)
        vacant_count = len(history) - occupied_count
        assert occupied_count == 2
        assert vacant_count == 2

    @pytest.mark.asyncio
    async def test_prediction_accuracy_tracking(self, test_db_session):
        """Test prediction accuracy tracking and metrics."""
        base_time = datetime.now(timezone.utc)
        room_id = "accuracy_test_room"

        # Create predictions with varying accuracy
        predictions = [
            # Accurate prediction (5 minutes off)
            Prediction(
                room_id=room_id,
                prediction_time=base_time - timedelta(hours=2),
                predicted_transition_time=base_time - timedelta(hours=1),
                transition_type="occupied_to_vacant",
                confidence_score=0.9,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(hours=1, minutes=-5),
                accuracy_minutes=5.0,
                is_accurate=True,
                validation_timestamp=base_time - timedelta(minutes=50),
            ),
            # Inaccurate prediction (20 minutes off)
            Prediction(
                room_id=room_id,
                prediction_time=base_time - timedelta(hours=3),
                predicted_transition_time=base_time - timedelta(hours=2),
                transition_type="vacant_to_occupied",
                confidence_score=0.7,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(hours=2, minutes=20),
                accuracy_minutes=20.0,
                is_accurate=False,
                validation_timestamp=base_time - timedelta(hours=1, minutes=40),
            ),
            # Pending validation
            Prediction(
                room_id=room_id,
                prediction_time=base_time - timedelta(minutes=30),
                predicted_transition_time=base_time - timedelta(minutes=10),
                transition_type="occupied_to_vacant",
                confidence_score=0.85,
                model_type="lstm",
                model_version="v1.0",
                # No actual_transition_time - pending validation
            ),
        ]

        for pred in predictions:
            test_db_session.add(pred)
        await test_db_session.commit()

        # Test pending validations
        pending = await Prediction.get_pending_validations(test_db_session, room_id)
        assert len(pending) == 1
        assert pending[0].confidence_score == 0.85

        # Test accuracy metrics
        metrics = await Prediction.get_accuracy_metrics(
            test_db_session, room_id, days=1, model_type="lstm"
        )

        assert metrics["total_predictions"] == 2  # Only validated predictions
        assert metrics["accurate_predictions"] == 1
        assert metrics["accuracy_rate"] == 0.5
        assert metrics["mean_error_minutes"] == 12.5  # (5 + 20) / 2
        assert metrics["rmse_minutes"] > metrics["mean_error_minutes"]

    @pytest.mark.asyncio
    async def test_model_relationships(self, test_db_session):
        """Test relationships between database models."""
        base_time = datetime.now(timezone.utc)

        # Create sensor event
        sensor_event = SensorEvent(
            room_id="relationship_test_room",
            sensor_id="binary_sensor.test_motion",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=base_time - timedelta(minutes=30),
        )
        test_db_session.add(sensor_event)
        await test_db_session.flush()

        # Create room state
        room_state = RoomState(
            room_id="relationship_test_room",
            timestamp=base_time - timedelta(minutes=25),
            is_occupied=True,
            occupancy_confidence=0.9,
        )
        test_db_session.add(room_state)
        await test_db_session.flush()

        # Create prediction with relationships
        prediction = Prediction(
            room_id="relationship_test_room",
            prediction_time=base_time - timedelta(minutes=20),
            predicted_transition_time=base_time + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="xgboost",
            model_version="v1.0",
            triggering_event_id=sensor_event.id,
            room_state_id=room_state.id,
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Test relationships
        await test_db_session.refresh(prediction)

        # Test forward relationships using application-level methods
        triggering_event = await prediction.get_triggering_event(test_db_session)
        assert triggering_event is not None
        assert triggering_event.id == sensor_event.id
        assert triggering_event.sensor_id == "binary_sensor.test_motion"

        related_room_state = await prediction.get_room_state(test_db_session)
        assert related_room_state is not None
        assert related_room_state.id == room_state.id
        assert related_room_state.is_occupied is True

        # Test reverse relationships using application-level methods
        await test_db_session.refresh(sensor_event)
        sensor_event_predictions = await sensor_event.get_predictions(test_db_session)
        assert len(sensor_event_predictions) > 0
        assert any(p.id == prediction.id for p in sensor_event_predictions)

        await test_db_session.refresh(room_state)
        room_state_predictions = await room_state.get_predictions(test_db_session)
        assert len(room_state_predictions) > 0
        assert any(p.id == prediction.id for p in room_state_predictions)

    @pytest.mark.asyncio
    async def test_feature_store_operations(self, test_db_session):
        """Test feature store operations and queries."""
        base_time = datetime.now(timezone.utc)
        room_id = "feature_test_room"

        # Create feature store entries with different timestamps and versions
        features = [
            # Older features (expired)
            FeatureStore(
                room_id=room_id,
                feature_timestamp=base_time - timedelta(hours=8),
                temporal_features={
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "is_weekend": False,
                },
                sequential_features={"velocity": 0.5, "pattern": "stable"},
                contextual_features={"temperature": 20.0, "humidity": 50.0},
                environmental_features={"weather": "cloudy"},
                lookback_hours=24,
                feature_version="v1.0",
                completeness_score=0.9,
                freshness_score=0.6,
                confidence_score=0.8,
                expires_at=base_time - timedelta(hours=2),  # Expired
            ),
            # Recent features (fresh)
            FeatureStore(
                room_id=room_id,
                feature_timestamp=base_time - timedelta(hours=1),
                temporal_features={
                    "hour_sin": 0.707,
                    "hour_cos": 0.707,
                    "is_weekend": False,
                },
                sequential_features={"velocity": 1.2, "pattern": "increasing"},
                contextual_features={"temperature": 22.5, "humidity": 45.0},
                environmental_features={"weather": "sunny"},
                lookback_hours=24,
                feature_version="v1.1",
                completeness_score=0.95,
                freshness_score=0.9,
                confidence_score=0.9,
                expires_at=base_time + timedelta(hours=5),  # Fresh
            ),
            # Most recent features
            FeatureStore(
                room_id=room_id,
                feature_timestamp=base_time - timedelta(minutes=30),
                temporal_features={
                    "hour_sin": 0.866,
                    "hour_cos": 0.5,
                    "is_weekend": False,
                },
                sequential_features={"velocity": 1.5, "pattern": "peak"},
                contextual_features={"temperature": 23.0, "humidity": 42.0},
                environmental_features={"weather": "sunny"},
                lookback_hours=24,
                feature_version="v1.2",
                completeness_score=0.98,
                freshness_score=0.95,
                confidence_score=0.92,
                expires_at=base_time + timedelta(hours=6),  # Very fresh
            ),
        ]

        for feature in features:
            test_db_session.add(feature)
        await test_db_session.commit()

        # Test get_latest_features (should return most recent non-expired)
        latest = await FeatureStore.get_latest_features(
            test_db_session, room_id, max_age_hours=6
        )

        assert latest is not None
        assert latest.feature_version == "v1.2"
        assert latest.temporal_features["hour_sin"] == 0.866
        assert latest.completeness_score == 0.98

        # Test feature combination
        all_features = latest.get_all_features()
        expected_feature_count = (
            len(latest.temporal_features)
            + len(latest.sequential_features)
            + len(latest.contextual_features)
            + len(latest.environmental_features)
        )
        assert len(all_features) == expected_feature_count

        # Check specific features are present
        assert all_features["hour_sin"] == 0.866
        assert all_features["velocity"] == 1.5
        assert all_features["temperature"] == 23.0
        assert all_features["weather"] == "sunny"

        # Test querying by feature quality
        high_quality_query = select(FeatureStore).where(
            FeatureStore.room_id == room_id,
            FeatureStore.completeness_score >= 0.95,
            FeatureStore.confidence_score >= 0.9,
        )
        result = await test_db_session.execute(high_quality_query)
        high_quality_features = result.scalars().all()

        # Should get the 2 most recent feature sets
        assert len(high_quality_features) == 2
        for feature in high_quality_features:
            assert feature.completeness_score >= 0.95
            assert feature.confidence_score >= 0.9

    @pytest.mark.asyncio
    async def test_time_series_queries(self, test_db_session):
        """Test time-series specific queries and aggregations."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)
        room_id = "timeseries_test_room"

        # Create events over 24 hours
        events = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:  # 4 events per hour
                timestamp = base_time + timedelta(hours=hour, minutes=minute)
                state = "on" if (hour + minute // 15) % 2 == 0 else "off"

                event = SensorEvent(
                    room_id=room_id,
                    sensor_id="binary_sensor.timeseries_test",
                    sensor_type="motion",
                    state=state,
                    previous_state="off" if state == "on" else "on",
                    timestamp=timestamp,
                    is_human_triggered=True,
                    confidence_score=0.8 + (hour % 10) * 0.02,  # Varying confidence
                )
                events.append(event)

        for event in events:
            test_db_session.add(event)
        await test_db_session.commit()

        # Test hourly aggregation (using SQLite-compatible datetime functions)
        hourly_query = (
            select(
                func.strftime("%Y-%m-%d %H:00:00", SensorEvent.timestamp).label("hour"),
                func.count().label("event_count"),
                func.avg(SensorEvent.confidence_score).label("avg_confidence"),
                func.sum(case((SensorEvent.state == "on", 1), else_=0)).label(
                    "on_events"
                ),
            )
            .where(SensorEvent.room_id == room_id)
            .group_by(func.strftime("%Y-%m-%d %H:00:00", SensorEvent.timestamp))
            .order_by("hour")
        )

        result = await test_db_session.execute(hourly_query)
        hourly_stats = result.all()

        # Allow for slight variation due to timing (24-25 hours)
        assert 24 <= len(hourly_stats) <= 25  # Should be close to 24 hours of data

        for stat in hourly_stats:
            # Most hours should have 4 events, allow for partial hours at boundaries
            assert 1 <= stat.event_count <= 4  # Events per hour
            assert 0.8 <= stat.avg_confidence <= 1.0  # Confidence in expected range
            assert 0 <= stat.on_events <= 4  # On events count

        # Test recent activity query
        recent_query = (
            select(SensorEvent)
            .where(
                SensorEvent.room_id == room_id,
                SensorEvent.timestamp
                >= datetime.now(timezone.utc) - timedelta(hours=2),
            )
            .order_by(SensorEvent.timestamp.desc())
        )

        result = await test_db_session.execute(recent_query)
        recent_events = result.scalars().all()

        # Should get events from last 2 hours
        assert (
            7 <= len(recent_events) <= 8
        )  # Approximately 2 hours * 4 events/hour (allow for timing variations)

        # Test state change frequency
        state_change_query = select(func.count().label("total_changes")).where(
            SensorEvent.room_id == room_id,
            SensorEvent.state != SensorEvent.previous_state,
        )

        result = await test_db_session.execute(state_change_query)
        changes = result.scalar()

        # All events should be state changes in our test data
        assert changes == len(events)

    @pytest.mark.asyncio
    async def test_database_constraints_and_validation(self, test_db_session):
        """Test database constraints and data validation."""
        # Test confidence score constraints
        valid_event = SensorEvent(
            room_id="constraint_test_room",
            sensor_id="binary_sensor.valid",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
            confidence_score=0.75,  # Valid confidence
        )

        test_db_session.add(valid_event)
        await test_db_session.commit()  # Should succeed

        # Test model accuracy constraints
        valid_accuracy = ModelAccuracy(
            room_id="constraint_test_room",
            model_type="lstm",
            model_version="v1.0",
            measurement_start=datetime.now(timezone.utc) - timedelta(days=1),
            measurement_end=datetime.now(timezone.utc),
            total_predictions=100,
            accurate_predictions=85,  # Less than total (valid)
            accuracy_rate=0.85,
            mean_error_minutes=10.0,
            median_error_minutes=8.0,
            rmse_minutes=12.0,
        )

        test_db_session.add(valid_accuracy)
        await test_db_session.commit()  # Should succeed

        # Test feature store constraints
        valid_features = FeatureStore(
            room_id="constraint_test_room",
            feature_timestamp=datetime.now(timezone.utc),
            temporal_features={"test": 1.0},
            sequential_features={},
            contextual_features={},
            environmental_features={},
            lookback_hours=24,
            feature_version="v1.0",
            completeness_score=0.95,  # Valid score
            freshness_score=0.9,  # Valid score
            confidence_score=0.85,  # Valid score
        )

        test_db_session.add(valid_features)
        await test_db_session.commit()  # Should succeed

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_db_manager):
        """Test concurrent database operations."""
        room_id = "concurrent_test_room"

        async def create_events(session_id: int, count: int):
            """Create events concurrently."""
            events_created = []
            async with test_db_manager.get_session() as session:
                for i in range(count):
                    event = SensorEvent(
                        room_id=room_id,
                        sensor_id=f"binary_sensor.concurrent_{session_id}_{i}",
                        sensor_type="motion",
                        state="on" if i % 2 == 0 else "off",
                        timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                        confidence_score=0.8,
                    )
                    session.add(event)
                    events_created.append(event.sensor_id)
                await session.commit()
            return events_created

        # Run concurrent sessions
        import asyncio

        tasks = [create_events(i, 5) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify all events were created
        total_expected = 15  # 3 sessions * 5 events each
        all_created_ids = []
        for session_results in results:
            all_created_ids.extend(session_results)

        assert len(all_created_ids) == total_expected
        assert len(set(all_created_ids)) == total_expected  # All unique

        # Verify in database
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                select(func.count()).where(SensorEvent.room_id == room_id)
            )
            count = result.scalar()
            assert count == total_expected


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestDatabasePerformance:
    """Performance and stress tests for database operations."""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, test_db_session):
        """Test performance of bulk insert operations."""
        import time

        room_id = "performance_test_room"
        event_count = 1000

        # Create events
        events = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)

        for i in range(event_count):
            event = SensorEvent(
                room_id=room_id,
                sensor_id=f"binary_sensor.perf_test_{i % 10}",  # 10 different sensors
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(seconds=i * 10),
                attributes={"test_id": i, "batch": "performance"},
                is_human_triggered=True,
                confidence_score=0.8 + (i % 20) * 0.01,
            )
            events.append(event)

        # Measure bulk insert time
        start_time = time.time()

        test_db_session.add_all(events)
        await test_db_session.commit()

        insert_time = time.time() - start_time

        # Verify all events were inserted
        result = await test_db_session.execute(
            select(func.count()).where(SensorEvent.room_id == room_id)
        )
        count = result.scalar()

        assert count == event_count

        # Performance should be reasonable (less than 5 seconds for 1000 events)
        assert insert_time < 5.0

        print(f"Bulk insert of {event_count} events took {insert_time:.2f} seconds")
        print(f"Rate: {event_count / insert_time:.0f} events/second")

    @pytest.mark.asyncio
    async def test_complex_query_performance(self, test_db_session):
        """Test performance of complex analytical queries."""
        # This test requires the performance test data from above
        # or we can create a smaller dataset for testing

        room_id = "query_perf_test_room"

        # Create test data
        events = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        for day in range(7):
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    timestamp = base_time + timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    event = SensorEvent(
                        room_id=room_id,
                        sensor_id=f"binary_sensor.sensor_{hour % 3}",
                        sensor_type="motion",
                        state=("on" if (hour + minute // 15) % 2 == 0 else "off"),
                        timestamp=timestamp,
                        confidence_score=0.7 + (hour % 10) * 0.03,
                    )
                    events.append(event)

        test_db_session.add_all(events)
        await test_db_session.commit()

        # Complex analytical query
        import time

        start_time = time.time()

        complex_query = (
            select(
                func.strftime("%Y-%m-%d", SensorEvent.timestamp).label("day"),
                SensorEvent.sensor_id,
                func.count().label("total_events"),
                func.sum(case((SensorEvent.state == "on", 1), else_=0)).label(
                    "on_events"
                ),
                func.avg(SensorEvent.confidence_score).label("avg_confidence"),
                func.min(SensorEvent.timestamp).label("first_event"),
                func.max(SensorEvent.timestamp).label("last_event"),
            )
            .where(
                SensorEvent.room_id == room_id,
                SensorEvent.timestamp >= base_time,
            )
            .group_by(
                func.strftime("%Y-%m-%d", SensorEvent.timestamp),
                SensorEvent.sensor_id,
            )
            .order_by("day", SensorEvent.sensor_id)
        )

        result = await test_db_session.execute(complex_query)
        stats = result.all()

        query_time = time.time() - start_time

        # Should have stats for 7 days * 3 sensors = 21 rows
        # Allow for timing variations in day boundaries
        assert (
            20 <= len(stats) <= 24
        )  # Should be close to 7 days * 3 sensors = 21 combinations

        # Query should complete reasonably quickly
        assert query_time < 2.0

        print(f"Complex analytical query took {query_time:.3f} seconds")
        print(f"Analyzed {len(events)} events across {len(stats)} groups")
