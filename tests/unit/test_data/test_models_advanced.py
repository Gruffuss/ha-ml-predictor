"""
Advanced tests for database models functionality.

Tests advanced model features, JSON operations, relationships, complex queries,
TimescaleDB functions, and edge cases not covered in basic tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
import uuid

from decimal import Decimal
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.data.storage.models import (
    FeatureStore,
    ModelAccuracy,
    Prediction,
    PredictionAudit,
    RoomState,
    SensorEvent,
    _get_json_column_type,
    _is_sqlite_engine,
    create_timescale_hypertables,
    get_bulk_insert_query,
    optimize_database_performance,
)


class TestSensorEventAdvancedFeatures:
    """Test advanced SensorEvent model features."""

    def test_sensor_event_initialization_defaults(self):
        """Test SensorEvent initialization with proper defaults."""
        # Test without explicit defaults
        event = SensorEvent(
            room_id="test_room",
            sensor_id="sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
        )

        # Should apply Python-level defaults
        assert event.is_human_triggered is True
        assert event.attributes == {}

        # Test with explicit defaults
        event2 = SensorEvent(
            room_id="test_room",
            sensor_id="sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
            is_human_triggered=False,
            attributes={"custom": "value"},
        )

        assert event2.is_human_triggered is False
        assert event2.attributes == {"custom": "value"}

    @pytest.mark.asyncio
    async def test_get_advanced_analytics(self, test_db_session):
        """Test advanced analytics with SQL functions."""
        base_time = datetime.now(timezone.utc)

        # Create diverse test events
        events = []
        confidence_scores = [0.7, 0.8, 0.9, 0.6, 0.95]

        for i, confidence in enumerate(confidence_scores):
            event = SensorEvent(
                room_id="analytics_room",
                sensor_id=f"sensor.test_{i % 3}",  # 3 unique sensors
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=base_time - timedelta(hours=i),
                confidence_score=confidence,
                is_human_triggered=(i % 3 != 0),  # Mix of human/automated
            )
            events.append(event)
            test_db_session.add(event)

        await test_db_session.commit()

        # Get advanced analytics
        analytics = await SensorEvent.get_advanced_analytics(
            test_db_session, "analytics_room", hours=6, include_statistics=True
        )

        # Test basic analytics
        assert analytics["room_id"] == "analytics_room"
        assert analytics["total_events"] == 5
        assert analytics["unique_sensors"] == 3
        assert analytics["human_triggered_events"] > 0
        assert analytics["automated_events"] > 0

        # Test confidence metrics
        assert 0.6 <= analytics["average_confidence"] <= 0.95
        assert 0 <= analytics["human_event_ratio"] <= 1

        # Test advanced statistics
        assert "median_confidence" in analytics
        assert "confidence_standard_deviation" in analytics
        assert "time_span_seconds" in analytics

        # Verify median calculation
        expected_median = sorted(confidence_scores)[len(confidence_scores) // 2]
        assert abs(analytics["median_confidence"] - expected_median) < 0.1

    @pytest.mark.asyncio
    async def test_get_advanced_analytics_without_statistics(self, test_db_session):
        """Test advanced analytics without statistical calculations."""
        event = SensorEvent(
            room_id="simple_room",
            sensor_id="sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
        )
        test_db_session.add(event)
        await test_db_session.commit()

        analytics = await SensorEvent.get_advanced_analytics(
            test_db_session, "simple_room", hours=1, include_statistics=False
        )

        # Should not include statistical fields
        assert "median_confidence" not in analytics
        assert "confidence_standard_deviation" not in analytics
        assert "time_span_seconds" not in analytics

        # Should still have basic analytics
        assert analytics["total_events"] == 1

    @pytest.mark.asyncio
    async def test_get_sensor_efficiency_metrics(self, test_db_session):
        """Test sensor efficiency metrics calculation."""
        base_time = datetime.now(timezone.utc)

        # Create events for multiple sensors with different patterns
        sensors_data = [
            {
                "sensor_id": "sensor.efficient",
                "events": 20,
                "state_changes": 15,
                "avg_confidence": 0.9,
            },
            {
                "sensor_id": "sensor.noisy",
                "events": 50,
                "state_changes": 10,
                "avg_confidence": 0.6,
            },
            {
                "sensor_id": "sensor.reliable",
                "events": 10,
                "state_changes": 8,
                "avg_confidence": 0.95,
            },
        ]

        for sensor_data in sensors_data:
            for i in range(sensor_data["events"]):
                # Create state change pattern based on expected changes
                has_state_change = i < sensor_data["state_changes"]

                event = SensorEvent(
                    room_id="efficiency_room",
                    sensor_id=sensor_data["sensor_id"],
                    sensor_type="motion",
                    state="on" if i % 2 == 0 else "off",
                    previous_state=(
                        "off"
                        if (i % 2 == 0 and has_state_change)
                        else ("on" if i % 2 == 0 else "off")
                    ),
                    timestamp=base_time - timedelta(minutes=i * 5),
                    confidence_score=sensor_data["avg_confidence"]
                    + (i % 3 - 1) * 0.05,  # Add variation
                )
                test_db_session.add(event)

        await test_db_session.commit()

        # Get efficiency metrics
        metrics = await SensorEvent.get_sensor_efficiency_metrics(
            test_db_session, "efficiency_room", days=1
        )

        assert len(metrics) == 3  # Three sensors

        # Test metrics structure
        for metric in metrics:
            required_fields = [
                "sensor_id",
                "sensor_type",
                "total_events",
                "state_changes",
                "average_confidence",
                "min_confidence",
                "max_confidence",
                "state_change_ratio",
                "average_interval_seconds",
                "efficiency_score",
            ]
            for field in required_fields:
                assert field in metric

            # Test efficiency score calculation
            assert 0 <= metric["efficiency_score"] <= 1
            assert metric["total_events"] > 0
            assert metric["state_change_ratio"] >= 0

        # Verify sensor-specific data
        sensor_metrics = {m["sensor_id"]: m for m in metrics}

        assert "sensor.efficient" in sensor_metrics
        assert "sensor.noisy" in sensor_metrics
        assert "sensor.reliable" in sensor_metrics

        # Efficient sensor should have good efficiency score
        efficient_metric = sensor_metrics["sensor.efficient"]
        reliable_metric = sensor_metrics["sensor.reliable"]

        # Reliable sensor should have highest efficiency due to high confidence
        assert (
            reliable_metric["efficiency_score"] >= efficient_metric["efficiency_score"]
        )

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation logic."""
        # Test various scenarios
        test_cases = [
            # (total_events, state_changes, avg_confidence, expected_score_range)
            (100, 50, 0.9, (0.6, 1.0)),  # High activity, good confidence
            (10, 5, 0.5, (0.2, 0.6)),  # Low activity, poor confidence
            (200, 100, 1.0, (0.8, 1.0)),  # High activity, perfect confidence
            (0, 0, 0.0, (0.0, 0.2)),  # No activity
        ]

        for (
            total_events,
            state_changes,
            avg_confidence,
            (min_score, max_score),
        ) in test_cases:
            state_change_ratio = state_changes / max(total_events, 1)

            score = SensorEvent._calculate_efficiency_score(
                total_events, state_changes, avg_confidence, state_change_ratio
            )

            assert (
                min_score <= score <= max_score
            ), f"Score {score} not in range {min_score}-{max_score}"

    @pytest.mark.asyncio
    async def test_get_temporal_patterns(self, test_db_session):
        """Test temporal pattern analysis."""
        base_time = datetime.now(timezone.utc).replace(
            hour=10, minute=0, second=0, microsecond=0
        )

        # Create events across different hours and days of week
        events = []
        for day_offset in range(7):  # One week
            for hour_offset in [0, 4, 8, 12, 16, 20]:  # Different hours
                timestamp = base_time - timedelta(days=day_offset, hours=hour_offset)

                event = SensorEvent(
                    room_id="pattern_room",
                    sensor_id="sensor.pattern",
                    sensor_type="motion",
                    state="on",
                    timestamp=timestamp,
                    confidence_score=0.8 + (hour_offset % 12) * 0.01,  # Vary by hour
                )
                events.append(event)
                test_db_session.add(event)

        await test_db_session.commit()

        # Get temporal patterns
        patterns = await SensorEvent.get_temporal_patterns(
            test_db_session, "pattern_room", days=8
        )

        # Test structure
        assert patterns["room_id"] == "pattern_room"
        assert patterns["analysis_period_days"] == 8
        assert "hourly_patterns" in patterns
        assert "day_of_week_patterns" in patterns
        assert "peak_hour" in patterns
        assert "peak_day" in patterns

        # Test hourly patterns
        hourly_patterns = patterns["hourly_patterns"]
        assert len(hourly_patterns) > 0

        for pattern in hourly_patterns:
            assert "hour" in pattern
            assert "event_count" in pattern
            assert "average_confidence" in pattern
            assert 0 <= pattern["hour"] <= 23
            assert pattern["event_count"] > 0

        # Test day of week patterns
        dow_patterns = patterns["day_of_week_patterns"]
        assert len(dow_patterns) > 0

        for pattern in dow_patterns:
            assert "day_of_week" in pattern
            assert "day_name" in pattern
            assert "event_count" in pattern
            assert "average_confidence" in pattern
            assert 0 <= pattern["day_of_week"] <= 6
            assert pattern["day_name"] in [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]

        # Test peak identification
        assert patterns["peak_hour"] is not None
        assert patterns["peak_day"] is not None

    @pytest.mark.asyncio
    async def test_get_predictions_relationship(self, test_db_session):
        """Test getting predictions related to sensor event."""
        # Create sensor event
        event = SensorEvent(
            room_id="test_room",
            sensor_id="sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
        )
        test_db_session.add(event)
        await test_db_session.flush()  # Get ID

        # Create prediction linked to this event
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            triggering_event_id=event.id,  # Link to sensor event
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get predictions for the sensor event
        predictions = await event.get_predictions(test_db_session)

        assert len(predictions) == 1
        assert predictions[0].id == prediction.id
        assert predictions[0].triggering_event_id == event.id


class TestRoomStateAdvancedFeatures:
    """Test advanced RoomState model features."""

    @pytest.mark.asyncio
    async def test_get_occupancy_sessions(self, test_db_session):
        """Test occupancy session analysis with UUIDs."""
        base_time = datetime.now(timezone.utc)

        # Create occupancy sessions with UUIDs
        session1_id = uuid.uuid4()
        session2_id = uuid.uuid4()

        # Session 1: 3 states
        session1_states = []
        for i in range(3):
            state = RoomState(
                room_id="session_room",
                timestamp=base_time - timedelta(hours=3, minutes=i * 10),
                occupancy_session_id=session1_id,
                is_occupied=True,
                occupancy_confidence=0.8 + i * 0.05,
                occupant_type="human",
                transition_trigger="motion_sensor",
            )
            session1_states.append(state)
            test_db_session.add(state)

        # Session 2: 2 states
        session2_states = []
        for i in range(2):
            state = RoomState(
                room_id="session_room",
                timestamp=base_time - timedelta(hours=1, minutes=i * 15),
                occupancy_session_id=session2_id,
                is_occupied=True,
                occupancy_confidence=0.7 + i * 0.1,
                occupant_type="cat",
                transition_trigger="motion_sensor",
            )
            session2_states.append(state)
            test_db_session.add(state)

        await test_db_session.commit()

        # Get occupancy sessions
        sessions = await RoomState.get_occupancy_sessions(
            test_db_session, "session_room", days=1
        )

        assert len(sessions) == 2

        # Test session structure
        for session in sessions:
            required_fields = [
                "session_id",
                "room_id",
                "states",
                "start_time",
                "end_time",
                "duration_seconds",
                "occupant_type",
                "confidence_range",
            ]
            for field in required_fields:
                assert field in session

            assert session["room_id"] == "session_room"
            assert len(session["states"]) > 0
            assert session["start_time"] is not None
            assert session["end_time"] is not None
            assert session["duration_seconds"] > 0

            # Test confidence range
            confidence_range = session["confidence_range"]
            assert "min" in confidence_range
            assert "max" in confidence_range
            assert "avg" in confidence_range
            assert (
                confidence_range["min"]
                <= confidence_range["avg"]
                <= confidence_range["max"]
            )

        # Verify session-specific data
        session_ids = [session["session_id"] for session in sessions]
        assert str(session1_id) in session_ids
        assert str(session2_id) in session_ids

    @pytest.mark.asyncio
    async def test_get_occupancy_sessions_with_optimized_loading(self, test_db_session):
        """Test occupancy sessions with selectinload optimization."""
        # Create a simple session
        session_id = uuid.uuid4()

        state = RoomState(
            room_id="optimized_room",
            timestamp=datetime.now(timezone.utc),
            occupancy_session_id=session_id,
            is_occupied=True,
            occupancy_confidence=0.85,
        )
        test_db_session.add(state)
        await test_db_session.commit()

        # Test with optimization enabled
        sessions_optimized = await RoomState.get_occupancy_sessions(
            test_db_session, "optimized_room", days=1, use_optimized_loading=True
        )

        # Test with optimization disabled
        sessions_normal = await RoomState.get_occupancy_sessions(
            test_db_session, "optimized_room", days=1, use_optimized_loading=False
        )

        # Should produce same results regardless of optimization
        assert len(sessions_optimized) == len(sessions_normal) == 1
        assert sessions_optimized[0]["session_id"] == sessions_normal[0]["session_id"]

    @pytest.mark.asyncio
    async def test_get_precision_occupancy_metrics(self, test_db_session):
        """Test precision occupancy metrics with Decimal calculations."""
        # Create room states with various confidence levels
        confidences = [
            Decimal("0.9500"),
            Decimal("0.8000"),
            Decimal("0.7500"),
            Decimal("0.6000"),
            Decimal("0.9000"),
            Decimal("0.8500"),
        ]

        for i, confidence in enumerate(confidences):
            state = RoomState(
                room_id="precision_room",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i * 10),
                is_occupied=True,
                occupancy_confidence=confidence,
            )
            test_db_session.add(state)

        await test_db_session.commit()

        # Get precision metrics with custom threshold
        threshold = Decimal("0.8000")
        metrics = await RoomState.get_precision_occupancy_metrics(
            test_db_session, "precision_room", precision_threshold=threshold
        )

        # Test metrics structure
        required_fields = [
            "room_id",
            "precision_threshold",
            "total_states",
            "high_confidence_states",
            "high_confidence_ratio",
            "confidence_statistics",
        ]
        for field in required_fields:
            assert field in metrics

        assert metrics["room_id"] == "precision_room"
        assert metrics["precision_threshold"] == float(threshold)
        assert metrics["total_states"] == 6

        # Count states above threshold
        expected_high_confidence = sum(1 for c in confidences if c >= threshold)
        assert metrics["high_confidence_states"] == expected_high_confidence

        expected_ratio = expected_high_confidence / 6
        assert abs(metrics["high_confidence_ratio"] - expected_ratio) < 0.01

        # Test confidence statistics
        stats = metrics["confidence_statistics"]
        assert "average" in stats
        assert "standard_deviation" in stats
        assert "minimum" in stats
        assert "maximum" in stats
        assert "quartiles" in stats

        # Verify quartile structure
        quartiles = stats["quartiles"]
        assert "q1" in quartiles
        assert "median" in quartiles
        assert "q3" in quartiles

        # Verify statistical relationships
        assert stats["minimum"] <= stats["average"] <= stats["maximum"]
        assert quartiles["q1"] <= quartiles["median"] <= quartiles["q3"]

    @pytest.mark.asyncio
    async def test_get_predictions_relationship(self, test_db_session):
        """Test getting predictions related to room state."""
        # Create room state
        room_state = RoomState(
            room_id="test_room",
            timestamp=datetime.now(timezone.utc),
            is_occupied=True,
            occupancy_confidence=0.85,
        )
        test_db_session.add(room_state)
        await test_db_session.flush()  # Get ID

        # Create prediction linked to this room state
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            room_state_id=room_state.id,  # Link to room state
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get predictions for the room state
        predictions = await room_state.get_predictions(test_db_session)

        assert len(predictions) == 1
        assert predictions[0].id == prediction.id
        assert predictions[0].room_state_id == room_state.id


class TestPredictionAdvancedFeatures:
    """Test advanced Prediction model features."""

    def test_prediction_time_compatibility(self):
        """Test prediction time compatibility between predicted_time and predicted_transition_time."""
        base_time = datetime.now(timezone.utc)
        transition_time = base_time + timedelta(minutes=15)

        # Test with only predicted_time
        pred1 = Prediction(
            room_id="test_room",
            prediction_time=base_time,
            predicted_time=transition_time,  # Old field
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )

        # Should set both fields to same value
        assert pred1.predicted_time == transition_time
        assert pred1.predicted_transition_time == transition_time

        # Test with both fields (different values)
        pred2 = Prediction(
            room_id="test_room",
            prediction_time=base_time,
            predicted_time=base_time + timedelta(minutes=10),
            predicted_transition_time=transition_time,  # Authoritative
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )

        # Should use predicted_transition_time as authoritative
        assert pred2.predicted_time == transition_time
        assert pred2.predicted_transition_time == transition_time

    def test_actual_time_compatibility(self):
        """Test actual time compatibility between actual_time and actual_transition_time."""
        base_time = datetime.now(timezone.utc)
        actual_time = base_time + timedelta(minutes=20)

        # Test with only actual_time
        pred1 = Prediction(
            room_id="test_room",
            prediction_time=base_time,
            predicted_transition_time=base_time + timedelta(minutes=15),
            actual_time=actual_time,  # Old field
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )

        # Should set both fields to same value
        assert pred1.actual_time == actual_time
        assert pred1.actual_transition_time == actual_time

        # Test with both fields (different values)
        pred2 = Prediction(
            room_id="test_room",
            prediction_time=base_time,
            predicted_transition_time=base_time + timedelta(minutes=15),
            actual_time=base_time + timedelta(minutes=18),
            actual_transition_time=actual_time,  # Authoritative
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )

        # Should use actual_transition_time as authoritative
        assert pred2.actual_time == actual_time
        assert pred2.actual_transition_time == actual_time

    @pytest.mark.asyncio
    async def test_get_triggering_event(self, test_db_session):
        """Test getting triggering sensor event."""
        # Create sensor event
        event = SensorEvent(
            room_id="test_room",
            sensor_id="sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.now(timezone.utc),
        )
        test_db_session.add(event)
        await test_db_session.flush()  # Get ID

        # Create prediction with triggering event
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            triggering_event_id=event.id,
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get triggering event
        triggering_event = await prediction.get_triggering_event(test_db_session)

        assert triggering_event is not None
        assert triggering_event.id == event.id
        assert triggering_event.sensor_id == "sensor.test"

        # Test with no triggering event
        prediction_no_trigger = Prediction(
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            triggering_event_id=None,
        )
        test_db_session.add(prediction_no_trigger)
        await test_db_session.commit()

        no_event = await prediction_no_trigger.get_triggering_event(test_db_session)
        assert no_event is None

    @pytest.mark.asyncio
    async def test_get_room_state(self, test_db_session):
        """Test getting associated room state."""
        # Create room state
        room_state = RoomState(
            room_id="test_room",
            timestamp=datetime.now(timezone.utc),
            is_occupied=True,
            occupancy_confidence=0.85,
        )
        test_db_session.add(room_state)
        await test_db_session.flush()  # Get ID

        # Create prediction with room state
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=10),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            room_state_id=room_state.id,
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get room state
        associated_state = await prediction.get_room_state(test_db_session)

        assert associated_state is not None
        assert associated_state.id == room_state.id
        assert associated_state.is_occupied is True

    @pytest.mark.asyncio
    async def test_get_predictions_with_events(self, test_db_session):
        """Test getting predictions with triggering events using application-level joins."""
        base_time = datetime.now(timezone.utc)

        # Create sensor events
        events = []
        for i in range(3):
            event = SensorEvent(
                room_id="join_room",
                sensor_id=f"sensor.test_{i}",
                sensor_type="motion",
                state="on",
                timestamp=base_time - timedelta(minutes=i * 10),
            )
            events.append(event)
            test_db_session.add(event)

        await test_db_session.flush()  # Get IDs

        # Create predictions linked to events
        predictions = []
        for i, event in enumerate(events):
            prediction = Prediction(
                room_id="join_room",
                prediction_time=base_time - timedelta(minutes=i * 10),
                predicted_transition_time=base_time + timedelta(minutes=10 - i * 5),
                transition_type="occupied_to_vacant",
                confidence_score=0.8 + i * 0.05,
                model_type="lstm",
                model_version="v1.0",
                triggering_event_id=event.id,
            )
            predictions.append(prediction)
            test_db_session.add(prediction)

        await test_db_session.commit()

        # Get predictions with events
        predictions_with_events = await Prediction.get_predictions_with_events(
            test_db_session, "join_room", hours=1
        )

        assert len(predictions_with_events) == 3

        # Test structure
        for pred, event in predictions_with_events:
            assert pred is not None
            assert event is not None  # Should have triggering events
            assert pred.triggering_event_id == event.id
            assert pred.room_id == "join_room"
            assert event.room_id == "join_room"

    @pytest.mark.asyncio
    async def test_get_predictions_with_full_context(self, test_db_session):
        """Test getting predictions with full context including JSON analysis."""
        base_time = datetime.now(timezone.utc)

        # Create prediction with rich JSON data
        feature_importance = {
            "temporal_hour_sin": 0.3,
            "sequential_transition_count": 0.25,
            "contextual_room_correlation": 0.2,
            "environmental_temperature": 0.15,
            "other_feature": 0.1,
        }

        alternatives = [
            {"time": "2024-01-15T14:35:00", "confidence": 0.7, "type": "alternative_1"},
            {"time": "2024-01-15T14:40:00", "confidence": 0.6, "type": "alternative_2"},
            {"time": "2024-01-15T14:25:00", "confidence": 0.5, "type": "alternative_3"},
        ]

        prediction = Prediction(
            room_id="context_room",
            prediction_time=base_time,
            predicted_transition_time=base_time + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.85,
            model_type="ensemble",
            model_version="v2.1",
            feature_importance=feature_importance,
            alternatives=alternatives,
            accuracy_minutes=8.5,
            is_accurate=True,
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get predictions with full context
        context_predictions = await Prediction.get_predictions_with_full_context(
            test_db_session, "context_room", hours=1, include_alternatives=True
        )

        assert len(context_predictions) == 1
        context = context_predictions[0]

        # Test basic fields
        assert context["room_id"] == "context_room"
        assert context["model_type"] == "ensemble"
        assert context["confidence_score"] == 0.85
        assert context["is_accurate"] is True

        # Test feature analysis
        feature_analysis = context["feature_analysis"]
        assert feature_analysis["total_features"] == 5
        assert "top_features" in feature_analysis
        assert "feature_categories" in feature_analysis

        # Test top features extraction
        top_features = feature_analysis["top_features"]
        assert len(top_features) <= 10  # Limited to top 10
        assert all(isinstance(f, tuple) and len(f) == 2 for f in top_features)
        # Should be sorted by importance (descending)
        importances = [f[1] for f in top_features]
        assert importances == sorted(importances, reverse=True)

        # Test feature categorization
        categories = feature_analysis["feature_categories"]
        expected_categories = [
            "temporal",
            "sequential",
            "contextual",
            "environmental",
            "other",
        ]
        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], int)

        # Test alternatives analysis
        alternatives_analysis = context["alternatives_analysis"]
        assert alternatives_analysis["total_alternatives"] == 3
        assert "confidence_spread" in alternatives_analysis
        assert "alternative_predictions" in alternatives_analysis

        # Test confidence spread analysis
        confidence_spread = alternatives_analysis["confidence_spread"]
        assert "min" in confidence_spread
        assert "max" in confidence_spread
        assert "mean" in confidence_spread
        assert "std" in confidence_spread

        # Should have correct confidence values
        assert confidence_spread["min"] == 0.5
        assert confidence_spread["max"] == 0.7
        assert abs(confidence_spread["mean"] - (0.7 + 0.6 + 0.5) / 3) < 0.01

    @pytest.mark.asyncio
    async def test_get_predictions_with_full_context_without_alternatives(
        self, test_db_session
    ):
        """Test getting predictions with context but without alternatives."""
        prediction = Prediction(
            room_id="context_room_simple",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            feature_importance={"feature1": 0.6, "feature2": 0.4},
        )
        test_db_session.add(prediction)
        await test_db_session.commit()

        # Get predictions without alternatives
        context_predictions = await Prediction.get_predictions_with_full_context(
            test_db_session, "context_room_simple", hours=1, include_alternatives=False
        )

        assert len(context_predictions) == 1
        context = context_predictions[0]

        # Should not include alternatives analysis
        assert "alternatives_analysis" not in context

        # Should still include feature analysis
        assert "feature_analysis" in context

    def test_extract_top_features(self):
        """Test top features extraction from JSON data."""
        feature_importance = {
            "high_importance": 0.8,
            "medium_importance": 0.5,
            "low_importance": 0.2,
            "negative_importance": -0.3,
            "string_value": "not_a_number",  # Should be filtered out
            "zero_importance": 0.0,
        }

        top_features = Prediction._extract_top_features(feature_importance)

        # Should only include numeric values
        assert len(top_features) == 5  # Excludes string_value

        # Should be sorted by absolute importance
        importances = [abs(f[1]) for f in top_features]
        assert importances == sorted(importances, reverse=True)

        # Test with empty input
        empty_features = Prediction._extract_top_features({})
        assert empty_features == []

        # Test with None input
        none_features = Prediction._extract_top_features(None)
        assert none_features == []

    def test_categorize_features(self):
        """Test feature categorization logic."""
        features = {
            "time_since_last": 0.3,
            "hour_cyclical": 0.2,
            "day_of_week": 0.1,
            "sequence_pattern": 0.25,
            "transition_count": 0.15,
            "movement_velocity": 0.1,
            "room_correlation": 0.2,
            "cross_room_activity": 0.15,
            "temperature_avg": 0.1,
            "humidity_level": 0.05,
            "light_sensor": 0.08,
            "unknown_feature": 0.1,
        }

        categories = Prediction._categorize_features(features)

        # Test category structure
        expected_categories = [
            "temporal",
            "sequential",
            "contextual",
            "environmental",
            "other",
        ]
        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], int)

        # Test specific categorizations
        assert (
            categories["temporal"] >= 3
        )  # time_since_last, hour_cyclical, day_of_week
        assert (
            categories["sequential"] >= 3
        )  # sequence_pattern, transition_count, movement_velocity
        assert categories["contextual"] >= 2  # room_correlation, cross_room_activity
        assert (
            categories["environmental"] >= 3
        )  # temperature_avg, humidity_level, light_sensor
        assert categories["other"] >= 1  # unknown_feature

    def test_analyze_confidence_spread(self):
        """Test confidence spread analysis for alternatives."""
        # Test with valid alternatives
        alternatives = [
            {"confidence": 0.8, "other_data": "value1"},
            {"confidence": 0.6, "other_data": "value2"},
            {"confidence": 0.9, "other_data": "value3"},
        ]

        spread = Prediction._analyze_confidence_spread(alternatives)

        assert spread["min"] == 0.6
        assert spread["max"] == 0.9
        assert spread["mean"] == (0.8 + 0.6 + 0.9) / 3
        assert spread["std"] > 0  # Should have some variance

        # Test with empty alternatives
        empty_spread = Prediction._analyze_confidence_spread([])
        assert empty_spread["min"] == 0.0
        assert empty_spread["max"] == 0.0
        assert empty_spread["mean"] == 0.0
        assert empty_spread["std"] == 0.0

        # Test with alternatives without confidence
        invalid_alternatives = [
            {"other_field": "value1"},
            {"other_field": "value2"},
        ]

        invalid_spread = Prediction._analyze_confidence_spread(invalid_alternatives)
        assert invalid_spread["min"] == 0.0
        assert invalid_spread["max"] == 0.0
        assert invalid_spread["mean"] == 0.0
        assert invalid_spread["std"] == 0.0

    def test_add_extended_metadata(self):
        """Test adding extended metadata to JSON field."""
        base_time = datetime.now(timezone.utc)

        prediction = Prediction(
            room_id="metadata_room",
            prediction_time=base_time,
            predicted_transition_time=base_time + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            feature_importance={"existing_feature": 0.5},
        )

        # Add extended metadata
        metadata = {
            "custom_field": "custom_value",
            "numeric_field": 42,
            "nested_field": {"sub_field": "sub_value"},
        }

        prediction.add_extended_metadata(metadata)

        # Should preserve existing features
        assert prediction.feature_importance["existing_feature"] == 0.5

        # Should add metadata section
        assert "_metadata" in prediction.feature_importance
        metadata_section = prediction.feature_importance["_metadata"]

        assert "extended_info" in metadata_section
        assert "added_at" in metadata_section
        assert "version" in metadata_section

        # Should contain the added metadata
        extended_info = metadata_section["extended_info"]
        assert extended_info == metadata

        # Should have timestamp and version
        assert metadata_section["version"] == "1.0"
        assert "T" in metadata_section["added_at"]  # ISO format timestamp


class TestPredictionAuditRelationships:
    """Test PredictionAudit model with proper relationships."""

    @pytest.mark.asyncio
    async def test_create_audit_entry(self, test_db_session):
        """Test creating audit entry with JSON details."""
        # Create prediction first
        prediction = Prediction(
            room_id="audit_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )
        test_db_session.add(prediction)
        await test_db_session.flush()  # Get ID

        # Create audit entry
        audit_details = {
            "action_type": "validation",
            "validation_result": "accurate",
            "error_margin_minutes": 5.2,
            "confidence_delta": 0.15,
        }

        audit = await PredictionAudit.create_audit_entry(
            test_db_session,
            prediction_id=prediction.id,
            action="validated",
            details=audit_details,
            user="test_user",
            notes="Prediction validated successfully",
        )

        assert audit.prediction_id == prediction.id
        assert audit.audit_action == "validated"
        assert audit.audit_user == "test_user"
        assert audit.audit_notes == "Prediction validated successfully"
        assert audit.audit_details == audit_details

    @pytest.mark.asyncio
    async def test_get_audit_trail_with_relationships(self, test_db_session):
        """Test getting audit trail with relationship loading."""
        # Create prediction
        prediction = Prediction(
            room_id="audit_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )
        test_db_session.add(prediction)
        await test_db_session.flush()

        # Create multiple audit entries
        audit_entries = []
        for i in range(3):
            audit = await PredictionAudit.create_audit_entry(
                test_db_session,
                prediction_id=prediction.id,
                action=f"action_{i}",
                details={"step": i},
                user="test_user",
            )
            audit_entries.append(audit)

        await test_db_session.commit()

        # Get audit trail with relationships
        trail = await PredictionAudit.get_audit_trail_with_relationships(
            test_db_session, prediction.id, load_related=True
        )

        assert len(trail) == 3

        # Should be ordered by timestamp
        timestamps = [audit.audit_timestamp for audit in trail]
        assert timestamps == sorted(timestamps)

        # Test without relationship loading
        trail_no_rel = await PredictionAudit.get_audit_trail_with_relationships(
            test_db_session, prediction.id, load_related=False
        )

        assert len(trail_no_rel) == 3

    def test_analyze_json_details(self):
        """Test JSON audit details analysis."""
        audit_details = {
            "string_field": "value",
            "numeric_field": 42,
            "float_field": 3.14,
            "boolean_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"},
            "metrics": {"accuracy": 0.95},
            "errors": ["Error 1", "Error 2"],
        }

        audit = PredictionAudit(
            prediction_id=1,
            audit_action="test",
            audit_details=audit_details,
        )

        analysis = audit.analyze_json_details()

        # Test basic analysis
        assert analysis["total_fields"] == 8
        assert analysis["has_metrics"] is True
        assert analysis["has_errors"] is True
        assert analysis["complexity_score"] > 0

        # Test field type counting
        field_types = analysis["field_types"]
        assert "str" in field_types
        assert "int" in field_types
        assert "float" in field_types
        assert "bool" in field_types
        assert "list" in field_types
        assert "dict" in field_types

    def test_calculate_json_complexity(self):
        """Test JSON complexity calculation."""
        # Simple data
        simple_data = {"key": "value"}
        simple_complexity = PredictionAudit._calculate_json_complexity(simple_data)
        assert simple_complexity == 2  # 1 for dict + 1 for value

        # Nested data
        nested_data = {"level1": {"level2": {"level3": ["item1", "item2"]}}}
        nested_complexity = PredictionAudit._calculate_json_complexity(nested_data)
        assert nested_complexity > simple_complexity

        # Deep nesting (should handle recursion limit)
        deep_data = {"a": {"b": {"c": {"d": {"e": {"f": "value"}}}}}}
        deep_complexity = PredictionAudit._calculate_json_complexity(deep_data)
        assert deep_complexity > 0  # Should not crash

    def test_update_validation_metrics(self):
        """Test updating JSON validation metrics."""
        audit = PredictionAudit(
            prediction_id=1,
            audit_action="validation",
            validation_metrics={"initial_metric": 0.5},
        )

        # Update metrics
        new_metrics = {
            "accuracy": 0.95,
            "precision": 0.87,
            "recall": 0.92,
        }

        audit.update_validation_metrics(new_metrics)

        # Should preserve existing metrics
        assert audit.validation_metrics["initial_metric"] == 0.5

        # Should add new metrics
        assert audit.validation_metrics["accuracy"] == 0.95
        assert audit.validation_metrics["precision"] == 0.87
        assert audit.validation_metrics["recall"] == 0.92

        # Should add timestamp
        assert "last_updated" in audit.validation_metrics


class TestFeatureStoreAdvanced:
    """Test advanced FeatureStore functionality."""

    @pytest.mark.asyncio
    async def test_get_latest_features_with_expiration(self, test_db_session):
        """Test getting latest features with expiration handling."""
        base_time = datetime.now(timezone.utc)

        # Create expired feature store
        expired_features = FeatureStore(
            room_id="feature_room",
            feature_timestamp=base_time - timedelta(hours=2),
            temporal_features={"hour": 10},
            sequential_features={"transitions": 5},
            contextual_features={"room_count": 3},
            environmental_features={"temp": 22.5},
            lookback_hours=24,
            feature_version="v1.0",
            expires_at=base_time - timedelta(hours=1),  # Expired
        )
        test_db_session.add(expired_features)

        # Create valid feature store
        valid_features = FeatureStore(
            room_id="feature_room",
            feature_timestamp=base_time - timedelta(minutes=30),
            temporal_features={"hour": 11},
            sequential_features={"transitions": 7},
            contextual_features={"room_count": 2},
            environmental_features={"temp": 23.0},
            lookback_hours=24,
            feature_version="v1.1",
            expires_at=base_time + timedelta(hours=1),  # Not expired
        )
        test_db_session.add(valid_features)

        await test_db_session.commit()

        # Get latest features
        latest = await FeatureStore.get_latest_features(
            test_db_session, "feature_room", max_age_hours=3
        )

        # Should get the valid (non-expired) features
        assert latest is not None
        assert latest.feature_version == "v1.1"
        assert latest.temporal_features["hour"] == 11

    @pytest.mark.asyncio
    async def test_get_latest_features_no_expiration(self, test_db_session):
        """Test getting latest features without expiration set."""
        base_time = datetime.now(timezone.utc)

        # Create feature store without expiration
        features = FeatureStore(
            room_id="no_expire_room",
            feature_timestamp=base_time - timedelta(minutes=30),
            temporal_features={"hour": 12},
            sequential_features={"transitions": 3},
            contextual_features={"room_count": 1},
            environmental_features={"temp": 21.0},
            lookback_hours=12,
            feature_version="v1.0",
            expires_at=None,  # No expiration
        )
        test_db_session.add(features)
        await test_db_session.commit()

        # Get latest features
        latest = await FeatureStore.get_latest_features(
            test_db_session, "no_expire_room", max_age_hours=1
        )

        assert latest is not None
        assert latest.feature_version == "v1.0"

    def test_get_all_features(self):
        """Test combining all feature categories."""
        features = FeatureStore(
            room_id="combine_room",
            feature_timestamp=datetime.now(timezone.utc),
            temporal_features={"temp1": 1.0, "temp2": 2.0},
            sequential_features={"seq1": 3.0, "seq2": 4.0},
            contextual_features={"ctx1": 5.0},
            environmental_features={"env1": 6.0, "env2": 7.0},
            lookback_hours=24,
            feature_version="v1.0",
        )

        all_features = features.get_all_features()

        # Should combine all features
        expected_features = {
            "temp1": 1.0,
            "temp2": 2.0,  # temporal
            "seq1": 3.0,
            "seq2": 4.0,  # sequential
            "ctx1": 5.0,  # contextual
            "env1": 6.0,
            "env2": 7.0,  # environmental
        }

        assert all_features == expected_features

    def test_get_all_features_with_none_values(self):
        """Test combining features with None values."""
        features = FeatureStore(
            room_id="none_room",
            feature_timestamp=datetime.now(timezone.utc),
            temporal_features={"temp1": 1.0},
            sequential_features=None,  # None value
            contextual_features={"ctx1": 2.0},
            environmental_features=None,  # None value
            lookback_hours=24,
            feature_version="v1.0",
        )

        all_features = features.get_all_features()

        # Should handle None values gracefully
        expected_features = {
            "temp1": 1.0,
            "ctx1": 2.0,
        }

        assert all_features == expected_features


class TestTimescaleDBFunctions:
    """Test TimescaleDB-specific functionality."""

    @pytest.mark.asyncio
    async def test_create_timescale_hypertables(self, test_db_session):
        """Test TimescaleDB hypertable creation."""
        with patch.object(test_db_session, "execute") as mock_execute:
            await create_timescale_hypertables(test_db_session)

            # Should have executed multiple TimescaleDB commands
            assert mock_execute.call_count >= 6

            # Check for key TimescaleDB operations
            executed_queries = [str(call[0][0]) for call in mock_execute.call_args_list]

            # Should create hypertable
            hypertable_queries = [
                q for q in executed_queries if "create_hypertable" in q
            ]
            assert len(hypertable_queries) > 0

            # Should create continuous aggregate
            aggregate_queries = [
                q for q in executed_queries if "sensor_events_hourly" in q
            ]
            assert len(aggregate_queries) > 0

            # Should set up compression
            compression_queries = [
                q for q in executed_queries if "compress" in q.lower()
            ]
            assert len(compression_queries) > 0

            # Should add policies
            policy_queries = [
                q for q in executed_queries if "add_" in q and "policy" in q
            ]
            assert len(policy_queries) > 0

    @pytest.mark.asyncio
    async def test_optimize_database_performance(self, test_db_session):
        """Test database performance optimization."""
        with patch.object(test_db_session, "execute") as mock_execute:
            await optimize_database_performance(test_db_session)

            # Should have executed optimization commands
            executed_queries = [str(call[0][0]) for call in mock_execute.call_args_list]

            # Should analyze tables
            analyze_queries = [q for q in executed_queries if "ANALYZE" in q.upper()]
            assert len(analyze_queries) >= 3  # At least 3 tables

            # Should create performance indexes
            index_queries = [q for q in executed_queries if "CREATE INDEX" in q.upper()]
            assert len(index_queries) >= 2  # At least 2 performance indexes

    def test_get_bulk_insert_query(self):
        """Test bulk insert query generation."""
        query = get_bulk_insert_query()

        # Should be a proper SQL query
        assert "INSERT INTO sensor_events" in query
        assert "VALUES %s" in query
        assert "ON CONFLICT" in query
        assert "DO UPDATE SET" in query

        # Should include all sensor_events columns
        required_columns = [
            "timestamp",
            "room_id",
            "sensor_id",
            "sensor_type",
            "state",
            "previous_state",
            "attributes",
            "is_human_triggered",
            "confidence_score",
        ]

        for column in required_columns:
            assert column in query


class TestDatabaseCompatibilityHelpers:
    """Test database compatibility helper functions."""

    def test_is_sqlite_engine_helper(self):
        """Test SQLite engine detection helper."""
        # Test with mock bind that has SQLite URL
        mock_bind = Mock()
        mock_bind.url = "sqlite:///test.db"

        assert _is_sqlite_engine(mock_bind) is True

        # Test with PostgreSQL URL
        mock_bind.url = "postgresql://localhost/db"
        assert _is_sqlite_engine(mock_bind) is False

        # Test with None bind
        assert _is_sqlite_engine(None) is False

    @patch.dict("os.environ", {"TEST_DB_URL": "sqlite:///test.db"})
    def test_get_json_column_type_sqlite(self):
        """Test JSON column type selection for SQLite."""
        from sqlalchemy import JSON

        json_type = _get_json_column_type()
        assert json_type == JSON

    @patch.dict("os.environ", {"TESTING": "false", "TEST_DB_URL": ""})
    def test_get_json_column_type_postgresql(self):
        """Test JSON column type selection for PostgreSQL."""
        from sqlalchemy.dialects.postgresql import JSONB

        json_type = _get_json_column_type()
        assert json_type == JSONB

    @patch.dict("os.environ", {"TESTING": "true"})
    def test_get_json_column_type_testing_env(self):
        """Test JSON column type selection in testing environment."""
        from sqlalchemy import JSON

        json_type = _get_json_column_type()
        assert json_type == JSON


@pytest.mark.unit
@pytest.mark.database_models
class TestModelsIntegration:
    """Integration tests for model functionality."""

    @pytest.mark.asyncio
    async def test_model_relationships_cascade(self, test_db_session):
        """Test model relationships and cascade behavior."""
        # Create prediction
        prediction = Prediction(
            room_id="cascade_room",
            prediction_time=datetime.now(timezone.utc),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
        )
        test_db_session.add(prediction)
        await test_db_session.flush()  # Get ID

        # Create audit entry with CASCADE delete
        audit = await PredictionAudit.create_audit_entry(
            test_db_session,
            prediction_id=prediction.id,
            action="created",
            details={"initial": "creation"},
        )

        await test_db_session.commit()

        # Verify relationship exists
        trail = await PredictionAudit.get_audit_trail_with_relationships(
            test_db_session, prediction.id
        )
        assert len(trail) == 1

        # Delete prediction (should cascade to audit due to ForeignKey ondelete="CASCADE")
        await test_db_session.delete(prediction)
        await test_db_session.commit()

        # Audit entry should be deleted due to cascade
        remaining_trail = await PredictionAudit.get_audit_trail_with_relationships(
            test_db_session, prediction.id
        )
        assert len(remaining_trail) == 0

    @pytest.mark.asyncio
    async def test_comprehensive_model_workflow(self, test_db_session):
        """Test comprehensive workflow across all models."""
        base_time = datetime.now(timezone.utc)

        # 1. Create sensor event
        event = SensorEvent(
            room_id="workflow_room",
            sensor_id="sensor.motion",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=base_time,
            confidence_score=0.9,
        )
        test_db_session.add(event)
        await test_db_session.flush()

        # 2. Create room state
        room_state = RoomState(
            room_id="workflow_room",
            timestamp=base_time,
            is_occupied=True,
            occupancy_confidence=0.85,
            occupancy_session_id=uuid.uuid4(),
            transition_trigger="motion_sensor",
        )
        test_db_session.add(room_state)
        await test_db_session.flush()

        # 3. Create feature store
        features = FeatureStore(
            room_id="workflow_room",
            feature_timestamp=base_time,
            temporal_features={"hour": 14, "day_of_week": 1},
            sequential_features={"transitions": 3},
            contextual_features={"cross_room": 0.7},
            environmental_features={"temp": 22.0},
            lookback_hours=24,
            feature_version="v1.0",
            completeness_score=0.95,
            freshness_score=0.9,
            confidence_score=0.88,
        )
        test_db_session.add(features)
        await test_db_session.flush()

        # 4. Create prediction
        prediction = Prediction(
            room_id="workflow_room",
            prediction_time=base_time,
            predicted_transition_time=base_time + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.82,
            model_type="ensemble",
            model_version="v2.0",
            triggering_event_id=event.id,
            room_state_id=room_state.id,
            feature_importance={"temp": 0.4, "transitions": 0.6},
            processing_time_ms=45.2,
        )
        test_db_session.add(prediction)
        await test_db_session.flush()

        # 5. Create model accuracy record
        accuracy = ModelAccuracy(
            room_id="workflow_room",
            model_type="ensemble",
            model_version="v2.0",
            measurement_start=base_time - timedelta(hours=24),
            measurement_end=base_time,
            total_predictions=100,
            accurate_predictions=85,
            accuracy_rate=0.85,
            mean_error_minutes=8.5,
            median_error_minutes=6.2,
            rmse_minutes=12.3,
            confidence_correlation=0.78,
            overconfidence_rate=0.15,
            feature_drift_score=0.05,
            concept_drift_score=0.02,
            performance_degradation=0.03,
        )
        test_db_session.add(accuracy)
        await test_db_session.flush()

        # 6. Create prediction audit
        audit = await PredictionAudit.create_audit_entry(
            test_db_session,
            prediction_id=prediction.id,
            action="created",
            details={
                "model_performance": "good",
                "feature_quality": "high",
                "confidence_level": "medium",
            },
            user="system",
            notes="Automatic prediction generation",
        )

        await test_db_session.commit()

        # Verify all relationships work

        # Get prediction with related data
        triggering_event = await prediction.get_triggering_event(test_db_session)
        assert triggering_event.id == event.id

        associated_room_state = await prediction.get_room_state(test_db_session)
        assert associated_room_state.id == room_state.id

        # Get audit trail
        audit_trail = await PredictionAudit.get_audit_trail_with_relationships(
            test_db_session, prediction.id
        )
        assert len(audit_trail) == 1
        assert audit_trail[0].audit_action == "created"

        # Get latest features
        latest_features = await FeatureStore.get_latest_features(
            test_db_session, "workflow_room"
        )
        assert latest_features.id == features.id

        # Get advanced analytics
        analytics = await SensorEvent.get_advanced_analytics(
            test_db_session, "workflow_room", hours=2
        )
        assert analytics["total_events"] >= 1
        assert analytics["room_id"] == "workflow_room"

        # Verify data integrity
        assert prediction.room_id == event.room_id == room_state.room_id
        assert prediction.triggering_event_id == event.id
        assert prediction.room_state_id == room_state.id
