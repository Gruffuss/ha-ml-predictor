"""
Unit tests for database models.

Tests SQLAlchemy models, relationships, class methods, and data validation.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

from src.data.storage.models import (
    SensorEvent, RoomState, Prediction, ModelAccuracy, FeatureStore,
    create_timescale_hypertables, optimize_database_performance,
    get_bulk_insert_query, Base
)
from src.core.constants import SensorType, SensorState


class TestSensorEvent:
    """Test SensorEvent model."""
    
    def test_sensor_event_creation(self):
        """Test creating a sensor event with all fields."""
        timestamp = datetime.utcnow()
        attributes = {"device_class": "motion", "friendly_name": "Test Sensor"}
        
        event = SensorEvent(
            room_id="living_room",
            sensor_id="binary_sensor.test_motion",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=timestamp,
            attributes=attributes,
            is_human_triggered=True,
            confidence_score=0.85,
            created_at=timestamp
        )
        
        assert event.room_id == "living_room"
        assert event.sensor_id == "binary_sensor.test_motion"
        assert event.sensor_type == "motion"
        assert event.state == "on"
        assert event.previous_state == "off"
        assert event.timestamp == timestamp
        assert event.attributes == attributes
        assert event.is_human_triggered is True
        assert event.confidence_score == 0.85
        assert event.created_at == timestamp
    
    def test_sensor_event_minimal(self):
        """Test creating sensor event with minimal required fields."""
        event = SensorEvent(
            room_id="bedroom",
            sensor_id="binary_sensor.bedroom_motion",
            sensor_type="presence",
            state="off",
            timestamp=datetime.utcnow()
        )
        
        assert event.room_id == "bedroom"
        assert event.sensor_id == "binary_sensor.bedroom_motion"
        assert event.sensor_type == "presence"
        assert event.state == "off"
        assert event.is_human_triggered is True  # Default value
    
    @pytest.mark.asyncio
    async def test_get_recent_events(self, test_db_session, sample_sensor_events):
        """Test getting recent events for a room."""
        # Add events to database
        for event in sample_sensor_events:
            test_db_session.add(event)
        await test_db_session.commit()
        
        # Get recent events
        recent_events = await SensorEvent.get_recent_events(
            test_db_session, 
            "test_room", 
            hours=2
        )
        
        assert len(recent_events) == len(sample_sensor_events)
        # Should be ordered by timestamp desc
        assert recent_events[0].timestamp >= recent_events[-1].timestamp
    
    @pytest.mark.asyncio
    async def test_get_recent_events_with_sensor_filter(self, test_db_session, sample_sensor_events):
        """Test getting recent events filtered by sensor type."""
        # Add events to database
        for event in sample_sensor_events:
            test_db_session.add(event)
        await test_db_session.commit()
        
        # Get recent events filtered by sensor type
        recent_events = await SensorEvent.get_recent_events(
            test_db_session,
            "test_room",
            hours=2,
            sensor_types=["presence"]
        )
        
        assert len(recent_events) > 0
        for event in recent_events:
            assert event.sensor_type == "presence"
    
    @pytest.mark.asyncio
    async def test_get_state_changes(self, test_db_session):
        """Test getting events where state changed."""
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        # Create events with state changes
        events = [
            SensorEvent(
                room_id="test_room",
                sensor_id="binary_sensor.test",
                sensor_type="motion",
                state="on",
                previous_state="off",
                timestamp=base_time + timedelta(minutes=i * 10)
            )
            for i in range(3)
        ]
        
        # Add one event with no state change
        events.append(SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="motion",
            state="on",
            previous_state="on",  # Same state
            timestamp=base_time + timedelta(minutes=30)
        ))
        
        for event in events:
            test_db_session.add(event)
        await test_db_session.commit()
        
        # Get state changes
        state_changes = await SensorEvent.get_state_changes(
            test_db_session,
            "test_room",
            start_time=base_time - timedelta(minutes=30),
            end_time=base_time + timedelta(hours=1)
        )
        
        # Should only get events where state != previous_state
        assert len(state_changes) == 3
        for event in state_changes:
            assert event.state != event.previous_state
    
    @pytest.mark.asyncio
    async def test_get_transition_sequences(self, test_db_session):
        """Test getting transition sequences for pattern analysis."""
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        # Create a sequence of events within time window
        sequence1_events = []
        for i in range(5):
            event = SensorEvent(
                room_id="test_room",
                sensor_id=f"binary_sensor.sensor_{i % 2}",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(minutes=i * 2)
            )
            sequence1_events.append(event)
            test_db_session.add(event)
        
        # Create another sequence after a gap
        sequence2_events = []
        for i in range(3):
            event = SensorEvent(
                room_id="test_room",
                sensor_id="binary_sensor.sensor_2",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(minutes=40 + i * 2)
            )
            sequence2_events.append(event)
            test_db_session.add(event)
        
        await test_db_session.commit()
        
        # Get transition sequences
        sequences = await SensorEvent.get_transition_sequences(
            test_db_session,
            "test_room",
            lookback_hours=2,
            min_sequence_length=3
        )
        
        # Should get 2 sequences
        assert len(sequences) == 2
        assert len(sequences[0]) >= 3
        assert len(sequences[1]) >= 3


class TestRoomState:
    """Test RoomState model."""
    
    def test_room_state_creation(self):
        """Test creating a room state with all fields."""
        timestamp = datetime.utcnow()
        certainty_factors = {"sensor_confidence": 0.9, "pattern_match": 0.8}
        
        room_state = RoomState(
            room_id="kitchen",
            timestamp=timestamp,
            is_occupied=True,
            occupancy_confidence=0.85,
            occupant_type="human",
            occupant_count=2,
            state_duration=300,
            transition_trigger="binary_sensor.kitchen_motion",
            certainty_factors=certainty_factors,
            created_at=timestamp
        )
        
        assert room_state.room_id == "kitchen"
        assert room_state.timestamp == timestamp
        assert room_state.is_occupied is True
        assert room_state.occupancy_confidence == 0.85
        assert room_state.occupant_type == "human"
        assert room_state.occupant_count == 2
        assert room_state.state_duration == 300
        assert room_state.transition_trigger == "binary_sensor.kitchen_motion"
        assert room_state.certainty_factors == certainty_factors
    
    @pytest.mark.asyncio
    async def test_get_current_state(self, test_db_session):
        """Test getting current room state."""
        base_time = datetime.utcnow()
        
        # Create multiple room states
        states = [
            RoomState(
                room_id="office",
                timestamp=base_time - timedelta(hours=2),
                is_occupied=False,
                occupancy_confidence=0.7
            ),
            RoomState(
                room_id="office",
                timestamp=base_time - timedelta(hours=1),
                is_occupied=True,
                occupancy_confidence=0.9
            ),
            RoomState(
                room_id="office",
                timestamp=base_time,  # Most recent
                is_occupied=False,
                occupancy_confidence=0.8
            )
        ]
        
        for state in states:
            test_db_session.add(state)
        await test_db_session.commit()
        
        # Get current state
        current_state = await RoomState.get_current_state(test_db_session, "office")
        
        assert current_state is not None
        assert current_state.timestamp == base_time
        assert current_state.is_occupied is False
        assert current_state.occupancy_confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_get_current_state_not_found(self, test_db_session):
        """Test getting current state for non-existent room."""
        current_state = await RoomState.get_current_state(test_db_session, "nonexistent_room")
        assert current_state is None
    
    @pytest.mark.asyncio
    async def test_get_occupancy_history(self, test_db_session):
        """Test getting occupancy history."""
        base_time = datetime.utcnow()
        
        # Create occupancy history over 25 hours
        states = []
        for i in range(26):  # 26 hours of data
            state = RoomState(
                room_id="bedroom",
                timestamp=base_time - timedelta(hours=25 - i),
                is_occupied=i % 3 == 0,  # Varies occupancy
                occupancy_confidence=0.7 + (i % 3) * 0.1
            )
            states.append(state)
            test_db_session.add(state)
        
        await test_db_session.commit()
        
        # Get last 24 hours
        history = await RoomState.get_occupancy_history(test_db_session, "bedroom", hours=24)
        
        # Should get states from last 24 hours (not all 26)
        assert len(history) <= 25  # 24 hours + current
        assert len(history) > 20   # Should have most of them
        
        # Should be ordered by timestamp
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i-1].timestamp


class TestPrediction:
    """Test Prediction model."""
    
    def test_prediction_creation(self):
        """Test creating a prediction with all fields."""
        prediction_time = datetime.utcnow()
        transition_time = prediction_time + timedelta(minutes=15)
        feature_importance = {"temperature": 0.3, "motion": 0.7}
        alternatives = [
            {"time": transition_time + timedelta(minutes=5), "confidence": 0.6},
            {"time": transition_time + timedelta(minutes=-5), "confidence": 0.4}
        ]
        
        prediction = Prediction(
            room_id="bathroom",
            prediction_time=prediction_time,
            predicted_transition_time=transition_time,
            transition_type="vacant_to_occupied",
            confidence_score=0.85,
            prediction_interval_lower=transition_time - timedelta(minutes=5),
            prediction_interval_upper=transition_time + timedelta(minutes=10),
            model_type="lstm",
            model_version="v2.1",
            feature_importance=feature_importance,
            alternatives=alternatives,
            created_at=prediction_time
        )
        
        assert prediction.room_id == "bathroom"
        assert prediction.prediction_time == prediction_time
        assert prediction.predicted_transition_time == transition_time
        assert prediction.transition_type == "vacant_to_occupied"
        assert prediction.confidence_score == 0.85
        assert prediction.model_type == "lstm"
        assert prediction.model_version == "v2.1"
        assert prediction.feature_importance == feature_importance
        assert prediction.alternatives == alternatives
    
    @pytest.mark.asyncio
    async def test_get_pending_validations(self, test_db_session):
        """Test getting predictions that need validation."""
        base_time = datetime.utcnow()
        
        # Create predictions at different times
        predictions = [
            # Should be included (prediction time passed, no validation)
            Prediction(
                room_id="living_room",
                prediction_time=base_time - timedelta(hours=3),
                predicted_transition_time=base_time - timedelta(minutes=30),
                transition_type="occupied_to_vacant",
                confidence_score=0.8,
                model_type="xgboost",
                model_version="v1.0",
                actual_transition_time=None  # Not validated yet
            ),
            # Should be excluded (already validated)
            Prediction(
                room_id="living_room",
                prediction_time=base_time - timedelta(hours=2),
                predicted_transition_time=base_time - timedelta(minutes=15),
                transition_type="vacant_to_occupied",
                confidence_score=0.7,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(minutes=10)  # Already validated
            ),
            # Should be excluded (prediction time not reached yet)
            Prediction(
                room_id="living_room",
                prediction_time=base_time,
                predicted_transition_time=base_time + timedelta(minutes=30),
                transition_type="occupied_to_vacant",
                confidence_score=0.9,
                model_type="ensemble",
                model_version="v1.0",
                actual_transition_time=None
            ),
            # Should be excluded (too old)
            Prediction(
                room_id="living_room",
                prediction_time=base_time - timedelta(hours=5),
                predicted_transition_time=base_time - timedelta(hours=4),
                transition_type="vacant_to_occupied",
                confidence_score=0.6,
                model_type="hmm",
                model_version="v1.0",
                actual_transition_time=None
            )
        ]
        
        for pred in predictions:
            test_db_session.add(pred)
        await test_db_session.commit()
        
        # Get pending validations
        pending = await Prediction.get_pending_validations(
            test_db_session,
            room_id="living_room",
            cutoff_hours=2
        )
        
        # Should only get the first prediction
        assert len(pending) == 1
        assert pending[0].model_type == "xgboost"
        assert pending[0].actual_transition_time is None
    
    @pytest.mark.asyncio
    async def test_get_accuracy_metrics(self, test_db_session):
        """Test calculating accuracy metrics."""
        base_time = datetime.utcnow()
        
        # Create predictions with accuracy data
        predictions = [
            # Accurate prediction (within 10 minutes)
            Prediction(
                room_id="office",
                prediction_time=base_time - timedelta(days=1),
                predicted_transition_time=base_time - timedelta(hours=23),
                transition_type="occupied_to_vacant",
                confidence_score=0.9,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(hours=23, minutes=5),
                accuracy_minutes=5.0,
                is_accurate=True,
                validation_timestamp=base_time - timedelta(hours=22)
            ),
            # Inaccurate prediction (30 minutes off)
            Prediction(
                room_id="office",
                prediction_time=base_time - timedelta(days=2),
                predicted_transition_time=base_time - timedelta(hours=47),
                transition_type="vacant_to_occupied",
                confidence_score=0.7,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(hours=46, minutes=30),
                accuracy_minutes=30.0,
                is_accurate=False,
                validation_timestamp=base_time - timedelta(hours=46)
            ),
            # Another accurate prediction
            Prediction(
                room_id="office",
                prediction_time=base_time - timedelta(days=3),
                predicted_transition_time=base_time - timedelta(hours=71),
                transition_type="occupied_to_vacant",
                confidence_score=0.8,
                model_type="lstm",
                model_version="v1.0",
                actual_transition_time=base_time - timedelta(hours=70, minutes=50),
                accuracy_minutes=10.0,
                is_accurate=True,
                validation_timestamp=base_time - timedelta(hours=70)
            )
        ]
        
        for pred in predictions:
            test_db_session.add(pred)
        await test_db_session.commit()
        
        # Get accuracy metrics
        metrics = await Prediction.get_accuracy_metrics(
            test_db_session,
            "office",
            days=7,
            model_type="lstm"
        )
        
        assert metrics["total_predictions"] == 3
        assert metrics["accurate_predictions"] == 2
        assert metrics["accuracy_rate"] == 2/3
        assert metrics["mean_error_minutes"] == (5.0 + 30.0 + 10.0) / 3
        assert abs(metrics["median_error_minutes"] - 10.0) < 0.1
        assert metrics["rmse_minutes"] > 0


class TestModelAccuracy:
    """Test ModelAccuracy model."""
    
    def test_model_accuracy_creation(self):
        """Test creating model accuracy record."""
        start_time = datetime.utcnow() - timedelta(days=1)
        end_time = datetime.utcnow()
        baseline_comparison = {"previous_accuracy": 0.75, "improvement": 0.05}
        
        accuracy = ModelAccuracy(
            room_id="guest_room",
            model_type="xgboost",
            model_version="v1.2",
            measurement_start=start_time,
            measurement_end=end_time,
            total_predictions=100,
            accurate_predictions=85,
            accuracy_rate=0.85,
            mean_error_minutes=12.5,
            median_error_minutes=8.0,
            rmse_minutes=15.2,
            confidence_correlation=0.7,
            overconfidence_rate=0.15,
            feature_drift_score=0.05,
            concept_drift_score=0.02,
            performance_degradation=-0.03,
            baseline_comparison=baseline_comparison
        )
        
        assert accuracy.room_id == "guest_room"
        assert accuracy.model_type == "xgboost"
        assert accuracy.model_version == "v1.2"
        assert accuracy.total_predictions == 100
        assert accuracy.accurate_predictions == 85
        assert accuracy.accuracy_rate == 0.85
        assert accuracy.mean_error_minutes == 12.5
        assert accuracy.median_error_minutes == 8.0
        assert accuracy.rmse_minutes == 15.2
        assert accuracy.confidence_correlation == 0.7
        assert accuracy.overconfidence_rate == 0.15
        assert accuracy.feature_drift_score == 0.05
        assert accuracy.concept_drift_score == 0.02
        assert accuracy.performance_degradation == -0.03
        assert accuracy.baseline_comparison == baseline_comparison


class TestFeatureStore:
    """Test FeatureStore model."""
    
    def test_feature_store_creation(self):
        """Test creating feature store record."""
        timestamp = datetime.utcnow()
        temporal_features = {
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "is_weekend": False,
            "time_since_last_change": 300
        }
        sequential_features = {
            "room_transition_1gram": "living_room",
            "movement_velocity": 1.2,
            "trigger_sequence_pattern": "motion->door->motion"
        }
        contextual_features = {
            "temperature": 22.5,
            "humidity": 45.0,
            "light_level": 250,
            "other_rooms_occupied": 2
        }
        environmental_features = {
            "weather": "sunny",
            "outdoor_temperature": 18.0
        }
        
        feature_store = FeatureStore(
            room_id="study",
            feature_timestamp=timestamp,
            temporal_features=temporal_features,
            sequential_features=sequential_features,
            contextual_features=contextual_features,
            environmental_features=environmental_features,
            lookback_hours=24,
            feature_version="v1.3",
            computation_time_ms=150.5,
            completeness_score=0.95,
            freshness_score=0.9,
            confidence_score=0.85,
            expires_at=timestamp + timedelta(hours=6)
        )
        
        assert feature_store.room_id == "study"
        assert feature_store.feature_timestamp == timestamp
        assert feature_store.temporal_features == temporal_features
        assert feature_store.sequential_features == sequential_features
        assert feature_store.contextual_features == contextual_features
        assert feature_store.environmental_features == environmental_features
        assert feature_store.lookback_hours == 24
        assert feature_store.feature_version == "v1.3"
        assert feature_store.computation_time_ms == 150.5
        assert feature_store.completeness_score == 0.95
        assert feature_store.freshness_score == 0.9
        assert feature_store.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_get_latest_features(self, test_db_session):
        """Test getting latest features for a room."""
        base_time = datetime.utcnow()
        
        # Create feature records at different times
        features = [
            # Older features
            FeatureStore(
                room_id="kitchen",
                feature_timestamp=base_time - timedelta(hours=8),
                temporal_features={"hour_sin": 0.0},
                sequential_features={},
                contextual_features={},
                environmental_features={},
                lookback_hours=24,
                feature_version="v1.0",
                expires_at=base_time - timedelta(hours=2)  # Expired
            ),
            # Recent features (not expired)
            FeatureStore(
                room_id="kitchen",
                feature_timestamp=base_time - timedelta(hours=2),
                temporal_features={"hour_sin": 0.5},
                sequential_features={},
                contextual_features={},
                environmental_features={},
                lookback_hours=24,
                feature_version="v1.1",
                expires_at=base_time + timedelta(hours=4)  # Not expired
            ),
            # Most recent features
            FeatureStore(
                room_id="kitchen",
                feature_timestamp=base_time - timedelta(minutes=30),
                temporal_features={"hour_sin": 0.866},
                sequential_features={},
                contextual_features={},
                environmental_features={},
                lookback_hours=24,
                feature_version="v1.2",
                expires_at=base_time + timedelta(hours=6)  # Not expired
            )
        ]
        
        for feature in features:
            test_db_session.add(feature)
        await test_db_session.commit()
        
        # Get latest features
        latest = await FeatureStore.get_latest_features(
            test_db_session,
            "kitchen",
            max_age_hours=6
        )
        
        assert latest is not None
        assert latest.feature_version == "v1.2"
        assert latest.temporal_features["hour_sin"] == 0.866
    
    def test_get_all_features(self):
        """Test combining all feature categories."""
        temporal_features = {"hour_sin": 0.5, "is_weekend": False}
        sequential_features = {"velocity": 1.2}
        contextual_features = {"temperature": 22.5, "humidity": 45.0}
        environmental_features = {"weather": "sunny"}
        
        feature_store = FeatureStore(
            room_id="test_room",
            feature_timestamp=datetime.utcnow(),
            temporal_features=temporal_features,
            sequential_features=sequential_features,
            contextual_features=contextual_features,
            environmental_features=environmental_features,
            lookback_hours=24,
            feature_version="v1.0"
        )
        
        all_features = feature_store.get_all_features()
        
        # Should combine all feature categories
        assert all_features["hour_sin"] == 0.5
        assert all_features["is_weekend"] is False
        assert all_features["velocity"] == 1.2
        assert all_features["temperature"] == 22.5
        assert all_features["humidity"] == 45.0
        assert all_features["weather"] == "sunny"
        
        # Should have all features from all categories
        expected_count = (
            len(temporal_features) + 
            len(sequential_features) + 
            len(contextual_features) + 
            len(environmental_features)
        )
        assert len(all_features) == expected_count


class TestModelRelationships:
    """Test relationships between models."""
    
    @pytest.mark.asyncio
    async def test_sensor_event_prediction_relationship(self, test_db_session):
        """Test relationship between SensorEvent and Prediction."""
        # Create sensor event
        event = SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.utcnow()
        )
        test_db_session.add(event)
        await test_db_session.flush()  # Get the ID
        
        # Create prediction triggered by this event
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.utcnow(),
            predicted_transition_time=datetime.utcnow() + timedelta(minutes=15),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            triggering_event_id=event.id
        )
        test_db_session.add(prediction)
        await test_db_session.commit()
        
        # Test relationship
        await test_db_session.refresh(prediction)
        assert prediction.triggering_event.id == event.id
        assert prediction.triggering_event.sensor_id == "binary_sensor.test"
    
    @pytest.mark.asyncio
    async def test_room_state_prediction_relationship(self, test_db_session):
        """Test relationship between RoomState and Prediction."""
        # Create room state
        room_state = RoomState(
            room_id="test_room",
            timestamp=datetime.utcnow(),
            is_occupied=True,
            occupancy_confidence=0.9
        )
        test_db_session.add(room_state)
        await test_db_session.flush()  # Get the ID
        
        # Create prediction based on this room state
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.utcnow(),
            predicted_transition_time=datetime.utcnow() + timedelta(minutes=20),
            transition_type="occupied_to_vacant",
            confidence_score=0.75,
            model_type="xgboost",
            model_version="v1.0",
            room_state_id=room_state.id
        )
        test_db_session.add(prediction)
        await test_db_session.commit()
        
        # Test relationship
        await test_db_session.refresh(prediction)
        assert prediction.room_state.id == room_state.id
        assert prediction.room_state.is_occupied is True


class TestModelConstraints:
    """Test model constraints and validation."""
    
    @pytest.mark.asyncio
    async def test_confidence_score_constraints(self, test_db_session):
        """Test confidence score constraints."""
        # Valid confidence score
        event = SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="motion",
            state="on",
            timestamp=datetime.utcnow(),
            confidence_score=0.8
        )
        test_db_session.add(event)
        
        try:
            await test_db_session.commit()
            # Should succeed
        except IntegrityError:
            pytest.fail("Valid confidence score should not raise IntegrityError")
    
    @pytest.mark.asyncio
    async def test_model_accuracy_constraints(self, test_db_session):
        """Test model accuracy constraints."""
        # Valid accuracy data
        accuracy = ModelAccuracy(
            room_id="test_room",
            model_type="lstm",
            model_version="v1.0",
            measurement_start=datetime.utcnow() - timedelta(days=1),
            measurement_end=datetime.utcnow(),
            total_predictions=100,
            accurate_predictions=85,  # Less than total
            accuracy_rate=0.85,
            mean_error_minutes=10.0,
            median_error_minutes=8.0,
            rmse_minutes=12.0
        )
        test_db_session.add(accuracy)
        
        try:
            await test_db_session.commit()
            # Should succeed
        except IntegrityError:
            pytest.fail("Valid accuracy data should not raise IntegrityError")


class TestUtilityFunctions:
    """Test utility functions for database operations."""
    
    @pytest.mark.asyncio
    async def test_create_timescale_hypertables(self, test_db_session):
        """Test TimescaleDB hypertable creation."""
        with patch.object(test_db_session, 'execute') as mock_execute:
            with patch.object(test_db_session, 'commit') as mock_commit:
                await create_timescale_hypertables(test_db_session)
                
                # Should execute TimescaleDB-specific commands
                assert mock_execute.call_count > 0
                mock_commit.assert_called_once()
                
                # Check that hypertable creation was called
                calls = [call.args[0].text for call in mock_execute.call_args_list if hasattr(call.args[0], 'text')]
                hypertable_calls = [call for call in calls if 'create_hypertable' in call]
                assert len(hypertable_calls) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_database_performance(self, test_db_session):
        """Test database performance optimization."""
        with patch.object(test_db_session, 'execute') as mock_execute:
            with patch.object(test_db_session, 'commit') as mock_commit:
                await optimize_database_performance(test_db_session)
                
                # Should execute optimization commands
                assert mock_execute.call_count > 0
                mock_commit.assert_called_once()
                
                # Check that ANALYZE commands were called
                calls = [call.args[0].text for call in mock_execute.call_args_list if hasattr(call.args[0], 'text')]
                analyze_calls = [call for call in calls if 'ANALYZE' in call]
                assert len(analyze_calls) > 0
    
    def test_get_bulk_insert_query(self):
        """Test bulk insert query generation."""
        query = get_bulk_insert_query()
        
        assert isinstance(query, str)
        assert "INSERT INTO sensor_events" in query
        assert "VALUES %s" in query
        assert "ON CONFLICT" in query
        
        # Should include all important columns
        expected_columns = [
            "timestamp", "room_id", "sensor_id", "sensor_type", 
            "state", "previous_state", "attributes", "is_human_triggered"
        ]
        for column in expected_columns:
            assert column in query


@pytest.mark.unit
@pytest.mark.database
class TestModelIntegration:
    """Integration tests for model interactions."""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_workflow(self, test_db_session):
        """Test complete workflow from sensor event to prediction validation."""
        # 1. Create initial sensor event
        trigger_event = SensorEvent(
            room_id="integration_test_room",
            sensor_id="binary_sensor.test_motion",
            sensor_type="motion",
            state="on",
            previous_state="off",
            timestamp=datetime.utcnow() - timedelta(minutes=30),
            is_human_triggered=True,
            confidence_score=0.85
        )
        test_db_session.add(trigger_event)
        await test_db_session.flush()
        
        # 2. Create room state based on event
        room_state = RoomState(
            room_id="integration_test_room",
            timestamp=datetime.utcnow() - timedelta(minutes=25),
            is_occupied=True,
            occupancy_confidence=0.9,
            transition_trigger=trigger_event.sensor_id
        )
        test_db_session.add(room_state)
        await test_db_session.flush()
        
        # 3. Create prediction based on room state
        prediction_time = datetime.utcnow() - timedelta(minutes=20)
        predicted_transition = datetime.utcnow() - timedelta(minutes=5)
        
        prediction = Prediction(
            room_id="integration_test_room",
            prediction_time=prediction_time,
            predicted_transition_time=predicted_transition,
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
            model_type="lstm",
            model_version="v1.0",
            triggering_event_id=trigger_event.id,
            room_state_id=room_state.id
        )
        test_db_session.add(prediction)
        await test_db_session.flush()
        
        # 4. Create actual transition event (for validation)
        actual_event = SensorEvent(
            room_id="integration_test_room",
            sensor_id="binary_sensor.test_motion",
            sensor_type="motion",
            state="off",
            previous_state="on",
            timestamp=datetime.utcnow(),  # Actual transition happened now
            is_human_triggered=True,
            confidence_score=0.9
        )
        test_db_session.add(actual_event)
        
        # 5. Update prediction with validation results
        prediction.actual_transition_time = actual_event.timestamp
        prediction.accuracy_minutes = abs((prediction.predicted_transition_time - actual_event.timestamp).total_seconds() / 60)
        prediction.is_accurate = prediction.accuracy_minutes <= 15  # Within threshold
        prediction.validation_timestamp = datetime.utcnow()
        
        await test_db_session.commit()
        
        # Verify the complete workflow
        await test_db_session.refresh(prediction)
        
        # Check relationships work
        assert prediction.triggering_event.id == trigger_event.id
        assert prediction.room_state.id == room_state.id
        
        # Check prediction was validated
        assert prediction.actual_transition_time is not None
        assert prediction.accuracy_minutes is not None
        assert prediction.is_accurate is not None
        assert prediction.validation_timestamp is not None
        
        # Check accuracy calculation
        expected_accuracy = abs((predicted_transition - actual_event.timestamp).total_seconds() / 60)
        assert abs(prediction.accuracy_minutes - expected_accuracy) < 0.1
    
    @pytest.mark.asyncio 
    async def test_feature_store_lifecycle(self, test_db_session):
        """Test feature store lifecycle with expiration."""
        base_time = datetime.utcnow()
        
        # Create feature store entries with different expiration times
        fresh_features = FeatureStore(
            room_id="lifecycle_test_room",
            feature_timestamp=base_time - timedelta(minutes=30),
            temporal_features={"hour_sin": 0.5},
            sequential_features={},
            contextual_features={},
            environmental_features={},
            lookback_hours=24,
            feature_version="v1.0",
            completeness_score=0.95,
            freshness_score=0.9,
            confidence_score=0.85,
            expires_at=base_time + timedelta(hours=2)  # Fresh
        )
        
        expired_features = FeatureStore(
            room_id="lifecycle_test_room",
            feature_timestamp=base_time - timedelta(hours=4),
            temporal_features={"hour_sin": 0.0},
            sequential_features={},
            contextual_features={},
            environmental_features={},
            lookback_hours=24,
            feature_version="v0.9",
            completeness_score=0.8,
            freshness_score=0.6,
            confidence_score=0.7,
            expires_at=base_time - timedelta(minutes=30)  # Expired
        )
        
        test_db_session.add(fresh_features)
        test_db_session.add(expired_features)
        await test_db_session.commit()
        
        # Get latest features - should only return fresh ones
        latest = await FeatureStore.get_latest_features(
            test_db_session,
            "lifecycle_test_room",
            max_age_hours=6
        )
        
        assert latest is not None
        assert latest.feature_version == "v1.0"
        assert latest.expires_at > base_time  # Not expired
        
        # Test feature combination
        all_features = latest.get_all_features()
        assert "hour_sin" in all_features
        assert all_features["hour_sin"] == 0.5