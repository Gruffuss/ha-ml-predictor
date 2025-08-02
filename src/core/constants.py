"""
Constants used throughout the Occupancy Prediction System.
"""

from enum import Enum
from typing import Dict, List


class SensorType(Enum):
    """Types of sensors supported by the system."""
    PRESENCE = "presence"
    DOOR = "door"
    CLIMATE = "climate"
    LIGHT = "light"
    MOTION = "motion"


class SensorState(Enum):
    """Possible sensor states."""
    ON = "on"
    OFF = "off"
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class EventType(Enum):
    """Types of events in the system."""
    STATE_CHANGE = "state_change"
    PREDICTION = "prediction"
    MODEL_UPDATE = "model_update"
    ACCURACY_UPDATE = "accuracy_update"


class ModelType(Enum):
    """Types of ML models used."""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    HMM = "hmm"
    GAUSSIAN_PROCESS = "gp"
    ENSEMBLE = "ensemble"


class PredictionType(Enum):
    """Types of predictions."""
    NEXT_OCCUPIED = "next_occupied"
    NEXT_VACANT = "next_vacant"
    OCCUPANCY_DURATION = "occupancy_duration"
    VACANCY_DURATION = "vacancy_duration"


# Binary sensor states that indicate presence/occupancy
PRESENCE_STATES = [SensorState.ON.value]
ABSENCE_STATES = [SensorState.OFF.value]

# Door states that indicate room access
DOOR_OPEN_STATES = [SensorState.OPEN.value, SensorState.ON.value]
DOOR_CLOSED_STATES = [SensorState.CLOSED.value, SensorState.OFF.value]

# States that should be ignored/filtered out
INVALID_STATES = [SensorState.UNKNOWN.value, SensorState.UNAVAILABLE.value]

# Minimum time between events to be considered separate (seconds)
MIN_EVENT_SEPARATION = 5

# Maximum time gap to consider events as part of same sequence (seconds)
MAX_SEQUENCE_GAP = 300  # 5 minutes

# Default prediction confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Feature engineering constants
TEMPORAL_FEATURE_NAMES = [
    "time_since_last_change",
    "current_state_duration", 
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "week_sin", "week_cos",
    "is_weekend",
    "is_holiday"
]

SEQUENTIAL_FEATURE_NAMES = [
    "room_transition_1gram",
    "room_transition_2gram", 
    "room_transition_3gram",
    "movement_velocity",
    "trigger_sequence_pattern",
    "cross_room_correlation"
]

CONTEXTUAL_FEATURE_NAMES = [
    "temperature",
    "humidity",
    "light_level",
    "door_state",
    "other_rooms_occupied",
    "historical_pattern_similarity"
]

# MQTT topic structure
MQTT_TOPICS = {
    "predictions": "{topic_prefix}/{room_id}/prediction",
    "confidence": "{topic_prefix}/{room_id}/confidence", 
    "accuracy": "{topic_prefix}/{room_id}/accuracy",
    "status": "{topic_prefix}/system/status",
    "health": "{topic_prefix}/system/health"
}

# Database table names
DB_TABLES = {
    "sensor_events": "sensor_events",
    "predictions": "predictions",
    "model_accuracy": "model_accuracy",
    "room_states": "room_states",
    "feature_store": "feature_store"
}

# API endpoints
API_ENDPOINTS = {
    "predictions": "/api/predictions/{room_id}",
    "accuracy": "/api/model/accuracy",
    "health": "/api/health",
    "retrain": "/api/model/retrain",
    "rooms": "/api/rooms",
    "sensors": "/api/sensors"
}

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    ModelType.LSTM: {
        "sequence_length": 50,
        "hidden_units": 64,
        "dropout": 0.2,
        "learning_rate": 0.001
    },
    ModelType.XGBOOST: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8
    },
    ModelType.HMM: {
        "n_components": 4,
        "covariance_type": "full",
        "n_iter": 100
    },
    ModelType.GAUSSIAN_PROCESS: {
        "kernel": "rbf",
        "alpha": 1e-6,
        "n_restarts_optimizer": 0
    },
    ModelType.ENSEMBLE: {
        "meta_learner": "xgboost",
        "cv_folds": 5,
        "stacking_method": "linear",
        "blend_weights": "auto"
    }
}

# Human vs Cat movement patterns (for pattern detection)
HUMAN_MOVEMENT_PATTERNS = {
    "min_duration_seconds": 30,
    "max_velocity_ms": 2.0,
    "typical_room_sequence_length": 3,
    "door_interaction_probability": 0.8
}

CAT_MOVEMENT_PATTERNS = {
    "min_duration_seconds": 5,
    "max_velocity_ms": 5.0,
    "typical_room_sequence_length": 5,
    "door_interaction_probability": 0.1
}