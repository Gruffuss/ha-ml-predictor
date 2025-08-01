# Occupancy Prediction System - Implementation Plan

## Directory Structure

```
occupancy-prediction/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   ├── rooms.yaml
│   ├── config.yaml
│   └── logging.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── exceptions.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── ha_client.py
│   │   │   ├── event_processor.py
│   │   │   └── bulk_importer.py
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── database.py
│   │   │   └── migrations/
│   │   └── validation/
│   │       ├── __init__.py
│   │       ├── event_validator.py
│   │       └── pattern_detector.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   ├── sequential.py
│   │   ├── contextual.py
│   │   ├── engineering.py
│   │   └── store.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_predictor.py
│   │   │   ├── xgboost_predictor.py
│   │   │   ├── hmm_predictor.py
│   │   │   └── gp_predictor.py
│   │   ├── ensemble.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   ├── adaptation/
│   │   ├── __init__.py
│   │   ├── validator.py
│   │   ├── drift_detector.py
│   │   ├── retrainer.py
│   │   └── optimizer.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── mqtt_publisher.py
│   │   ├── api_server.py
│   │   └── ha_entities.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── time_utils.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   ├── setup_database.py
│   ├── initial_training.py
│   └── validate_predictions.py
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

## Sprint Structure

### Sprint 1: Foundation & Data Infrastructure

**Objectives:**
- Set up project structure and development environment
- Implement Home Assistant integration
- Create database schema and event storage

**Deliverables:**

#### 1.1 Core Configuration System
```python
# src/core/config.py
from dataclasses import dataclass
from typing import Dict, List, Any
import yaml

@dataclass
class RoomConfig:
    name: str
    room_id: str
    sensors: Dict[str, Any]
    
@dataclass
class SystemConfig:
    ha_url: str
    ha_token: str
    mqtt_broker: str
    mqtt_port: int
    db_connection: str
    prediction_interval: int = 300  # 5 minutes
    rooms: Dict[str, RoomConfig]
```

#### 1.2 Data Ingestion Logic
```python
# src/data/ingestion/ha_client.py
class HomeAssistantClient:
    """Handles all communication with Home Assistant"""
    
    async def connect(self):
        """Establish WebSocket connection for real-time events"""
        
    async def fetch_historical_data(self, entity_id: str, 
                                   start_date: datetime, 
                                   end_date: datetime):
        """Bulk fetch historical state changes"""
        
    async def subscribe_to_events(self, entity_ids: List[str]):
        """Subscribe to real-time state change events"""
        
    def process_event(self, event: Dict[str, Any]) -> Event:
        """Convert HA event to internal Event model"""
```

#### 1.3 Event Storage Schema
```sql
-- TimescaleDB schema
CREATE TABLE sensor_events (
    id BIGSERIAL,
    room_id VARCHAR(50) NOT NULL,
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    state VARCHAR(10) NOT NULL,
    previous_state VARCHAR(10),
    timestamp TIMESTAMPTZ NOT NULL,
    attributes JSONB,
    is_human_triggered BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_events', 'timestamp');

-- Indexes for efficient querying
CREATE INDEX idx_room_sensor_time ON sensor_events (room_id, sensor_id, timestamp DESC);
CREATE INDEX idx_room_state_changes ON sensor_events (room_id, timestamp DESC) 
    WHERE state != previous_state;
```

### Sprint 2: Feature Engineering Pipeline

**Objectives:**
- Implement comprehensive feature extraction
- Create feature store for efficient model training
- Build human/cat movement pattern detector

**Deliverables:**

#### 2.1 Temporal Feature Engineering
```python
# src/features/temporal.py
class TemporalFeatureExtractor:
    """Extract time-based features from event sequences"""
    
    def extract_features(self, events: List[Event], 
                        target_time: datetime) -> Dict[str, float]:
        features = {}
        
        # Time since last state change
        features['time_since_last_change'] = self._time_since_last_change(events)
        
        # Duration in current state
        features['current_state_duration'] = self._current_state_duration(events)
        
        # Cyclical time encodings
        features.update(self._cyclical_time_features(target_time))
        
        # Historical patterns at this time
        features.update(self._historical_time_patterns(events, target_time))
        
        return features
```

#### 2.2 Sequential Pattern Features
```python
# src/features/sequential.py
class SequentialFeatureExtractor:
    """Extract movement and transition patterns"""
    
    def extract_features(self, events: List[Event], 
                        room_configs: Dict[str, RoomConfig]) -> Dict[str, float]:
        # Room transition sequences (n-grams)
        transitions = self._extract_room_transitions(events)
        
        # Movement velocity through sensors
        velocity_features = self._calculate_movement_velocity(events)
        
        # Sensor triggering patterns
        trigger_patterns = self._analyze_trigger_sequences(events)
        
        # Human vs cat classification
        movement_class = self._classify_movement_pattern(events, room_configs)
        
        return {**transitions, **velocity_features, 
                **trigger_patterns, **movement_class}
```

#### 2.3 Feature Store Implementation
```python
# src/features/store.py
class FeatureStore:
    """Manages feature computation and caching"""
    
    def compute_features(self, room_id: str, 
                        target_time: datetime,
                        lookback_hours: int = 24) -> pd.DataFrame:
        """Compute all features for a prediction target"""
        
    def get_training_data(self, start_date: datetime,
                         end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate feature matrix and targets for training"""
```

### Sprint 3: Model Development & Training

**Objectives:**
- Implement base predictive models
- Create ensemble architecture
- Build initial training pipeline

**Deliverables:**

#### 3.1 Base Model Implementations
```python
# src/models/base/lstm_predictor.py
class LSTMPredictor(BasePredictor):
    """LSTM for sequence-based predictions"""
    
    def __init__(self, input_dim: int, sequence_length: int):
        self.model = self._build_model(input_dim, sequence_length)
        
    def _build_model(self, input_dim: int, sequence_length: int):
        """Build LSTM architecture with attention mechanism"""
        
    def predict_transition_time(self, features: np.ndarray) -> Tuple[datetime, float]:
        """Predict next state transition time with confidence"""

# src/models/base/xgboost_predictor.py
class XGBoostPredictor(BasePredictor):
    """Gradient boosting for tabular features"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self._default_params()
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame, 
              sample_weights: np.ndarray = None):
        """Train with time-aware cross-validation"""
```

#### 3.2 Ensemble Architecture
```python
# src/models/ensemble.py
class OccupancyEnsemble:
    """Meta-learner combining multiple base models"""
    
    def __init__(self, base_models: List[BasePredictor]):
        self.base_models = base_models
        self.meta_learner = self._create_meta_learner()
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train ensemble with stacking"""
        # Generate base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Train meta-learner on base predictions
        self.meta_learner.fit(base_predictions, y)
        
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions with confidence intervals"""
        return {
            'predicted_time': predicted_time,
            'confidence': confidence,
            'prediction_interval': (lower_bound, upper_bound),
            'alternatives': top_k_predictions
        }
```

#### 3.3 Training Pipeline
```python
# src/models/trainer.py
class ModelTrainer:
    """Orchestrates model training process"""
    
    def train_initial_models(self, historical_months: int = 6):
        """Initial training on historical data"""
        
    def train_room_specific_model(self, room_id: str, 
                                 features: pd.DataFrame,
                                 targets: pd.DataFrame):
        """Train specialized model for each room"""
        
    def validate_predictions(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate accuracy metrics"""
```

### Sprint 4: Self-Adaptation System

**Objectives:**
- Implement real-time prediction validation
- Create adaptive retraining mechanisms
- Build drift detection system

**Deliverables:**

#### 4.1 Prediction Validator
```python
# src/adaptation/validator.py
class PredictionValidator:
    """Tracks prediction accuracy in real-time"""
    
    def __init__(self, accuracy_threshold: int = 15):  # minutes
        self.accuracy_threshold = accuracy_threshold
        self.prediction_history = defaultdict(list)
        
    def record_prediction(self, room_id: str, 
                         predicted_time: datetime,
                         confidence: float):
        """Store prediction for later validation"""
        
    def validate_prediction(self, room_id: str, 
                           actual_time: datetime):
        """Compare actual transition time with prediction"""
        
    def get_accuracy_metrics(self, room_id: str, 
                           window_hours: int = 24) -> Dict[str, float]:
        """Calculate recent prediction accuracy"""
```

#### 4.2 Adaptive Retraining
```python
# src/adaptation/retrainer.py
class AdaptiveRetrainer:
    """Manages continuous model updates"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.retraining_queue = PriorityQueue()
        
    async def monitor_and_retrain(self):
        """Background task for adaptive retraining"""
        while True:
            # Check for retraining triggers
            triggers = await self._check_triggers()
            
            if triggers:
                await self._schedule_retraining(triggers)
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    def _incremental_update(self, room_id: str, 
                           new_data: pd.DataFrame):
        """Perform online learning update"""
```

#### 4.3 Drift Detection
```python
# src/adaptation/drift_detector.py
class ConceptDriftDetector:
    """Detects changes in occupancy patterns"""
    
    def detect_drift(self, features: pd.DataFrame, 
                     predictions: pd.DataFrame,
                     actuals: pd.DataFrame) -> bool:
        """Use statistical tests to detect concept drift"""
        
    def identify_new_patterns(self, recent_events: List[Event]) -> List[Pattern]:
        """Detect emergence of new behavioral patterns"""
```

### Sprint 5: Integration & API Development

**Objectives:**
- Implement MQTT publisher for Home Assistant
- Create REST API endpoints
- Build HA sensor entity definitions

**Deliverables:**

#### 5.1 MQTT Integration
```python
# src/integration/mqtt_publisher.py
class MQTTPublisher:
    """Publishes predictions to Home Assistant via MQTT"""
    
    def __init__(self, broker: str, port: int):
        self.client = mqtt.Client()
        self.setup_client()
        
    async def publish_prediction(self, room_id: str, 
                               prediction: Dict[str, Any]):
        """Publish prediction to MQTT topics"""
        base_topic = f"occupancy/predictions/{room_id}"
        
        # Publish individual components
        await self._publish_json(
            f"{base_topic}/next_transition",
            {
                "predicted_time": prediction['predicted_time'].isoformat(),
                "transition_type": prediction['transition_type'],
                "confidence": prediction['confidence'],
                "time_until": self._format_time_delta(prediction['predicted_time']),
                "alternatives": prediction.get('alternatives', [])
            }
        )
        
    def create_ha_discovery_config(self, room_id: str) -> Dict[str, Any]:
        """Generate HA MQTT discovery configuration"""
```

#### 5.2 REST API Server
```python
# src/integration/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Occupancy Prediction API")

@app.get("/api/predictions/{room_id}")
async def get_room_prediction(room_id: str):
    """Get current predictions for a room"""
    
@app.get("/api/model/accuracy")
async def get_model_accuracy():
    """Get accuracy metrics for all models"""
    
@app.post("/api/model/retrain")
async def trigger_manual_retrain(room_id: Optional[str] = None):
    """Manually trigger model retraining"""
```

### Sprint 6: Testing & Validation

**Objectives:**
- Implement comprehensive unit tests
- Create integration tests
- Build prediction validation framework

**Deliverables:**

#### 6.1 Unit Test Suite
```python
# tests/unit/test_feature_engineering.py
class TestFeatureEngineering:
    """Test feature extraction logic"""
    
    def test_temporal_features(self, sample_events):
        """Test temporal feature extraction"""
        
    def test_movement_pattern_detection(self, human_events, cat_events):
        """Test human vs cat classification"""
        
    def test_sequence_features(self, transition_events):
        """Test room transition feature extraction"""

# tests/unit/test_models.py
class TestPredictionModels:
    """Test model predictions"""
    
    def test_prediction_format(self, trained_model, sample_features):
        """Ensure predictions have correct format"""
        
    def test_confidence_calibration(self, trained_model, test_data):
        """Test that confidence scores are well-calibrated"""
```

#### 6.2 Integration Tests
```python
# tests/integration/test_ha_integration.py
class TestHomeAssistantIntegration:
    """Test HA data flow"""
    
    async def test_event_subscription(self, ha_client):
        """Test real-time event subscription"""
        
    async def test_mqtt_publishing(self, mqtt_publisher):
        """Test MQTT message publishing"""
```

#### 6.3 Validation Framework
```python
# scripts/validate_predictions.py
class PredictionValidationReport:
    """Generate comprehensive validation reports"""
    
    def analyze_prediction_accuracy(self, days: int = 7):
        """Analyze recent prediction performance"""
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed accuracy report"""
```

### Sprint 7: Production Deployment

**Objectives:**
- Create deployment configuration
- Implement monitoring and logging
- Set up CI/CD pipeline

**Deliverables:**

#### 7.1 Docker Configuration
```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run setup
RUN python setup.py install

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "src.main"]
```

#### 7.2 Monitoring & Logging
```python
# src/utils/logger.py
class StructuredLogger:
    """Structured logging for production monitoring"""
    
    def __init__(self, service_name: str):
        self.logger = self._setup_logger(service_name)
        
    def log_prediction(self, room_id: str, prediction: Dict[str, Any]):
        """Log prediction with metadata for analysis"""
        
    def log_accuracy_metric(self, room_id: str, metrics: Dict[str, float]):
        """Log accuracy metrics for monitoring"""

# src/utils/metrics.py
class MetricsCollector:
    """Prometheus metrics for monitoring"""
    
    prediction_latency = Histogram('prediction_latency_seconds', 
                                  'Time to generate prediction')
    prediction_accuracy = Gauge('prediction_accuracy_minutes',
                               'Prediction accuracy in minutes',
                               ['room_id'])
    model_retrain_count = Counter('model_retrain_total',
                                 'Number of model retrains',
                                 ['room_id', 'trigger'])
```

## Coding Standards

### Python Style Guide
- Follow PEP 8 with 88-character line limit (Black formatter)
- Use type hints for all function signatures
- Docstrings for all public methods (Google style)
- Async/await for all I/O operations

### Code Organization
- One class per file for major components
- Group related functionality in modules
- Use dependency injection for testability
- Configuration through environment variables

### Error Handling
- Custom exceptions for domain-specific errors
- Graceful degradation for prediction failures
- Comprehensive error logging with context
- Circuit breakers for external services

### Performance Guidelines
- Batch database operations
- Use connection pooling
- Cache feature computations
- Profile memory usage regularly

## Deployment Process

1. **Initial Setup**
   ```bash
   # Clone repository
   git clone <repo-url>
   cd occupancy-prediction
   
   # Setup Python environment
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Setup database
   python scripts/setup_database.py
   
   # Import historical data
   python scripts/initial_training.py --months 6
   ```

2. **LXC Container Configuration**
   ```bash
   # Create container
   lxc-create -n occupancy-prediction -t download -- -d ubuntu -r 22.04
   
   # Configure resources
   lxc config set occupancy-prediction limits.cpu 2
   lxc config set occupancy-prediction limits.memory 6GB
   
   # Start and enter container
   lxc start occupancy-prediction
   lxc exec occupancy-prediction bash
   ```

3. **Production Startup**
   ```bash
   # Using Docker Compose
   docker-compose up -d
   
   # Or systemd service
   sudo systemctl start occupancy-prediction
   sudo systemctl enable occupancy-prediction
   ```

## Validation Criteria

### Accuracy Requirements
- Average prediction error < 15 minutes
- 90% of predictions within 20 minutes
- No systematic bias in predictions

### Performance Requirements
- Prediction generation < 100ms
- Feature computation < 500ms
- Model update < 5 minutes

### Reliability Requirements
- 99.9% uptime for prediction service
- Graceful handling of HA disconnections
- Automatic recovery from failures