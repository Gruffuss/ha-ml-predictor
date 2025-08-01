# Room Occupancy Prediction System - Detailed Architecture

## System Components

### 1. Data Ingestion Layer
```
┌─────────────────────────────────────────────────────────┐
│                  Home Assistant API                       │
│  ├── Historical Data Bulk Import (6 months)              │
│  ├── Real-time Event Stream (WebSocket)                  │
│  └── State Change Events                                 │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Event Processing Pipeline                    │
│  ├── Event Validation & Deduplication                   │
│  ├── Human/Cat Movement Pattern Detector                │
│  └── Event Enrichment (timestamps, sequences)           │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Temporal Event Store                     │
│  ├── PostgreSQL with TimescaleDB                        │
│  ├── Partitioned by room and time                       │
│  └── Indexed for rapid sequence queries                 │
└─────────────────────────────────────────────────────────┘
```

### 2. Feature Engineering Engine
```
┌─────────────────────────────────────────────────────────┐
│              Feature Extraction Pipeline                  │
├─────────────────────────────────────────────────────────┤
│  Temporal Features:                                      │
│  ├── Time since last occupancy                          │
│  ├── Duration of current state                          │
│  ├── Hour/Day/Week cyclical encodings                   │
│  └── Holiday/Special day indicators                     │
├─────────────────────────────────────────────────────────┤
│  Sequential Features:                                    │
│  ├── Room transition sequences (n-grams)                │
│  ├── Movement velocity patterns                         │
│  ├── Sensor triggering order                            │
│  └── Cross-room correlation patterns                    │
├─────────────────────────────────────────────────────────┤
│  Contextual Features:                                    │
│  ├── Environmental (temp, humidity, light)              │
│  ├── Door state sequences                               │
│  ├── Multi-room occupancy states                        │
│  └── Historical pattern similarity scores               │
└─────────────────────────────────────────────────────────┘
```

### 3. Prediction Engine
```
┌─────────────────────────────────────────────────────────┐
│              Ensemble Model Architecture                  │
├─────────────────────────────────────────────────────────┤
│  Base Models:                                            │
│  ├── LSTM Networks (sequence patterns)                  │
│  ├── XGBoost (temporal features)                        │
│  ├── Hidden Markov Models (state transitions)           │
│  └── Gaussian Process (uncertainty quantification)      │
├─────────────────────────────────────────────────────────┤
│  Meta-Learner:                                           │
│  ├── Stacking ensemble with time-aware CV               │
│  ├── Per-room model specialization                      │
│  └── Confidence calibration layer                       │
├─────────────────────────────────────────────────────────┤
│  Outputs per room:                                      │
│  ├── Next occupancy time (if vacant)                    │
│  ├── Next vacancy time (if occupied)                    │
│  ├── Confidence intervals                               │
│  └── Alternative scenarios (top-3 predictions)          │
└─────────────────────────────────────────────────────────┘
```

### 4. Self-Adaptation System
```
┌─────────────────────────────────────────────────────────┐
│              Continuous Learning Pipeline                 │
├─────────────────────────────────────────────────────────┤
│  Prediction Validator:                                   │
│  ├── Tracks prediction accuracy in real-time            │
│  ├── Calculates time deltas (predicted vs actual)       │
│  └── Maintains accuracy metrics per room/time           │
├─────────────────────────────────────────────────────────┤
│  Adaptation Triggers:                                    │
│  ├── Accuracy degradation (>15 min avg error)           │
│  ├── New pattern detection                              │
│  ├── Scheduled daily micro-updates                      │
│  └── Concept drift detection                            │
├─────────────────────────────────────────────────────────┤
│  Update Strategies:                                      │
│  ├── Online learning for base models                    │
│  ├── Incremental training batches                       │
│  ├── Feature importance re-weighting                    │
│  └── Ensemble weight optimization                       │
└─────────────────────────────────────────────────────────┘
```

### 5. Integration & API Layer
```
┌─────────────────────────────────────────────────────────┐
│                    MQTT Publisher                         │
├─────────────────────────────────────────────────────────┤
│  Topic Structure:                                        │
│  ├── occupancy/predictions/{room}/next_occupied_time    │
│  ├── occupancy/predictions/{room}/next_vacant_time      │
│  ├── occupancy/predictions/{room}/confidence            │
│  └── occupancy/predictions/{room}/last_updated          │
├─────────────────────────────────────────────────────────┤
│  Payload Format:                                         │
│  {                                                       │
│    "predicted_time": "2024-01-15T14:30:00",            │
│    "confidence": 0.85,                                   │
│    "time_until": "25 minutes",                          │
│    "alternatives": [...]                                 │
│  }                                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 REST API (Optional)                       │
├─────────────────────────────────────────────────────────┤
│  GET /api/predictions/{room}                             │
│  GET /api/model/accuracy                                 │
│  GET /api/health                                         │
│  POST /api/model/retrain (manual trigger)               │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Diagram
```
Sensors → HA → Event Stream → Feature Pipeline → Models → Predictions → MQTT → HA
    ↑                              ↓                ↓           ↓
    └──────── Validation ←─────────┴───────────────┴───────────┘
```