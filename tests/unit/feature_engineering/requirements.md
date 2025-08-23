# Feature Engineering Testing Requirements

## Overview
This document contains detailed testing requirements for the ha-ml-predictor feature engineering components to achieve 85%+ test coverage. Each component has been analyzed for actual implementation details and specific testing scenarios.

### src/features/sequential.py - Sequential Pattern Features
**Classes Found:** SequentialFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_room_transition_features, _extract_velocity_features, _extract_sensor_sequence_features, _extract_cross_room_features, _extract_movement_classification_features, _extract_ngram_features, _create_sequences_for_classification, _create_movement_sequence, _get_default_features, get_feature_names, clear_cache

**Required Tests:**

**Unit Tests:** 
- **SequentialFeatureExtractor.__init__() Tests:**
  - Test initialization with valid SystemConfig object
  - Test initialization with None config (no classifier scenario)
  - Test initialization with config containing MovementPatternClassifier
  - Test sequence_cache initialization as empty dict
  - Test classifier initialization when config provided vs None
  
- **extract_features() Core Logic Tests:**
  - Test feature extraction with empty events list (should return defaults)
  - Test feature extraction with single event (should return appropriate defaults)
  - Test feature extraction with valid event sequence and target_time
  - Test lookback_hours filtering with events before/after cutoff
  - Test room_configs parameter handling (with and without configs)
  - Test integration of all sub-feature extractors (room transitions, velocity, sequences, cross-room, movement classification, n-grams)
  - Test feature aggregation and return dictionary structure
  - Test event sorting by timestamp before processing
  
- **_extract_room_transition_features() Tests:**
  - Test with less than 2 events (should return default values)
  - Test room transition counting with multiple room changes
  - Test unique rooms visited calculation
  - Test room revisit ratio calculation with repeated room visits
  - Test average room dwell time calculation
  - Test maximum room sequence length detection
  - Test transition regularity calculation with consistent/inconsistent intervals
  - Test room sequence detection with same room repeated events
  - Test dwell time calculation edge cases (same timestamp events)
  - Test statistics calculations with various interval patterns
  
- **_extract_velocity_features() Tests:**
  - Test with less than 2 events (should return defaults)
  - Test interval calculation using numpy diff on timestamps
  - Test basic interval statistics (mean, min, max, variance)
  - Test movement velocity score normalization (inverse of avg interval)
  - Test burst detection with intervals < 30 seconds
  - Test pause detection with intervals > 600 seconds
  - Test advanced velocity features with 3+ events (acceleration, autocorrelation, entropy)
  - Test velocity acceleration calculation using numpy std of interval changes
  - Test interval autocorrelation with normalized intervals
  - Test velocity entropy calculation using histogram bins
  - Test movement regularity coefficient of variation
  - Test numpy array operations and mathematical edge cases
  
- **_extract_sensor_sequence_features() Tests:**
  - Test with empty events (should return defaults)
  - Test unique sensors triggered counting
  - Test sensor revisit count calculation
  - Test dominant sensor ratio calculation
  - Test sensor diversity score using entropy
  - Test sensor type distribution analysis
  - Test presence/motion sensor ratio calculation
  - Test door sensor ratio calculation
  - Test entropy calculation edge cases (single sensor, multiple sensors)
  - Test Counter operations with sensor sequences
  
- **_extract_cross_room_features() Tests:**
  - Test with single room events (should return limited features)
  - Test active room count calculation
  - Test room correlation using sliding window approach
  - Test multi-room sequence ratio calculation with numpy
  - Test room switching frequency calculation
  - Test room activity entropy with numpy probability calculations
  - Test spatial clustering score with room runs analysis
  - Test room transition predictability using transition matrix
  - Test sliding window correlation with deque operations
  - Test numpy array operations for room sequence analysis
  - Test transition matrix normalization and probability calculations
  
- **_extract_movement_classification_features() Tests:**
  - Test with no events or no classifier (should return defaults)
  - Test with valid classifier and room_configs
  - Test sequence creation and classification integration
  - Test pattern matching against HUMAN_MOVEMENT_PATTERNS constants
  - Test pattern matching against CAT_MOVEMENT_PATTERNS constants
  - Test door interaction counting using SensorType filtering
  - Test unique pattern tracking with Set operations
  - Test confidence score aggregation
  - Test velocity and sequence length scoring
  - Test classification result aggregation across sequences
  
- **_extract_ngram_features() Tests:**
  - Test with less than 3 events (should return defaults)
  - Test bigram generation and counting
  - Test trigram generation and counting
  - Test common pattern ratio calculations
  - Test pattern repetition score calculation
  - Test Counter operations with n-gram tuples
  - Test sequence parsing from sensor IDs
  
- **_create_sequences_for_classification() Tests:**
  - Test sequence creation by room grouping
  - Test time gap filtering using MIN_EVENT_SEPARATION
  - Test sequence boundary detection using MAX_SEQUENCE_GAP
  - Test minimum sequence length requirement (≥2 events)
  - Test MovementSequence creation integration
  - Test filtering of None sequences
  - Test room-based event grouping
  
- **_create_movement_sequence() Tests:**
  - Test with less than 2 events (should return None)
  - Test MovementSequence creation with valid events
  - Test start/end time calculation
  - Test duration calculation
  - Test rooms_visited set creation
  - Test sensors_triggered set creation
  
- **Utility Method Tests:**
  - Test _get_default_features() returns complete feature dictionary
  - Test get_feature_names() returns all feature names from defaults
  - Test clear_cache() empties sequence_cache dictionary

**Integration Tests:**
- **Feature Pipeline Integration:**
  - Test complete feature extraction with realistic SensorEvent sequences
  - Test integration with MovementPatternClassifier from event_processor
  - Test room_configs integration with RoomConfig objects
  - Test SystemConfig integration with configuration loading
  - Test feature caching behavior across multiple extractions
  
- **Cross-Component Integration:**
  - Test integration with SensorEvent model from data.storage.models
  - Test integration with constants from core.constants module
  - Test integration with FeatureExtractionError exception handling
  - Test integration with MovementSequence from event_processor
  - Test logging integration for error reporting
  
- **Data Flow Integration:**
  - Test feature extraction with events from database queries
  - Test target_time filtering with real timestamp scenarios
  - Test lookback_hours with various time ranges
  - Test room configuration mapping with real room setups

**Edge Cases:**
- **Data Quality Edge Cases:**
  - Test with events having identical timestamps
  - Test with events in non-chronological order
  - Test with very large time gaps between events
  - Test with very rapid event sequences (burst scenarios)
  - Test with missing sensor_id or room_id fields
  - Test with malformed timestamp data
  - Test with sensor events spanning multiple days/weeks
  
- **Mathematical Edge Cases:**
  - Test numpy operations with NaN and infinity values
  - Test statistical calculations with zero variance
  - Test division by zero in ratio calculations
  - Test empty array operations in numpy
  - Test correlation calculations with constant sequences
  - Test entropy calculations with zero probability events
  - Test autocorrelation with perfectly regular sequences
  
- **Memory and Performance Edge Cases:**
  - Test with very large event sequences (1000+ events)
  - Test sequence_cache behavior with memory constraints
  - Test deque operations with sliding window overflow
  - Test numpy array memory usage with large datasets
  - Test pattern matching performance with complex room configurations
  
- **Configuration Edge Cases:**
  - Test with missing room_configs parameter
  - Test with empty room_configs dictionary
  - Test with rooms not present in room_configs
  - Test with SystemConfig containing no rooms
  - Test with classifier initialization failures

**Error Handling:**
- **Feature Extraction Errors:**
  - Test FeatureExtractionError raising with proper room_id context
  - Test exception handling in main extract_features method
  - Test error propagation from sub-feature extractors
  - Test logging of extraction failures
  - Test graceful degradation to default features on errors
  
- **Data Processing Errors:**
  - Test numpy operation failures and error recovery
  - Test statistics calculation errors (empty sequences, invalid data)
  - Test timestamp parsing errors in event processing
  - Test Counter operation failures with invalid data
  - Test Set operation failures in pattern tracking
  
- **Configuration Errors:**
  - Test missing classifier handling in movement classification
  - Test missing room configuration handling
  - Test invalid SystemConfig parameter handling
  - Test SensorType enum access failures
  
- **Memory Management Errors:**
  - Test cache overflow scenarios
  - Test memory allocation failures with large datasets
  - Test cleanup after extraction errors
  - Test thread safety in caching operations

**Coverage Target:** 85%+

### src/features/temporal.py - Temporal Feature Extraction
**Classes Found:** TemporalFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_time_since_features, _extract_duration_features, _extract_generic_sensor_features, _extract_cyclical_features, _extract_historical_patterns, _extract_transition_timing_features, _extract_room_state_features, _get_default_features, get_feature_names, validate_feature_names, clear_cache, extract_batch_features

**Required Tests:**
**Unit Tests:** 
- Test __init__ with various timezone_offset values (positive, negative, zero)
- Test extract_features with empty events list returns default features
- Test extract_features with valid events and all optional parameters
- Test extract_features with lookback_hours filtering functionality
- Test _extract_time_since_features with various event states ("on", "off", motion, presence)
- Test _extract_time_since_features with reversed chronological order handling
- Test _extract_time_since_features with time capping at 24 hours (86400 seconds)
- Test _extract_duration_features with single state changes and multiple transitions
- Test _extract_duration_features statistical calculations (mean, std, percentiles)
- Test _extract_duration_features with numpy array handling and empty arrays
- Test _extract_generic_sensor_features with mixed attribute types (numeric, boolean, string)
- Test _extract_generic_sensor_features with Mock attributes and runtime errors
- Test _extract_generic_sensor_features sensor type ratio calculations
- Test _extract_cyclical_features with timezone offset adjustments
- Test _extract_cyclical_features cyclical encodings (sin/cos transformations)
- Test _extract_cyclical_features binary indicators (weekend, work hours, sleep hours)
- Test _extract_historical_patterns with pandas DataFrame operations
- Test _extract_historical_patterns statistical features (variance, trend, seasonality)
- Test _extract_historical_patterns with weighted nearby hours calculations
- Test _extract_transition_timing_features with interval calculations
- Test _extract_transition_timing_features recent transition rate calculations
- Test _extract_transition_timing_features variability and regularity metrics
- Test _extract_room_state_features with RoomState objects and timestamp sorting
- Test get_feature_names returns complete list matching default features
- Test validate_feature_names with TEMPORAL_FEATURE_NAMES mapping
- Test clear_cache functionality
- Test extract_batch_features with multiple event batches and room states

**Integration Tests:**
- Test integration with SensorEvent model attributes and methods
- Test integration with RoomState model and database objects
- Test integration with TEMPORAL_FEATURE_NAMES constants
- Test integration with FeatureExtractionError exception handling
- Test pandas and numpy dependency integration
- Test timezone handling with real datetime objects
- Test feature extraction pipeline with realistic sensor data sequences
- Test batch processing performance with large event datasets

**Edge Cases:**
- Test with None/empty events list at various method levels
- Test with single event in events list
- Test with events having identical timestamps
- Test with events spanning multiple days/weeks/months
- Test with extreme timezone offsets (+/-12 hours)
- Test with malformed or None attribute values in events
- Test with division by zero scenarios (zero durations, empty arrays)
- Test with NaN/infinite values from statistical calculations
- Test with events having inconsistent state values ("of" vs "off")
- Test with very large time differences (years apart)
- Test with leap year and DST boundary conditions
- Test with room_states containing None values for optional fields
- Test memory usage with extremely large event sequences
- Test with events having non-standard sensor_type values

**Error Handling:**
- Test FeatureExtractionError raising with proper room_id extraction
- Test exception handling in extract_features main method
- Test pandas DataFrame creation failures
- Test numpy array operations with invalid data
- Test timezone conversion errors
- Test Mock attribute access exceptions (TypeError, AttributeError, RuntimeError)
- Test statistical calculation failures (empty sequences, invalid data)
- Test logging error messages for debugging
- Test graceful handling of malformed timestamp data
- Test handling of inconsistent sensor event formats

**Coverage Target:** 85%+

### src/features/contextual.py - Contextual Features
**Classes Found:** ContextualFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_environmental_features, _extract_door_state_features, _extract_multi_room_features, _extract_seasonal_features, _extract_sensor_correlation_features, _extract_room_context_features, _extract_numeric_values, _is_realistic_value, _calculate_trend, _calculate_change_rate, _calculate_room_activity_correlation, _calculate_room_state_correlation, _calculate_natural_light_score, _calculate_light_change_rate, _get_default_features, get_feature_names, clear_cache, _filter_environmental_events, _filter_door_events

**Required Tests:**
**Unit Tests:**
- Test __init__ with optional SystemConfig and threshold configuration validation
- Test extract_features with empty events list returns _get_default_features()
- Test extract_features with lookback_hours filtering and cutoff_time calculations
- Test extract_features main workflow with all feature extraction methods called
- Test _extract_environmental_features with SensorType enum filtering (CLIMATE, LIGHT)
- Test _extract_environmental_features temperature thresholds (cold: 18.0, comfortable: 22.0, warm: 26.0)
- Test _extract_environmental_features humidity comfort zones and stability calculations
- Test _extract_environmental_features light level categorization (dark: 100, dim: 300, bright: 1000)
- Test _extract_environmental_features statistical calculations (mean, variance, trend, change_rate)
- Test _extract_environmental_features with fallback sensor_id analysis for type detection
- Test _extract_door_state_features with SensorType.DOOR filtering and current state tracking
- Test _extract_door_state_features transition counting and duration calculations
- Test _extract_door_state_features time-based open ratio calculation methodology
- Test _extract_door_state_features recent activity detection (1 hour window)
- Test _extract_multi_room_features with room event grouping and state history processing
- Test _extract_multi_room_features simultaneous occupancy ratio calculations
- Test _extract_multi_room_features room activity correlation using sliding window approach
- Test _extract_multi_room_features dominant room activity and entropy-based balance
- Test _extract_multi_room_features with less than 2 rooms handling
- Test _extract_seasonal_features with month-based season indicators and holiday detection
- Test _extract_seasonal_features natural light availability patterns by season
- Test _extract_sensor_correlation_features with deque-based sliding window (300s)
- Test _extract_sensor_correlation_features multi-sensor event ratio and type diversity
- Test _extract_room_context_features with SensorType ratio calculations and room complexity
- Test _extract_numeric_values with state and attribute value extraction
- Test _is_realistic_value with sensor type validation ranges (temp: -50-100, humidity: 0-100, lux: 0-100000)
- Test _calculate_trend with linear slope calculation from value sequences
- Test _calculate_change_rate with consecutive absolute difference calculations
- Test _calculate_room_activity_correlation with 10-minute window correlation analysis
- Test _calculate_room_state_correlation with 5-minute window state change tracking
- Test _calculate_natural_light_score with time-based expected light pattern matching
- Test _calculate_light_change_rate with absolute light value change rate calculation
- Test get_feature_names returns complete list from _get_default_features keys
- Test clear_cache functionality for context_cache dictionary
- Test _filter_environmental_events with environmental sensor type filtering
- Test _filter_door_events with door sensor type filtering

**Integration Tests:**
- Test integration with SensorEvent model attributes (sensor_type, sensor_id, state, attributes)
- Test integration with RoomState model timestamp and occupancy tracking
- Test integration with SystemConfig and RoomConfig objects
- Test integration with SensorType enum constants (CLIMATE, LIGHT, DOOR, PRESENCE, MOTION)
- Test integration with FeatureExtractionError exception raising and room_id extraction
- Test numpy correlation coefficient calculations with np.corrcoef()
- Test statistics module integration (mean, variance) with proper error handling
- Test collections.defaultdict and deque usage in sliding window algorithms
- Test datetime timedelta operations and timezone handling
- Test feature extraction pipeline with realistic sensor data from multiple rooms
- Test environmental thresholds integration with actual sensor value ranges
- Test door state transition tracking with Home Assistant "on"/"open" state conventions

**Edge Cases:**
- Test with empty events list and None room_states
- Test with single event or single room scenarios
- Test with events having None/malformed attributes
- Test with events having invalid sensor_type values not in SensorType enum
- Test with room_states having identical timestamps
- Test with events spanning extreme time ranges (years)
- Test with numeric values at realistic range boundaries (temperature: -50, 100; humidity: 0, 100)
- Test with unrealistic sensor values (NaN, infinity, negative humidity)
- Test with door events having non-standard states ("partially_open", "locked")
- Test with correlation calculations on constant value sequences
- Test with room activity vectors of different lengths
- Test with empty sliding windows in sensor correlation analysis
- Test with seasonal boundary conditions (leap years, month transitions)
- Test with light values at threshold boundaries (exactly 100, 300, 1000 lux)
- Test with room configurations having missing or empty sensor lists
- Test memory usage with large multi-room datasets and correlation matrices

**Error Handling:**
- Test FeatureExtractionError raising with proper room_id extraction from events
- Test exception handling in main extract_features method with comprehensive logging
- Test numpy correlation calculation failures (LinAlgError, ValueError, FloatingPointError)
- Test statistics calculations with empty sequences and invalid data types
- Test attribute access errors on SensorEvent objects (missing attributes)
- Test division by zero in ratio calculations (door_open_ratio, occupancy_ratio)
- Test float conversion errors in _extract_numeric_values with invalid state values
- Test defaultdict key access with missing room data
- Test deque operations with invalid event sequences
- Test correlation coefficient calculation with all-zero vectors
- Test natural light score calculation with invalid hour values
- Test trend calculation with insufficient data points (less than 2)
- Test realistic value validation with type mismatches
- Test environmental threshold comparisons with None values

**Coverage Target:** 85%+

### src/features/store.py - Feature Store Management
**Classes Found:** FeatureRecord, FeatureCache, FeatureStore
**Methods Analyzed:** FeatureRecord.to_dict(), FeatureRecord.from_dict(), FeatureRecord.is_valid(), FeatureCache._make_key(), FeatureCache.get(), FeatureCache.put(), FeatureCache.clear(), FeatureCache.get_stats(), FeatureStore.__init__(), FeatureStore.initialize(), FeatureStore.get_features(), FeatureStore.get_batch_features(), FeatureStore.compute_training_data(), FeatureStore._compute_features(), FeatureStore._get_data_for_features(), FeatureStore._get_features_from_db(), FeatureStore._persist_features_to_db(), FeatureStore._compute_data_hash(), FeatureStore.get_stats(), FeatureStore.clear_cache(), FeatureStore.reset_stats(), FeatureStore.health_check(), FeatureStore.get_statistics(), FeatureStore.__aenter__(), FeatureStore.__aexit__()

**Required Tests:**
**Unit Tests:** 
- Test FeatureRecord serialization/deserialization with datetime handling
- Test FeatureRecord.is_valid() with various datetime scenarios (timezone-aware vs naive, mock datetime)
- Test FeatureCache LRU eviction mechanics with max_size boundaries
- Test FeatureCache key generation with different parameter combinations
- Test FeatureCache hit/miss statistics tracking accuracy
- Test FeatureStore initialization with/without persistence and custom parameters
- Test FeatureStore async initialization with database connection success/failure
- Test FeatureStore.get_features() with cache hits, misses, and forced recomputation
- Test FeatureStore batch processing with concurrent requests and exception handling
- Test FeatureStore training data generation with various time intervals and lookback periods
- Test FeatureStore statistics collection and reset functionality
- Test FeatureStore health check with various component failure states
- Test async context manager entry/exit methods

**Integration Tests:**
- Test FeatureStore with real DatabaseManager integration for data persistence
- Test FeatureStore with FeatureEngineeringEngine integration for feature computation
- Test end-to-end feature computation from database query to cached result
- Test batch feature extraction with real database queries and concurrent processing
- Test training data generation with actual sensor events and room states
- Test feature cache persistence across FeatureStore lifecycle
- Test FeatureStore with multiple rooms and overlapping time windows
- Test async database operations with connection pooling and error recovery

**Edge Cases:**
- Test FeatureCache with zero max_size and negative values
- Test FeatureRecord.is_valid() with edge datetime values (year boundaries, leap years)
- Test FeatureStore with empty feature types list and None parameters
- Test batch processing with empty request lists and duplicate requests
- Test training data generation with invalid date ranges (end < start)
- Test feature computation with missing or corrupted database data
- Test cache key generation with extremely long room IDs and feature type lists
- Test concurrent cache access with race conditions
- Test memory usage with very large cache sizes and feature dictionaries
- Test datetime handling across different timezones and DST transitions
- Test feature store with database unavailable during runtime
- Test batch operations with mixed success/failure results

**Error Handling:**
- Test FeatureRecord.from_dict() with malformed datetime strings and missing fields
- Test FeatureCache operations with invalid room_ids and malformed target_times
- Test FeatureStore initialization with invalid config and missing dependencies
- Test database query failures in _get_data_for_features() with proper exception handling
- Test feature computation failures with graceful fallback to default features
- Test batch processing with individual task exceptions and partial result handling
- Test async operations with database connection timeouts and network failures
- Test cache operations with memory constraints and serialization errors
- Test health check failures with proper error reporting and status degradation
- Test async context manager with database initialization failures
- Test statistics collection with corrupted internal state
- Test concurrent access errors with proper locking and recovery

**Coverage Target:** 85%+

### src/features/engineering.py - Feature Engineering Pipeline
**Classes Found:** FeatureEngineeringEngine
**Methods Analyzed:** __init__, extract_features, extract_batch_features, _extract_features_parallel, _extract_features_sequential, _add_metadata_features, get_feature_names, create_feature_dataframe, _get_default_features, get_extraction_stats, reset_stats, clear_caches, validate_configuration, compute_feature_correlations, analyze_feature_importance, _validate_configuration, compute_feature_statistics, _calculate_skewness, _calculate_kurtosis, _calculate_entropy, _count_outliers, __del__

**Required Tests:**
**Unit Tests:**
- Test __init__ with default parameters (config=None, enable_parallel=True, max_workers=3)
- Test __init__ with custom SystemConfig and parameter variations
- Test __init__ initialization of temporal, sequential, and contextual extractors
- Test __init__ ThreadPoolExecutor setup with enable_parallel=True/False
- Test __init__ stats dictionary initialization with proper structure
- Test __init__ configuration validation when config is provided vs None
- Test extract_features with all parameter combinations (room_id, target_time, events, room_states, lookback_hours, feature_types)
- Test extract_features validation with empty/None room_id raises FeatureExtractionError
- Test extract_features with missing room configuration (warning logged)
- Test extract_features with feature_types=None defaults to all three types
- Test extract_features event filtering by room_id and time window (cutoff_time)
- Test extract_features calls parallel vs sequential extraction based on enable_parallel
- Test extract_features metadata feature addition and statistics updating
- Test extract_batch_features with multiple extraction requests
- Test extract_batch_features parallel processing with asyncio.gather
- Test extract_batch_features sequential processing fallback
- Test extract_batch_features exception handling with _get_default_features
- Test _extract_features_parallel with all three feature types
- Test _extract_features_parallel with loop.run_in_executor calls
- Test _extract_features_parallel with room_config handling (None vs valid)
- Test _extract_features_parallel exception handling and failed_extractors tracking
- Test _extract_features_parallel feature prefixing ("temporal_", "sequential_", "contextual_")
- Test _extract_features_parallel metadata features for failed extractors
- Test _extract_features_sequential with individual try/catch blocks per feature type
- Test _extract_features_sequential with room_config dictionary creation
- Test _extract_features_sequential feature prefixing and stats updating
- Test _add_metadata_features with numpy array operations and normalization
- Test _add_metadata_features vector norm calculation and feature value processing
- Test get_feature_names with feature_types parameter variations
- Test get_feature_names calls to individual extractor get_feature_names methods
- Test create_feature_dataframe with empty and non-empty feature_dicts
- Test create_feature_dataframe DataFrame creation with consistent columns
- Test _get_default_features combining defaults from all extractors
- Test get_extraction_stats returns stats copy
- Test reset_stats resets all statistics to initial values
- Test clear_caches calls clear_cache on all extractors
- Test validate_configuration with _original_config_was_none flag
- Test validate_configuration with None vs valid config scenarios
- Test validate_configuration extractor initialization checks
- Test validate_configuration parallel processing validation
- Test compute_feature_correlations with pandas correlation matrix
- Test compute_feature_correlations with numpy high correlation detection
- Test compute_feature_correlations with empty DataFrame edge case
- Test analyze_feature_importance with target correlation calculations
- Test analyze_feature_importance with target datetime conversion
- Test analyze_feature_importance with numpy correlation coefficient calculations
- Test _validate_configuration with ConfigurationError raising
- Test compute_feature_statistics with pandas describe() and numpy calculations
- Test compute_feature_statistics summary statistics (constant features, high variance)
- Test _calculate_skewness with numpy moment calculations
- Test _calculate_kurtosis with numpy moment calculations and -3 adjustment
- Test _calculate_entropy with histogram discretization and log calculations
- Test _count_outliers with IQR method and numpy operations
- Test __del__ executor shutdown functionality

**Integration Tests:**
- Test integration with TemporalFeatureExtractor, SequentialFeatureExtractor, ContextualFeatureExtractor
- Test integration with SystemConfig and RoomConfig objects
- Test integration with SensorEvent and RoomState models
- Test integration with ConfigurationError and FeatureExtractionError exceptions
- Test ThreadPoolExecutor integration with asyncio event loop
- Test pandas DataFrame operations with real feature data
- Test numpy statistical calculations with realistic datasets
- Test feature extraction pipeline end-to-end with mock data
- Test batch processing with concurrent tasks and resource management
- Test correlation analysis integration with scikit-learn compatible output format

**Edge Cases:**
- Test with None config and _original_config_was_none=True validation
- Test with config.rooms empty or None
- Test with max_workers < 1 raising ConfigurationError  
- Test with ThreadPoolExecutor creation failure
- Test with all feature extractors returning exceptions in parallel mode
- Test with partial extractor failures in parallel mode
- Test with empty events and room_states lists
- Test with single-element feature_dicts in DataFrame creation
- Test with feature dictionaries having inconsistent keys
- Test with correlation matrix containing NaN values
- Test with empty DataFrame in correlation and importance analysis
- Test with feature importance calculations having zero variance
- Test with statistical calculations on empty or single-value arrays
- Test with extreme values causing overflow in statistical calculations
- Test with malformed target data in analyze_feature_importance
- Test with features_df and targets_df having mismatched lengths
- Test with division by zero in normalization operations
- Test with numpy array operations on non-numeric data
- Test memory usage with large feature extraction batches
- Test executor shutdown during active task processing

**Error Handling:**
- Test FeatureExtractionError raising with proper feature_type, room_id, and cause
- Test exception handling in extract_features with stats.failed_extractions increment
- Test exception propagation from individual feature extractors
- Test asyncio.gather exception handling in extract_batch_features
- Test ThreadPoolExecutor task failures and cleanup
- Test pandas DataFrame operations with invalid data types
- Test numpy array operations with NaN/infinite values
- Test correlation matrix calculations with singular matrices
- Test ConfigurationError raising in _validate_configuration
- Test logging error messages for debugging and monitoring
- Test executor shutdown exceptions in __del__
- Test extractor initialization failures during __init__
- Test validate_configuration with None extractors
- Test feature name conflicts between extractors
- Test memory exhaustion during large batch processing
- Test timeout scenarios in parallel feature extraction

**Coverage Target:** 85%+

## Summary

This comprehensive feature engineering testing requirements document covers all 5+ feature engineering components with detailed testing specifications including:

- **Sequential Pattern Features**: Room transitions, velocity analysis, sensor sequences, cross-room correlations, movement classification, and n-gram patterns
- **Temporal Feature Extraction**: Time-based features, cyclical encodings, historical patterns, duration analysis, and timezone handling
- **Contextual Features**: Environmental data processing, multi-room analysis, seasonal patterns, sensor correlations, and natural light scoring
- **Feature Store Management**: Caching mechanisms, persistence, batch processing, training data generation, and health monitoring  
- **Feature Engineering Pipeline**: Parallel/sequential processing, metadata features, statistical analysis, correlation computation, and validation

Each component includes comprehensive unit tests, integration tests, edge cases, error handling scenarios, and specific coverage targets of 85%+ to ensure robust feature engineering functionality.

**Key Testing Focus Areas:**
- Mathematical accuracy of statistical calculations
- Edge case handling for sensor data irregularities
- Performance optimization for large-scale processing
- Integration between all feature extraction components
- Error handling and graceful degradation
- Memory management and resource cleanup
- Async processing and threading safety
- Configuration validation and flexibility

**Mock Requirements:**
- Mock SystemConfig, RoomConfig, SensorEvent, and RoomState objects
- Mock MovementPatternClassifier for classification testing
- Mock numpy, pandas, and statistics operations for deterministic testing
- Mock ThreadPoolExecutor and asyncio operations for parallel processing tests
- Mock logging and datetime utilities for consistent test behavior

**Test Fixtures Needed:**
- Realistic sensor event sequences with various temporal patterns
- Multi-room configurations with complex sensor mappings
- Environmental data spanning different seasons and conditions
- Movement pattern data for classification testing
- Statistical edge case datasets for mathematical validation
- Large-scale batch processing datasets for performance testing