# Data Layer Testing Requirements

## Overview
This document contains detailed testing requirements for the ha-ml-predictor data layer components to achieve 85%+ test coverage. Each component has been analyzed for actual implementation details and specific testing scenarios.

### src/data/ingestion/event_processor.py - Event Processing Pipeline

**Classes Found:** 
- MovementSequence (dataclass)
- ValidationResult (dataclass) 
- ClassificationResult (dataclass)
- EventValidator
- MovementPatternClassifier
- EventProcessor

**Methods Analyzed:**
- MovementSequence: average_velocity, trigger_pattern
- EventValidator: __init__, validate_event
- MovementPatternClassifier: __init__, classify_movement, analyze_sequence_patterns, get_sequence_time_analysis, extract_movement_signature, compare_movement_patterns, and 15+ private methods for metric calculations
- EventProcessor: __init__, process_event, process_event_batch, validate_event_sequence_integrity, validate_room_configuration, and 8+ private helper methods

**Required Tests:**

**Unit Tests:**
- **MovementSequence Tests:**
  - Test average_velocity calculation with 2+ events vs single event
  - Test trigger_pattern string generation from sensor event sequences
  - Test property methods with empty/null events list
  - Test duration calculation edge cases (zero duration, negative timestamps)

- **EventValidator Tests:**
  - Test validate_event with valid sensor events (all required fields present)
  - Test validate_event with missing required fields (room_id, sensor_id, state, timestamp)
  - Test state validation using PRESENCE_STATES, ABSENCE_STATES, INVALID_STATES constants
  - Test state transition validation (presence->absence, same category transitions)
  - Test timestamp validation (future timestamps, old timestamps >24h)
  - Test room and sensor configuration validation with unknown room_id
  - Test confidence score calculations for various warning conditions
  - Test SensorState enum validation for valid/invalid states

- **MovementPatternClassifier Tests:**
  - Test classify_movement with human-pattern sequences (door interactions, moderate velocity)
  - Test classify_movement with cat-pattern sequences (high velocity, no doors, revisits)
  - Test _calculate_movement_metrics with comprehensive metric validation
  - Test _score_human_pattern and _score_cat_pattern scoring algorithms
  - Test mathematical functions: _calculate_movement_entropy, _calculate_spatial_dispersion
  - Test timing analysis: _calculate_avg_dwell_time, _calculate_timing_variance
  - Test velocity calculations: _calculate_max_velocity with edge cases
  - Test door/presence sensor counting methods with room configuration
  - Test analyze_sequence_patterns return format (classification, confidence, metrics)
  - Test get_sequence_time_analysis timing calculations
  - Test extract_movement_signature path and frequency extraction
  - Test compare_movement_patterns similarity scoring
  - Test confidence adjustments for short sequences (<3 events, <5 seconds duration)

- **EventProcessor Tests:**
  - Test __init__ with config parameter and default config loading
  - Test process_event with valid HAEvent conversion to SensorEvent
  - Test process_event room configuration lookup and sensor type determination
  - Test event validation integration and invalid event filtering
  - Test duplicate event detection using MIN_EVENT_SEPARATION
  - Test event enrichment with movement classification
  - Test _update_event_tracking deque management and time tracking
  - Test process_event_batch with batching and async sleep behavior
  - Test _determine_sensor_type from entity_id patterns and room config
  - Test _is_duplicate_event timing logic
  - Test _create_movement_sequence with MAX_SEQUENCE_GAP filtering
  - Test _check_room_state_change with tracking_manager integration
  - Test get_processing_stats and reset_stats functionality
  - Test validate_event_sequence_integrity mathematical analysis
  - Test validate_room_configuration validation logic

**Integration Tests:**
- **End-to-End Event Processing:**
  - Test complete HAEvent -> SensorEvent pipeline with real room configurations
  - Test batch processing with mixed valid/invalid events
  - Test event sequence analysis with realistic sensor data patterns
  - Test classification integration with actual movement pattern data
  - Test tracking_manager integration for room state changes

- **Configuration Integration:**
  - Test event processor with various room configurations (single room, multi-room)
  - Test sensor type determination with complex room configurations
  - Test validation with missing/incomplete room configurations

- **Statistics and Tracking:**
  - Test statistics accumulation across multiple event processing
  - Test recent events tracking and deque behavior with maxlen=100
  - Test room state change detection patterns for presence vs motion sensors

**Edge Cases:**
- **Temporal Edge Cases:**
  - Events with identical timestamps
  - Events with timestamps in different timezones
  - Events with microsecond precision timing differences
  - Sequence analysis with events spanning MAX_SEQUENCE_GAP boundary

- **Data Quality Edge Cases:**
  - Empty or null attributes in HAEvent/SensorEvent
  - Very large event batches (>1000 events)
  - Events with malformed entity_ids or room_ids
  - Sequences with all identical sensor triggers
  - Mathematical edge cases: zero duration, infinite velocity calculations

- **Classification Edge Cases:**
  - Movement sequences with single event
  - Sequences with all sensors triggered simultaneously
  - Zero-distance movement (same sensor repeated)
  - Extremely fast sequences (<1 second duration)
  - Sequences spanning multiple rooms with complex transitions

- **Configuration Edge Cases:**
  - Room configuration with no sensors
  - Sensors with missing entity_ids in configuration
  - Mixed sensor types (strings vs dicts) in room configuration
  - Room configuration validation with missing sensor types

**Error Handling:**
- **Exception Testing:**
  - Test ConfigurationError handling in validate_event
  - Test DataValidationError handling in sequence validation
  - Test FeatureExtractionError handling in mathematical calculations
  - Test graceful handling of tracking_manager failures
  - Test exception propagation vs suppression patterns

- **Robustness Testing:**
  - Test behavior with corrupted event data
  - Test mathematical function edge cases (division by zero, log of zero)
  - Test memory behavior with very large event sequences
  - Test async method cancellation and timeout handling

- **Validation Error Testing:**
  - Test validation with completely invalid HAEvent objects
  - Test sequence integrity validation with corrupted event sequences
  - Test mathematical anomaly detection with statistical outliers

**Coverage Target: 85%+**

**Additional Coverage Requirements:**
- All mathematical calculation methods must have edge case coverage
- All private helper methods must be tested through public method calls
- All error conditions and exception paths must be covered
- All configuration validation branches must be tested
- All statistical analysis code paths must be verified

### src/data/storage/models.py - SQLAlchemy Models
**Classes Found:** SensorEvent, RoomState, Prediction, ModelAccuracy, FeatureStore, PredictionAudit
**Methods Analyzed:** SensorEvent.__init__(), SensorEvent.get_recent_events(), SensorEvent.get_state_changes(), SensorEvent.get_transition_sequences(), SensorEvent.get_predictions(), SensorEvent.get_advanced_analytics(), SensorEvent.get_sensor_efficiency_metrics(), SensorEvent.get_temporal_patterns(), SensorEvent._calculate_efficiency_score(), RoomState.get_current_state(), RoomState.get_occupancy_history(), RoomState.get_predictions(), RoomState.get_occupancy_sessions(), RoomState.get_precision_occupancy_metrics(), Prediction.__init__(), Prediction.get_pending_validations(), Prediction.get_accuracy_metrics(), Prediction.get_triggering_event(), Prediction.get_room_state(), Prediction.get_predictions_with_events(), Prediction.get_predictions_with_full_context(), Prediction._extract_top_features(), Prediction._categorize_features(), Prediction._analyze_confidence_spread(), Prediction.add_extended_metadata(), FeatureStore.get_latest_features(), FeatureStore.get_all_features(), PredictionAudit.create_audit_entry(), PredictionAudit.get_audit_trail_with_relationships(), PredictionAudit.analyze_json_details(), PredictionAudit._calculate_json_complexity(), PredictionAudit.update_validation_metrics(), plus utility functions _is_sqlite_engine(), _get_database_specific_column_config(), _get_json_column_type(), create_timescale_hypertables(), optimize_database_performance(), get_bulk_insert_query()

**Required Tests:**
- Unit Tests: Model initialization tests, class method functionality, JSON field handling, data validation, relationship handling, query method tests, utility function tests, static method tests, configuration tests
- Integration Tests: Database operations with real SQLAlchemy sessions, TimescaleDB integration, cross-model relationships, bulk operations, async session handling
- Edge Cases: Large JSON payloads, extreme timestamp values, Unicode data, null/None handling, database-specific behavior differences
- Error Handling: Constraint violations, foreign key errors, JSON parsing errors, database connection failures, TimescaleDB unavailability
- Coverage Target: 85%+

### src/data/ingestion/ha_client.py - Home Assistant Client
**Classes Found:** HAEvent, RateLimiter, HomeAssistantClient
**Methods Analyzed:** HAEvent.__init__(), HAEvent.is_valid(), RateLimiter.__init__(), RateLimiter.acquire(), HomeAssistantClient.__init__(), HomeAssistantClient.__aenter__(), HomeAssistantClient.__aexit__(), HomeAssistantClient.connect(), HomeAssistantClient.disconnect(), HomeAssistantClient._cleanup_connections(), HomeAssistantClient._test_authentication(), HomeAssistantClient._connect_websocket(), HomeAssistantClient._authenticate_websocket(), HomeAssistantClient._handle_websocket_messages(), HomeAssistantClient._process_websocket_message(), HomeAssistantClient._handle_event(), HomeAssistantClient._validate_and_normalize_state(), HomeAssistantClient._should_process_event(), HomeAssistantClient._notify_event_handlers(), HomeAssistantClient._reconnect(), HomeAssistantClient.subscribe_to_events(), HomeAssistantClient.add_event_handler(), HomeAssistantClient.remove_event_handler(), HomeAssistantClient.get_entity_state(), HomeAssistantClient.get_entity_history(), HomeAssistantClient.get_bulk_history(), HomeAssistantClient.validate_entities(), HomeAssistantClient.convert_ha_event_to_sensor_event(), HomeAssistantClient.convert_history_to_sensor_events(), HomeAssistantClient.is_connected()

**Required Tests:**
- Unit Tests: 
  - **HAEvent Class Tests:**
    - Test HAEvent creation with all required fields (entity_id, state, previous_state, timestamp, attributes)
    - Test is_valid() method with valid events, invalid states from INVALID_STATES constant
    - Test is_valid() with missing entity_id, missing timestamp, empty state scenarios
    - Test event_type default value and custom event_type assignment
  
  - **RateLimiter Class Tests:**
    - Test __init__() with default parameters (300 requests, 60 seconds window) 
    - Test __init__() with custom max_requests and window_seconds parameters
    - Test acquire() method under normal conditions (requests within limit)
    - Test acquire() method when rate limit reached - should wait, not raise exception
    - Test request window sliding behavior with time-based request expiration
    - Test concurrent acquire() calls with asyncio.Lock protection
    - Test wait time calculation when rate limit exceeded
    - Test request timestamp tracking and cleanup of old requests
  
  - **HomeAssistantClient Initialization Tests:**
    - Test __init__() with default config from get_config()
    - Test __init__() with custom SystemConfig parameter
    - Test initialization of session, websocket, rate_limiter, and state variables
    - Test connection state initialization (_connected, _reconnect_attempts, etc.)
    - Test event handling initialization (_event_handlers, _subscribed_entities, etc.)
    - Test WebSocket message tracking initialization (_ws_message_id, _pending_responses)
  
  - **Connection Management Tests:**
    - Test connect() method establishing HTTP session with proper headers
    - Test connect() calling _test_authentication() and _connect_websocket()
    - Test connect() setting _connected state and resetting reconnect attempts  
    - Test disconnect() method cleaning up connections and setting _connected=False
    - Test __aenter__() and __aexit__() async context manager methods
    - Test _cleanup_connections() closing websocket and session gracefully
    - Test connection failure scenarios with HomeAssistantConnectionError
  
  - **Authentication Tests:**
    - Test _test_authentication() with successful API response (status 200)
    - Test _test_authentication() with authentication failure (status 401) - HomeAssistantAuthenticationError
    - Test _test_authentication() with other API errors - HomeAssistantAPIError
    - Test _test_authentication() with network errors - HomeAssistantConnectionError
    - Test proper URL construction using urljoin() for API endpoint
  
  - **WebSocket Connection Tests:**
    - Test _connect_websocket() URL construction (http->ws, https->wss conversion)
    - Test _connect_websocket() with successful websockets.connect()
    - Test _connect_websocket() with connection failures - WebSocketError
    - Test WebSocket timeout and ping configuration (ping_interval=20, ping_timeout=10)
  
  - **WebSocket Authentication Tests:**
    - Test _authenticate_websocket() handling auth_required message
    - Test _authenticate_websocket() sending auth response with access token
    - Test _authenticate_websocket() successful auth_ok response
    - Test _authenticate_websocket() authentication failure scenarios
    - Test WebSocketError for unexpected message types during auth
  
  - **WebSocket Message Handling Tests:**
    - Test _handle_websocket_messages() processing incoming messages
    - Test _process_websocket_message() routing different message types (event, result, pong)
    - Test _handle_event() processing state_changed events  
    - Test event filtering by subscribed entities
    - Test event validation and state normalization
    - Test event deduplication using MIN_EVENT_SEPARATION
    - Test JSON parsing error handling for malformed messages
    - Test ConnectionClosed exception handling with reconnection
  
  - **State Validation Tests:**
    - Test _validate_and_normalize_state() with exact state mappings (on->on, off->off, etc.)
    - Test _validate_and_normalize_state() with partial matches (active->on, inactive->off)
    - Test _validate_and_normalize_state() with motion detection patterns (detect->on, clear->off)
    - Test _validate_and_normalize_state() with door states (open->open, closed->closed)
    - Test _validate_and_normalize_state() with unavailable/unknown states
    - Test _validate_and_normalize_state() with unknown states logging and passthrough
    - Test case insensitive state normalization and whitespace trimming
  
  - **Event Processing Tests:**
    - Test _should_process_event() with valid events passing through
    - Test _should_process_event() with invalid events (HAEvent.is_valid() = False)
    - Test _should_process_event() enforcing MIN_EVENT_SEPARATION timing
    - Test _notify_event_handlers() calling registered handlers
    - Test _notify_event_handlers() with async and sync handler functions
    - Test _notify_event_handlers() error handling for handler exceptions
    - Test timestamp parsing with various formats (Z suffix, timezone handling)
    - Test double timezone suffix handling in timestamps
  
  - **Reconnection Logic Tests:**
    - Test _reconnect() with exponential backoff calculation
    - Test _reconnect() maximum attempts limit (_max_reconnect_attempts = 10)
    - Test _reconnect() delay capping at 300 seconds (5 minutes)
    - Test _reconnect() re-subscribing to entities after reconnection
    - Test _reconnect() failure handling and retry attempts
    - Test reconnection triggered by WebSocket connection loss
  
  - **Event Subscription Tests:**
    - Test subscribe_to_events() with entity ID list
    - Test subscribe_to_events() WebSocket command formatting
    - Test subscribe_to_events() response handling with success/failure
    - Test subscribe_to_events() timeout handling (10 second wait)
    - Test subscribe_to_events() when not connected - HomeAssistantConnectionError
    - Test _subscribed_entities set updates and tracking
  
  - **Event Handler Management Tests:**
    - Test add_event_handler() adding handlers to list
    - Test remove_event_handler() removing handlers from list
    - Test remove_event_handler() with non-existent handler (no error)
    - Test event handler list management and callback execution
  
  - **REST API Tests:**
    - Test get_entity_state() with successful response and state normalization
    - Test get_entity_state() with 404 response returning None
    - Test get_entity_state() with 429 rate limiting - RateLimitExceededError  
    - Test get_entity_state() with other HTTP errors - HomeAssistantAPIError
    - Test get_entity_state() with network errors - HomeAssistantConnectionError
    - Test get_entity_state() rate limiting integration with rate_limiter.acquire()
    - Test proper URL construction using urljoin() for entity state endpoint
  
  - **Historical Data Tests:**
    - Test get_entity_history() parameter formatting and URL construction
    - Test get_entity_history() with successful response and state normalization
    - Test get_entity_history() with 404 response - EntityNotFoundError
    - Test get_entity_history() with rate limiting scenarios
    - Test get_entity_history() end_time defaulting to current UTC time
    - Test get_entity_history() response parsing and validation
  
  - **Bulk Operations Tests:**
    - Test get_bulk_history() batching logic with batch_size parameter
    - Test get_bulk_history() AsyncGenerator yielding behavior
    - Test get_bulk_history() rate limiting handling in bulk operations
    - Test get_bulk_history() error handling and entity skipping
    - Test get_bulk_history() retry logic for rate limited requests
    - Test get_bulk_history() small delays between requests (0.1s sleep)
  
  - **Entity Validation Tests:**
    - Test validate_entities() checking entity existence
    - Test validate_entities() returning boolean mapping for each entity
    - Test validate_entities() error handling for validation failures
  
  - **Data Conversion Tests:**
    - Test convert_ha_event_to_sensor_event() field mapping
    - Test convert_ha_event_to_sensor_event() with room_id and sensor_type parameters
    - Test convert_ha_event_to_sensor_event() setting is_human_triggered and created_at
    - Test convert_history_to_sensor_events() processing list of history records
    - Test convert_history_to_sensor_events() timestamp parsing and format handling
    - Test convert_history_to_sensor_events() previous_state tracking across records
    - Test convert_history_to_sensor_events() handling invalid timestamp formats
  
  - **Connection Status Tests:**
    - Test is_connected property with various connection states
    - Test is_connected checking _connected, websocket existence, and websocket.closed status

- Integration Tests:
  - **Full Connection Workflow:**
    - Test complete connect() -> authenticate -> subscribe -> receive events workflow
    - Test connection failure recovery and reconnection scenarios
    - Test WebSocket authentication flow with real Home Assistant instance
    - Test event subscription and handler notification integration
  
  - **Rate Limiting Integration:**
    - Test rate limiter behavior with actual API requests
    - Test rate limit handling across different API endpoints
    - Test bulk operations rate limiting and retry behavior
  
  - **Historical Data Integration:**
    - Test historical data fetching with real Home Assistant API
    - Test bulk historical data processing with large datasets
    - Test data conversion integration from HA formats to SensorEvent models
  
  - **Error Recovery Integration:**
    - Test complete error recovery scenarios (connection loss, authentication failure)
    - Test reconnection with existing subscriptions and handlers
    - Test graceful degradation when Home Assistant is unavailable

- Edge Cases:
  - **Connection Edge Cases:**
    - Test connection with invalid Home Assistant URLs
    - Test connection with malformed authentication tokens
    - Test network timeout scenarios during connection establishment
    - Test WebSocket connection drops during active usage
    - Test concurrent connection attempts and thread safety
  
  - **Data Processing Edge Cases:**
    - Test events with extremely large attributes JSON payloads
    - Test events with missing or null fields in various combinations
    - Test timestamp parsing with edge case formats and timezones
    - Test state validation with very long state strings
    - Test Unicode characters in entity IDs and state values
  
  - **Rate Limiting Edge Cases:**
    - Test rate limiter behavior at exact limit boundaries
    - Test concurrent rate limiter access with high contention
    - Test rate limiter with very small time windows
    - Test system clock changes affecting rate limit calculations
  
  - **WebSocket Edge Cases:**
    - Test WebSocket message size limits and very large messages
    - Test rapid WebSocket reconnection scenarios
    - Test WebSocket authentication with expired tokens
    - Test malformed JSON in WebSocket messages
    - Test WebSocket pong/ping timeout scenarios
  
  - **API Edge Cases:**
    - Test API requests with very long entity ID lists
    - Test historical data requests spanning very long time periods
    - Test API responses with unexpected content types
    - Test network interruptions during large bulk operations

- Error Handling:
  - **Connection Errors:**
    - Test HomeAssistantConnectionError scenarios with proper error context
    - Test HomeAssistantAuthenticationError with token length information
    - Test WebSocketError handling with connection details
    - Test network timeout and DNS resolution failures
  
  - **API Errors:**
    - Test HomeAssistantAPIError with status codes, response text, and method info
    - Test EntityNotFoundError for non-existent entities
    - Test RateLimitExceededError with retry timing information
    - Test malformed API responses and JSON parsing errors
  
  - **Data Processing Errors:**
    - Test invalid event data handling and filtering
    - Test corrupted WebSocket messages and recovery
    - Test timestamp parsing failures and fallback behavior
    - Test handler exception isolation (one handler failure doesn't affect others)
  
  - **Resource Management Errors:**
    - Test cleanup during connection failures
    - Test resource cleanup in exception scenarios
    - Test memory management with large numbers of pending responses
    - Test WebSocket resource cleanup on abnormal disconnection

- Coverage Target: 85%+

### src/data/storage/database.py - Database Connection Management
**Classes Found:** DatabaseManager
**Methods Analyzed:** DatabaseManager.__init__(), initialize(), _create_engine(), _setup_connection_events(), _setup_session_factory(), _verify_connection(), get_session(), execute_query(), execute_optimized_query(), analyze_query_performance(), _get_optimization_suggestions(), get_connection_pool_metrics(), health_check(), _health_check_loop(), close(), _cleanup(), get_connection_stats(), is_initialized, get_database_manager(), get_db_session(), close_database_manager(), execute_sql_file(), check_table_exists(), get_database_version(), get_timescaledb_version()

**Required Tests:**
- Unit Tests: 
  - DatabaseManager initialization with config parameter and default config loading
  - Connection string validation and async driver conversion (postgresql -> postgresql+asyncpg)
  - Engine creation with proper SQLAlchemy 2.0 async configuration (pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping)
  - Connection event listener setup (connect, checkout, checkin, invalidate) with proper statistics tracking
  - Session factory creation with AsyncSession and proper transaction settings (expire_on_commit=False, autoflush=True)
  - Database connectivity verification with basic SELECT 1 test
  - TimescaleDB extension verification and logging behavior
  - Session context manager functionality with automatic commit/rollback
  - Retry logic with exponential backoff for connection failures (max_retries=5, base_delay=1.0, backoff_multiplier=2.0)
  - Query execution with parameters, fetch_one, fetch_all options
  - Query timeout handling using asyncio.wait_for with configurable timeout
  - Optimized query execution with prepared statements and query plan caching
  - Query performance analysis with EXPLAIN ANALYZE and execution time measurement
  - Connection pool metrics extraction (pool_size, checked_out, overflow, invalid_count, utilization_percent)
  - Health check comprehensive testing (connectivity, TimescaleDB status, performance metrics, pool status)
  - Background health check loop with configurable interval and cancellation
  - Connection statistics tracking (_connection_stats dictionary management)
  - Resource cleanup and engine disposal
  - Property methods (is_initialized getter)

- Integration Tests:
  - End-to-end database connection workflow with real PostgreSQL/TimescaleDB
  - Session management across multiple concurrent operations
  - Connection pool behavior under load with multiple simultaneous connections
  - TimescaleDB-specific functionality (hypertable operations, time-based queries)
  - Error recovery testing with database connection interruptions
  - Health check loop integration with real database monitoring
  - SQL file execution with transaction management
  - Global database manager singleton behavior and initialization
  - Cross-component integration with other database-dependent modules

- Edge Cases:
  - Connection string edge cases (missing protocol, invalid format, special characters)
  - Engine configuration with NullPool for testing scenarios (pool_size <= 0)
  - Connection event handling with None/missing parameters
  - TimescaleDB extension absence handling and warning logging
  - Query timeout with very long-running operations
  - Prepared statement failures and fallback to regular queries
  - Connection pool exhaustion scenarios (utilization > 100%)
  - Health check failure handling and error accumulation
  - Background task cancellation during shutdown
  - Concurrent session access and thread safety
  - Very large query results and memory management
  - Database version parsing with various PostgreSQL/TimescaleDB versions

- Error Handling:
  - DatabaseConnectionError propagation with proper context (connection_string, cause)
  - DatabaseQueryError handling with query and parameter details
  - SQLAlchemy specific error handling (OperationalError, DisconnectionError, SQLTimeoutError)
  - Connection retry exhaustion and failure reporting
  - Session rollback failures during error recovery
  - Health check failures with proper error categorization
  - File operation errors in execute_sql_file()
  - TimescaleDB version parsing failures and graceful fallback
  - Encryption key and password masking in connection strings
  - Query performance analysis errors and fallback behavior
  - Background task exception handling and recovery
  - Resource cleanup failures during shutdown

- Coverage Target: 85%+

### src/data/validation/event_validator.py - Event Validation Logic
**Classes Found:** ValidationRule, ValidationError, ValidationResult, SecurityValidator, SchemaValidator, IntegrityValidator, PerformanceValidator, ComprehensiveEventValidator
**Methods Analyzed:** ValidationRule.__init__(), ValidationError.__init__(), ValidationResult.__init__(), ValidationResult.has_errors, ValidationResult.has_warnings, ValidationResult.has_security_issues, SecurityValidator.__init__(), SecurityValidator.validate_input_security(), SecurityValidator.sanitize_input(), SchemaValidator.__init__(), SchemaValidator.validate_sensor_event_schema(), SchemaValidator.validate_room_configuration(), IntegrityValidator.__init__(), IntegrityValidator.calculate_event_hash(), IntegrityValidator.validate_data_consistency(), IntegrityValidator.validate_cross_system_consistency(), PerformanceValidator.__init__(), PerformanceValidator.bulk_validate_events(), PerformanceValidator._validate_batch(), PerformanceValidator._validate_single_event(), PerformanceValidator.get_performance_stats(), ComprehensiveEventValidator.__init__(), ComprehensiveEventValidator._initialize_validation_rules(), ComprehensiveEventValidator.validate_event(), ComprehensiveEventValidator.validate_events_bulk(), ComprehensiveEventValidator.validate_room_events(), ComprehensiveEventValidator.sanitize_event_data(), ComprehensiveEventValidator.get_validation_summary()

**Required Tests:**

**Unit Tests:**
- **ValidationRule Dataclass Tests:**
  - Test initialization with all required fields (rule_id, name, description, severity)
  - Test default values (enabled=True, metadata=empty dict)
  - Test ErrorSeverity enum integration and validation
  - Test metadata field handling with various dictionary structures
  - Test field modification and immutability considerations

- **ValidationError Dataclass Tests:**
  - Test initialization with required fields (rule_id, field, value, message, severity)
  - Test optional fields (suggestion=None, context=empty dict)
  - Test value field with various data types (str, int, dict, None, complex objects)
  - Test context dictionary with nested structures and metadata
  - Test suggestion field with helpful error recovery information

- **ValidationResult Dataclass Tests:**
  - Test initialization with is_valid boolean and default lists/values
  - Test property methods: has_errors, has_warnings, has_security_issues
  - Test validation_id auto-generation with UUID4
  - Test integrity_hash handling (None vs actual hash values)
  - Test processing_time_ms timing accuracy
  - Test confidence_score calculation and bounds (0.0-1.0)
  - Test errors/warnings list management and iteration

- **SecurityValidator Initialization Tests:**
  - Test __init__() compiling all regex patterns (SQL injection, XSS, path traversal)
  - Test compiled_patterns dictionary structure and accessibility
  - Test regex pattern compilation with re.IGNORECASE flags
  - Test pattern organization by security threat type
  - Test SQL_INJECTION_PATTERNS completeness (union, drop, delete, insert, update, exec, script injection)
  - Test XSS_PATTERNS completeness (script tags, javascript, event handlers, iframe, object)
  - Test PATH_TRAVERSAL_PATTERNS completeness (../, ..\, URL encoding, system paths)

- **Security Input Validation Tests:**
  - Test validate_input_security() with clean inputs (no security issues)
  - Test SQL injection detection with various attack patterns
  - Test XSS detection with script tags, javascript URLs, event handlers
  - Test path traversal detection with directory navigation attempts
  - Test input length validation (10000 character limit - SEC004 rule)
  - Test null byte injection detection (\x00 characters - SEC005 rule)
  - Test non-string input handling (int, float, bool conversion to string)
  - Test error severity assignment (CRITICAL for SQL, HIGH for XSS/path traversal)
  - Test error context inclusion (pattern_matched, input_length)
  - Test error suggestion generation for security remediation

- **Input Sanitization Tests:**
  - Test sanitize_input() standard mode with HTML/XML escaping
  - Test sanitize_input() aggressive mode with dangerous character removal
  - Test null byte removal (\x00 characters)
  - Test HTML entity encoding (&, <, >, ", ')
  - Test aggressive SQL keyword removal (SELECT, DELETE, UPDATE, etc.)
  - Test aggressive JavaScript term removal (script, alert, eval, etc.)
  - Test non-string input conversion before sanitization
  - Test whitespace trimming in output
  - Test character replacement vs removal strategies

- **SchemaValidator Initialization Tests:**
  - Test __init__() with SystemConfig parameter
  - Test room_configs dictionary extraction from SystemConfig.rooms
  - Test config access and room configuration mapping

- **Schema Validation Tests:**
  - Test validate_sensor_event_schema() with complete valid events
  - Test required field validation (room_id, sensor_id, sensor_type, state, timestamp)
  - Test missing field detection (SCH001 rule) with None values (SCH002 rule)
  - Test field type validation (room_id/sensor_id as strings - SCH003/SCH004 rules)
  - Test sensor_type validation against SensorType enum values (SCH005 rule)
  - Test state validation against SensorState enum values (SCH006 rule)
  - Test timestamp format validation (ISO format with timezone - SCH007/SCH008 rules)
  - Test attributes field validation (dictionary type - SCH009 rule)
  - Test error severity assignment and suggestion generation
  - Test datetime object timezone validation

- **Room Configuration Validation Tests:**
  - Test validate_room_configuration() with valid room data
  - Test room_id format validation (alphanumeric, underscore, hyphen - SCH010 rule)
  - Test room name validation (string type - SCH011, non-empty - SCH012 rules)
  - Test error generation for invalid room configurations
  - Test room data structure handling and validation

- **IntegrityValidator Initialization Tests:**
  - Test __init__() with AsyncSession parameter
  - Test session assignment and database connectivity setup

- **Event Hash Calculation Tests:**
  - Test calculate_event_hash() with complete event data
  - Test hash consistency with identical events
  - Test hash uniqueness with different events
  - Test sorted key handling for consistent hashing
  - Test None value filtering in hash calculation
  - Test datetime ISO format conversion in hashing
  - Test SHA-256 hash generation and format
  - Test large event data hash performance

- **Data Consistency Validation Tests:**
  - Test validate_data_consistency() with clean event lists
  - Test duplicate event detection using event hashes (INT001 rule)
  - Test timestamp ordering validation and analysis
  - Test MIN_EVENT_SEPARATION enforcement (INT002 rule)
  - Test event sequence integrity checking
  - Test empty event list handling
  - Test timestamp parsing and normalization
  - Test error context inclusion (event_index, event_indices)

- **Cross-System Consistency Tests:**
  - Test validate_cross_system_consistency() with database integration
  - Test RoomState query execution and recent state fetching
  - Test state transition logic validation
  - Test PRESENCE_STATES integration and transition checking
  - Test rapid state transition detection (60-second threshold - INT003 rule)
  - Test database error handling (INT004 rule)
  - Test timestamp comparison and delta calculations
  - Test occupancy state transition validation

- **PerformanceValidator Initialization Tests:**
  - Test __init__() with batch_size parameter (default 1000)
  - Test validation_stats defaultdict initialization
  - Test batch processing configuration

- **Bulk Validation Tests:**
  - Test bulk_validate_events() with large event lists
  - Test batch processing with configurable batch_size
  - Test parallel validation using asyncio.gather()
  - Test memory management during bulk processing
  - Test statistics tracking (batches_processed, events_processed)
  - Test progress monitoring and performance metrics
  - Test exception handling in bulk operations

- **Single Event Validation Tests:**
  - Test _validate_single_event() comprehensive validation pipeline
  - Test schema validation integration
  - Test security validation for all event fields
  - Test error/warning classification by severity
  - Test security flags generation for various threat levels
  - Test confidence score calculation based on errors/warnings
  - Test processing time measurement using asyncio event loop timing
  - Test integrity hash calculation for validation results
  - Test exception handling during validation process (VAL001 rule)

- **Performance Statistics Tests:**
  - Test get_performance_stats() dictionary conversion
  - Test validation statistics tracking and accuracy
  - Test performance metrics collection and reporting

- **ComprehensiveEventValidator Initialization Tests:**
  - Test __init__() with session and batch_size parameters
  - Test get_config() integration and system configuration loading
  - Test component validator initialization (security, schema, integrity, performance)
  - Test validation rules initialization and configuration

- **Validation Rules Management Tests:**
  - Test _initialize_validation_rules() with predefined rules
  - Test ValidationRule creation for security rules (SEC001, SEC002)
  - Test ValidationRule creation for schema rules (SCH001)
  - Test ValidationRule creation for integrity rules (INT001)
  - Test rule severity assignment and categorization

- **Single Event Validation Integration Tests:**
  - Test validate_event() end-to-end pipeline
  - Test performance validator integration for single events
  - Test complete validation result generation

- **Bulk Event Validation Integration Tests:**
  - Test validate_events_bulk() with mixed valid/invalid events
  - Test integrity validation integration with bulk processing
  - Test error aggregation and result combination
  - Test first result error accumulation from integrity checks

- **Room-Specific Validation Tests:**
  - Test validate_room_events() with room-specific logic
  - Test cross-system consistency integration
  - Test error handling for cross-system validation failures
  - Test room-specific validation result aggregation

- **Event Data Sanitization Tests:**
  - Test sanitize_event_data() with various event structures
  - Test field-by-field sanitization (string fields only)
  - Test nested dictionary sanitization in attributes
  - Test aggressive vs standard sanitization modes
  - Test non-string field preservation during sanitization

- **Validation Summary Tests:**
  - Test get_validation_summary() comprehensive statistics
  - Test validation rules counting (total vs active)
  - Test performance statistics inclusion
  - Test security pattern statistics (SQL injection, XSS, path traversal counts)

**Integration Tests:**
- **Database Integration:**
  - Test IntegrityValidator with real AsyncSession and database queries
  - Test RoomState integration and cross-system consistency validation
  - Test database error handling and recovery scenarios
  - Test TimescaleDB-specific query performance and optimization

- **Configuration Integration:**
  - Test SchemaValidator with real SystemConfig and room configurations
  - Test room configuration validation with complex sensor setups
  - Test entity ID validation against actual room configurations

- **Security Integration:**
  - Test complete security validation pipeline with real attack patterns
  - Test sanitization effectiveness against actual security threats
  - Test pattern matching accuracy with edge case inputs

- **Performance Integration:**
  - Test bulk validation performance with realistic event volumes (1000+ events)
  - Test memory usage during large-scale validation operations
  - Test async processing efficiency and task scheduling

- **End-to-End Validation:**
  - Test complete event validation from raw input to final ValidationResult
  - Test error aggregation across all validation layers
  - Test validation result consistency and accuracy

**Edge Cases:**
- **Input Data Edge Cases:**
  - Test validation with extremely large event payloads
  - Test validation with deeply nested event attributes
  - Test validation with Unicode characters and special encodings
  - Test validation with null/None values in various combinations
  - Test validation with malformed timestamp formats
  - Test validation with circular references in event data

- **Security Edge Cases:**
  - Test sophisticated SQL injection attempts with encoding
  - Test XSS attempts with various encoding techniques
  - Test path traversal with URL encoding and mixed separators
  - Test input length exactly at boundary values (10000 characters)
  - Test null byte injection in various positions
  - Test combined attack vectors in single inputs

- **Database Edge Cases:**
  - Test cross-system validation with empty database tables
  - Test validation with database connection interruptions
  - Test validation with very large result sets from database queries
  - Test validation with database query timeouts

- **Performance Edge Cases:**
  - Test bulk validation with single-element batches
  - Test validation with extremely large batch sizes
  - Test concurrent validation operations and thread safety
  - Test memory exhaustion scenarios with large event sets
  - Test validation with mixed valid/invalid event ratios

**Error Handling:**
- **Validation Process Errors:**
  - Test comprehensive error handling in _validate_single_event() (VAL001 rule)
  - Test exception isolation between validation layers
  - Test graceful degradation when validation components fail
  - Test error context preservation and propagation

- **Database Errors:**
  - Test database connection failures during cross-system validation
  - Test SQL query errors and timeout handling
  - Test database transaction failures and rollback scenarios
  - Test database schema inconsistencies and error reporting

- **Security Validation Errors:**
  - Test regex compilation failures and fallback behavior
  - Test pattern matching errors with extreme inputs
  - Test sanitization failures and data corruption prevention

- **Schema Validation Errors:**
  - Test enum validation failures and error reporting
  - Test type conversion errors and validation failures
  - Test configuration access errors and fallback handling

- **Performance Errors:**
  - Test batch processing failures and recovery
  - Test async task failures and exception propagation
  - Test memory allocation failures during bulk operations
  - Test timeout handling in validation operations

**Coverage Target: 85%+

### src/data/ingestion/bulk_importer.py - Historical Data Import
**Classes Found:** ImportProgress (dataclass), ImportConfig (dataclass), BulkImporter
**Methods Analyzed:** ImportProgress.duration_seconds, ImportProgress.entity_progress_percent, ImportProgress.event_progress_percent, ImportProgress.events_per_second, ImportProgress.to_dict, BulkImporter.__init__, BulkImporter.import_historical_data, BulkImporter._initialize_components, BulkImporter._cleanup_components, BulkImporter._load_resume_data, BulkImporter._save_resume_data, BulkImporter._estimate_total_events, BulkImporter._process_entities_batch, BulkImporter._process_entity_with_semaphore, BulkImporter._process_single_entity, BulkImporter._process_history_chunk, BulkImporter._convert_history_record_to_ha_event, BulkImporter._convert_ha_events_to_sensor_events, BulkImporter._determine_sensor_type, BulkImporter._bulk_insert_events, BulkImporter._update_progress, BulkImporter._generate_import_report, BulkImporter.get_import_stats, BulkImporter.validate_data_sufficiency, BulkImporter.optimize_import_performance, BulkImporter.verify_import_integrity, BulkImporter.create_import_checkpoint, BulkImporter._generate_sufficiency_recommendation

**Required Tests:**

**Unit Tests:**
- **ImportProgress Dataclass Tests:**
  - Test default initialization with start_time field_factory (datetime.utcnow)
  - Test duration_seconds property calculation from start_time to utcnow
  - Test entity_progress_percent calculation (processed/total * 100) with division by zero handling
  - Test event_progress_percent calculation with zero total events edge case
  - Test events_per_second calculation with zero duration handling
  - Test to_dict() serialization with datetime.isoformat() conversion and error slicing ([-10:])
  - Test property calculations with various progress states
  - Test List[str] errors field management and append behavior

- **ImportConfig Dataclass Tests:**
  - Test default parameter values (months_to_import=6, batch_size=1000, entity_batch_size=10, max_concurrent_entities=3, chunk_days=7)
  - Test optional field handling (resume_file=None, progress_callback=None)
  - Test boolean field defaults (skip_existing=True, validate_events=True, store_raw_data=False)
  - Test progress_callback callable type validation

- **BulkImporter Initialization Tests:**
  - Test __init__ with config parameter and default get_config() loading
  - Test import_config parameter handling with default ImportConfig() creation
  - Test component initialization (ha_client=None, event_processor=None)
  - Test progress tracking initialization (ImportProgress instance)
  - Test resume data initialization (_resume_data dict, _completed_entities set)
  - Test statistics initialization with all counter keys (entities_processed, events_imported, etc.)

- **Component Management Tests:**
  - Test _initialize_components() HomeAssistantClient creation and connect() call
  - Test _initialize_components() EventProcessor creation with config
  - Test _cleanup_components() proper ha_client.disconnect() call
  - Test _cleanup_components() with None client handling
  - Test component initialization logging

- **Resume Functionality Tests:**
  - Test _load_resume_data() with missing resume_file (early return)
  - Test _load_resume_data() with non-existent file path (Path.exists() false)
  - Test _load_resume_data() pickle loading with proper deserialization
  - Test _load_resume_data() _completed_entities set restoration from list
  - Test _load_resume_data() exception handling with warning logging
  - Test _save_resume_data() data structure creation with all required fields
  - Test _save_resume_data() Path operations (parent.mkdir, pickle.dump)
  - Test _save_resume_data() exception handling and warning logging

- **Event Estimation Tests:**
  - Test _estimate_total_events() sample size calculation (min(5, len(entity_ids)))
  - Test _estimate_total_events() one-day sampling with start/end date calculation
  - Test _estimate_total_events() ha_client.get_entity_history() integration
  - Test _estimate_total_events() HomeAssistantError handling with stats increment
  - Test _estimate_total_events() generic exception handling and HomeAssistantError conversion
  - Test _estimate_total_events() total events calculation (avg * entities * days)
  - Test _estimate_total_events() progress.total_events assignment

- **Entity Processing Tests:**
  - Test _process_entities_batch() completed entities filtering
  - Test _process_entities_batch() early return with all entities completed
  - Test _process_entities_batch() batch creation with entity_batch_size
  - Test _process_entities_batch() asyncio.Semaphore creation with max_concurrent_entities
  - Test _process_entities_batch() asyncio.gather() with return_exceptions=True
  - Test _process_entities_batch() progress tracking and update calls
  - Test _process_entity_with_semaphore() semaphore acquire/release pattern
  - Test _process_entity_with_semaphore() _process_single_entity() delegation

- **Single Entity Processing Tests:**
  - Test _process_single_entity() current_entity progress assignment
  - Test _process_single_entity() time chunking with chunk_days configuration
  - Test _process_single_entity() while loop iteration with chunk_end calculation
  - Test _process_single_entity() ha_client.get_entity_history() call per chunk
  - Test _process_single_entity() _process_history_chunk() delegation
  - Test _process_single_entity() progress.processed_events increment
  - Test _process_single_entity() error handling with progress.errors append
  - Test _process_single_entity() _completed_entities set management
  - Test _process_single_entity() asyncio.sleep(0.01) yield control
  - Test _process_single_entity() statistics tracking (entities_processed increment)

- **History Chunk Processing Tests:**
  - Test _process_history_chunk() empty history_data handling (return 0)
  - Test _process_history_chunk() _convert_history_record_to_ha_event() conversion loop
  - Test _process_history_chunk() DataValidationError handling with debug logging and stats increment
  - Test _process_history_chunk() generic exception conversion to DataValidationError
  - Test _process_history_chunk() event_processor.process_event_batch() when validate_events=True
  - Test _process_history_chunk() _convert_ha_events_to_sensor_events() when validate_events=False
  - Test _process_history_chunk() _bulk_insert_events() call and return count
  - Test _process_history_chunk() comprehensive exception handling (DataValidationError, DatabaseError, HomeAssistantError, generic)
  - Test _process_history_chunk() statistics increment for different error types
  - Test _process_history_chunk() traceback logging for debug information

- **Data Conversion Tests:**
  - Test _convert_history_record_to_ha_event() timestamp extraction (last_changed, last_updated fallback)
  - Test _convert_history_record_to_ha_event() timestamp parsing with fromisoformat() and Z suffix replacement
  - Test _convert_history_record_to_ha_event() HAEvent creation with all fields (entity_id, state, timestamp, attributes)
  - Test _convert_history_record_to_ha_event() exception handling with None return
  - Test _convert_ha_events_to_sensor_events() room configuration lookup
  - Test _convert_ha_events_to_sensor_events() sensor type determination
  - Test _convert_ha_events_to_sensor_events() SensorEvent creation with default values (is_human_triggered=True, created_at=utcnow)
  - Test _determine_sensor_type() room_config.sensors dictionary lookup
  - Test _determine_sensor_type() entity_id pattern matching fallback ("presence", "motion", "door", "temperature", "light")
  - Test _determine_sensor_type() default "motion" return

- **Database Operations Tests:**
  - Test _bulk_insert_events() empty events handling (return 0)
  - Test _bulk_insert_events() get_db_session() context manager usage
  - Test _bulk_insert_events() insert_data preparation with all SensorEvent fields
  - Test _bulk_insert_events() confidence_score getattr() with default None
  - Test _bulk_insert_events() large batch handling (> batch_size) with get_bulk_insert_query()
  - Test _bulk_insert_events() small batch handling with PostgreSQL INSERT ... ON CONFLICT
  - Test _bulk_insert_events() skip_existing configuration with on_conflict_do_nothing()
  - Test _bulk_insert_events() session.commit() and rowcount return
  - Test _bulk_insert_events() exception handling with DatabaseError raising

- **Progress Management Tests:**
  - Test _update_progress() last_update timestamp assignment
  - Test _update_progress() progress_callback invocation (sync and async)
  - Test _update_progress() callback exception handling with warning logging
  - Test _update_progress() periodic logging (every 10 entities)
  - Test _update_progress() progress percentage and events per second logging

- **Reporting and Statistics Tests:**
  - Test _generate_import_report() comprehensive report structure creation
  - Test _generate_import_report() import_summary with duration and throughput metrics
  - Test _generate_import_report() error_summary with all error type counts
  - Test _generate_import_report() data_quality success_rate and error_rate calculations
  - Test _generate_import_report() JSON logging with indent=2
  - Test _generate_import_report() optional report file saving
  - Test get_import_stats() current progress and statistics dictionary return

- **Data Validation Tests:**
  - Test validate_data_sufficiency() SQL query construction with room_id parameter
  - Test validate_data_sufficiency() daily event count aggregation and analysis
  - Test validate_data_sufficiency() InsufficientTrainingDataError raising with detailed parameters
  - Test validate_data_sufficiency() sufficiency analysis (days and events thresholds)
  - Test validate_data_sufficiency() recommendation generation based on analysis results
  - Test validate_data_sufficiency() database error handling and exception conversion
  - Test _generate_sufficiency_recommendation() message generation based on conditions

- **Performance Optimization Tests:**
  - Test optimize_import_performance() current settings reporting
  - Test optimize_import_performance() performance metrics calculation
  - Test optimize_import_performance() throughput-based optimization suggestions (<50 events/sec, >200 events/sec)
  - Test optimize_import_performance() concurrency optimization suggestions
  - Test optimize_import_performance() psutil memory usage monitoring
  - Test optimize_import_performance() memory usage recommendations (>500MB threshold)

- **Integrity Verification Tests:**
  - Test verify_import_integrity() temporal consistency SQL query with sampling
  - Test verify_import_integrity() statistical analysis of data quality issues
  - Test verify_import_integrity() integrity score calculation (1 - issues/total)
  - Test verify_import_integrity() recommendations based on integrity score thresholds
  - Test verify_import_integrity() exception handling with issues_found append

- **Checkpoint Management Tests:**
  - Test create_import_checkpoint() checkpoint data structure creation
  - Test create_import_checkpoint() timestamp and configuration capture
  - Test create_import_checkpoint() JSON file writing with datetime formatting
  - Test create_import_checkpoint() exception handling with False return

**Integration Tests:**
- **End-to-End Import Workflow:**
  - Test complete import_historical_data() workflow with date range calculation
  - Test entity_ids resolution from config.get_all_entity_ids()
  - Test component initialization and cleanup integration
  - Test resume data loading and saving across import sessions
  - Test progress tracking throughout entire import process
  - Test final report generation with realistic statistics

- **Home Assistant Client Integration:**
  - Test ha_client connection and authentication in _initialize_components()
  - Test entity history fetching with various time ranges and chunk sizes
  - Test rate limiting and error handling during bulk historical data requests
  - Test websocket cleanup during _cleanup_components()

- **Database Integration:**
  - Test TimescaleDB bulk insert operations with large datasets
  - Test PostgreSQL-specific INSERT ... ON CONFLICT behavior
  - Test database session management and transaction handling
  - Test bulk insert performance with various batch sizes
  - Test data sufficiency validation with real database queries

- **Event Processing Integration:**
  - Test event_processor.process_event_batch() integration when validate_events=True
  - Test validation bypass path when validate_events=False
  - Test event conversion from Home Assistant format to SensorEvent models
  - Test movement classification integration during event processing

- **Configuration Integration:**
  - Test SystemConfig integration for entity_ids and room configuration
  - Test ImportConfig parameter usage throughout import process
  - Test room-entity mapping for sensor type determination
  - Test config-based timeout and batch size handling

**Edge Cases:**
- **Date and Time Edge Cases:**
  - Test import with zero months_to_import configuration
  - Test date range calculation with None start_date and end_date
  - Test timestamp parsing with various Home Assistant datetime formats
  - Test timezone handling in timestamp conversion
  - Test historical data import spanning daylight saving time changes
  - Test chunk boundary handling with exact chunk_days alignment

- **Data Volume Edge Cases:**
  - Test import with zero entities (empty entity_ids list)
  - Test import with single entity vs thousands of entities
  - Test very large batch sizes (>10000) and memory management
  - Test entities with no historical data available
  - Test entities with millions of historical events
  - Test chunk processing with very small chunk_days (1 day)

- **Configuration Edge Cases:**
  - Test ImportConfig with extreme parameter values (batch_size=1, max_concurrent_entities=100)
  - Test resume functionality with corrupted resume files
  - Test progress_callback with various callable types (sync, async, lambda)
  - Test configuration with missing or None fields

- **Network and API Edge Cases:**
  - Test Home Assistant API rate limiting scenarios
  - Test network disconnection during entity processing
  - Test Home Assistant historical data API pagination
  - Test API response with malformed historical data
  - Test concurrent entity processing with API timeouts

- **Database Edge Cases:**
  - Test bulk insert with duplicate timestamps and conflict resolution
  - Test database connection loss during import process
  - Test bulk insert with very large SensorEvent attribute JSON payloads
  - Test TimescaleDB hypertable behavior with historical timestamp ordering
  - Test database transaction rollback scenarios

- **Memory and Performance Edge Cases:**
  - Test import process memory usage monitoring
  - Test queue management with maximum concurrent entity limits
  - Test progress tracking with very frequent updates
  - Test statistics tracking with overflow scenarios
  - Test background task cancellation during shutdown

**Error Handling:**
- **Home Assistant API Errors:**
  - Test HomeAssistantConnectionError handling during component initialization
  - Test HomeAssistantAuthenticationError propagation from ha_client operations
  - Test HomeAssistantAPIError handling during entity history fetching
  - Test EntityNotFoundError handling for non-existent entities
  - Test API timeout errors and retry logic
  - Test rate limiting error handling with proper statistics tracking

- **Database Errors:**
  - Test DatabaseConnectionError during session creation
  - Test DatabaseError during bulk insert operations
  - Test database constraint violations during insert
  - Test database timeout errors during large batch operations
  - Test connection pool exhaustion scenarios
  - Test TimescaleDB-specific errors and fallback behavior

- **Data Processing Errors:**
  - Test DataValidationError handling during event conversion
  - Test JSON parsing errors in Home Assistant historical data
  - Test invalid timestamp format handling
  - Test missing required fields in historical records
  - Test character encoding issues in historical data
  - Test memory exhaustion during large data processing

- **File System Errors:**
  - Test resume file creation permission errors
  - Test resume file corruption and recovery
  - Test checkpoint file writing failures
  - Test disk space exhaustion during file operations
  - Test concurrent file access issues

- **Configuration and Validation Errors:**
  - Test InsufficientTrainingDataError scenarios with detailed error context
  - Test configuration validation failures
  - Test entity ID validation errors
  - Test room configuration lookup failures
  - Test sensor type determination fallback behavior

- **Concurrency and Resource Errors:**
  - Test asyncio.Semaphore exhaustion and queueing
  - Test asyncio task cancellation during entity processing
  - Test background task exception handling and recovery
  - Test resource cleanup failures during error conditions
  - Test progress callback execution failures

**Coverage Target: 85%+**

### src/data/validation/pattern_detector.py - Pattern Detection
**Classes Found:** PatternAnomaly, DataQualityMetrics, StatisticalPatternAnalyzer, CorruptionDetector, RealTimeQualityMonitor
**Methods Analyzed:** StatisticalPatternAnalyzer.__init__(), analyze_sensor_behavior(), _calculate_state_distribution(), _detect_statistical_anomalies(), detect_sensor_malfunction(), CorruptionDetector.__init__(), detect_data_corruption(), _detect_timestamp_corruption(), _detect_state_corruption(), _detect_id_corruption(), _detect_encoding_corruption(), RealTimeQualityMonitor.__init__(), calculate_quality_metrics(), _calculate_consistency_score(), _calculate_accuracy_score(), _calculate_timeliness_score(), get_quality_trends(), detect_quality_alerts()

**Required Tests:**
- Unit Tests: 
  - **PatternAnomaly Dataclass Tests:**
    - Test initialization with all required fields (anomaly_id, anomaly_type, description, severity, confidence, detected_at, affected_sensors)
    - Test optional fields handling (statistical_measures, context dictionaries with default_factory)
    - Test ErrorSeverity enum integration and field validation
    - Test datetime field handling and timezone awareness
    - Test affected_sensors list manipulation and validation
    
  - **DataQualityMetrics Dataclass Tests:**
    - Test initialization with quality score fields (completeness_score, consistency_score, accuracy_score, timeliness_score)
    - Test score validation (0-1 range enforcement)
    - Test anomaly_count integer field validation
    - Test corruption_indicators list handling
    - Test quality_trends dictionary with default_factory behavior
    
  - **StatisticalPatternAnalyzer Initialization Tests:**
    - Test __init__() with default parameters (window_size=100, confidence_level=0.95)
    - Test __init__() with custom window_size and confidence_level parameters
    - Test initialization of sensor_baselines defaultdict(dict) structure
    - Test pattern_cache dictionary initialization
    - Test parameter validation for window_size and confidence_level ranges
    
  - **Statistical Analysis Tests:**
    - Test analyze_sensor_behavior() with valid event lists containing timestamp and state fields
    - Test analyze_sensor_behavior() with empty events list (returns error dict)
    - Test analyze_sensor_behavior() with single event (insufficient data handling)
    - Test timestamp parsing with ISO format strings and datetime objects
    - Test timestamp parsing with Z suffix and +00:00 timezone handling
    - Test interval calculation between consecutive events (total_seconds())
    - Test statistical measures calculation (mean, median, standard deviation of intervals)
    - Test event frequency calculation (events per hour)
    - Test state distribution calculation via _calculate_state_distribution()
    - Test statistical anomaly detection with _detect_statistical_anomalies() for datasets >10 events
    - Test sensor baseline updates in sensor_baselines dictionary
    - Test exception handling for malformed timestamp formats
    
  - **State Distribution Calculation Tests:**
    - Test _calculate_state_distribution() with various state combinations
    - Test _calculate_state_distribution() with empty state list (returns empty dict)
    - Test state counting with defaultdict(int) and percentage calculation
    - Test state distribution accuracy with duplicate states
    - Test state distribution with single state (100% distribution)
    
  - **Statistical Anomaly Detection Tests:**
    - Test _detect_statistical_anomalies() with sufficient data (>3 intervals)
    - Test _detect_statistical_anomalies() with insufficient data (<3 intervals, returns basic structure)
    - Test z-score calculation using statistics.mean() and statistics.stdev()
    - Test outlier detection with z_threshold=2.5 standard deviations
    - Test outlier structure with index, value, and z_score fields
    - Test anomaly counting and z_scores list population
    - Test normality testing with stats.shapiro() when len(intervals) > 8
    - Test distribution_type classification ("normal", "non_normal", "unknown")
    - Test SCIPY_AVAILABLE fallback behavior with MockStats
    - Test exception handling in shapiro test with stats module fallback
    
  - **Sensor Malfunction Detection Tests:**
    - Test detect_sensor_malfunction() with baseline comparison logic
    - Test detect_sensor_malfunction() with empty recent_events (returns empty list)
    - Test detect_sensor_malfunction() without baseline data (returns empty list)
    - Test frequency anomaly detection (5x higher than normal, 5x lower than normal)
    - Test PatternAnomaly creation for high_frequency with proper severity (ErrorSeverity.HIGH)
    - Test PatternAnomaly creation for low_frequency with proper severity (ErrorSeverity.MEDIUM)
    - Test confidence calculation for frequency anomalies (min(0.95, freq_ratio/10))
    - Test interval pattern change detection (unstable_timing when std > 3x baseline)
    - Test statistical_measures dictionary population with frequency and stability ratios
    - Test anomaly_id generation with sensor_id formatting
    - Test affected_sensors list with single sensor_id
    - Test baseline_freq and current_freq zero handling
    
  - **CorruptionDetector Initialization Tests:**
    - Test __init__() with known_corrupt_patterns regex list initialization
    - Test regex pattern validation for null bytes, high ASCII, repeated characters, long numbers
    - Test pattern compilation and matching functionality
    
  - **Data Corruption Detection Tests:**
    - Test detect_data_corruption() orchestration calling all detection methods
    - Test detect_data_corruption() with empty events (returns empty errors list)
    - Test error aggregation from timestamp, state, ID, and encoding corruption detection
    - Test ValidationError list compilation and deduplication
    
  - **Timestamp Corruption Detection Tests:**
    - Test _detect_timestamp_corruption() with valid ISO timestamp strings
    - Test _detect_timestamp_corruption() with datetime objects
    - Test _detect_timestamp_corruption() with malformed timestamps (ValidationError COR001)
    - Test timestamp parsing with Z suffix replacement to +00:00
    - Test impossible time jumps detection (>365 days, ValidationError COR002)
    - Test duplicate timestamp detection (ValidationError COR003)
    - Test timestamp sorting and sequential comparison logic
    - Test ValidationError context with event indices
    - Test exception handling for timestamp parsing failures
    
  - **State Corruption Detection Tests:**
    - Test _detect_state_corruption() with normal state values
    - Test _detect_state_corruption() with suspiciously long states (>20 chars, ValidationError COR004)
    - Test _detect_state_corruption() with non-printable characters (ValidationError COR005)
    - Test character code validation (ord(c) < 32 or ord(c) > 126)
    - Test whitespace character exceptions (tab and newline allowed)
    - Test state length validation and truncation in error messages ([:50])
    - Test repr() formatting for non-printable character display
    
  - **ID Field Corruption Detection Tests:**
    - Test _detect_id_corruption() with valid room_id and sensor_id fields
    - Test _detect_id_corruption() with room_id corruption (>100 chars or control chars, ValidationError COR006)
    - Test _detect_id_corruption() with sensor_id corruption (>200 chars or control chars, ValidationError COR007)
    - Test control character detection (ord(c) < 32)
    - Test ID length validation and truncation ([:50] for display)
    - Test None handling for missing ID fields
    
  - **Encoding Corruption Detection Tests:**
    - Test _detect_encoding_corruption() with valid UTF-8 strings
    - Test _detect_encoding_corruption() with Unicode replacement character detection (ValidationError COR008)
    - Test _detect_encoding_corruption() with UnicodeEncodeError handling (ValidationError COR009)
    - Test encoding validation with value.encode("utf-8")
    - Test field iteration across all event string fields
    - Test error message truncation and repr() formatting
    
  - **RealTimeQualityMonitor Initialization Tests:**
    - Test __init__() with default window_minutes=60 parameter
    - Test __init__() with custom window_minutes parameter
    - Test quality_history deque initialization with maxlen=1000
    - Test sensor_quality defaultdict with lambda: deque(maxlen=100)
    - Test alert_thresholds dictionary with default threshold values
    - Test threshold validation (completeness=0.8, consistency=0.7, accuracy=0.75, timeliness=0.9)
    
  - **Quality Metrics Calculation Tests:**
    - Test calculate_quality_metrics() with comprehensive metrics calculation
    - Test calculate_quality_metrics() with empty events (returns zero scores)
    - Test completeness_score calculation using actual vs expected sensors
    - Test consistency_score calculation via _calculate_consistency_score()
    - Test accuracy_score calculation via _calculate_accuracy_score()
    - Test timeliness_score calculation via _calculate_timeliness_score()
    - Test anomaly counting using StatisticalPatternAnalyzer integration
    - Test corruption detection using CorruptionDetector integration
    - Test DataQualityMetrics object creation with all calculated scores
    - Test quality_history deque management and timestamp recording
    
  - **Consistency Score Calculation Tests:**
    - Test _calculate_consistency_score() with multiple sensors
    - Test _calculate_consistency_score() with single sensor (returns 1.0)
    - Test _calculate_consistency_score() with empty events (returns 0.0)
    - Test sensor event grouping using defaultdict(list)
    - Test timestamp parsing and sorting for interval calculation
    - Test coefficient of variation calculation (std_interval / mean_interval)
    - Test consistency score derivation (1.0 - min(cv, 1.0))
    - Test statistics.mean() and statistics.stdev() integration
    - Test exception handling for timestamp parsing failures
    
  - **Accuracy Score Calculation Tests:**
    - Test _calculate_accuracy_score() with valid sensor data
    - Test _calculate_accuracy_score() with empty events (returns 0.0)
    - Test state validity checking against SensorState enum values
    - Test sensor type validity checking against SensorType enum values
    - Test timestamp ordering validation for temporal accuracy
    - Test accuracy factor aggregation and statistics.mean() calculation
    - Test accuracy factor weighting (state, sensor type, ordering)
    
  - **Timeliness Score Calculation Tests:**
    - Test _calculate_timeliness_score() with recent events (high scores)
    - Test _calculate_timeliness_score() with old events (low scores)
    - Test _calculate_timeliness_score() with empty events (returns 0.0)
    - Test age-based scoring (< 1 hour: 1.0, < 6 hours: 0.8, < 24 hours: 0.6)
    - Test exponential decay for events > 24 hours old
    - Test future timestamp handling (score = 0.5)
    - Test timezone handling with UTC conversion
    - Test math.exp() exponential decay calculation
    - Test datetime.now(timezone.utc) current time comparison
    - Test exception handling for invalid timestamp formats (score = 0.1)
    
  - **Quality Trends Analysis Tests:**
    - Test get_quality_trends() with time-based filtering
    - Test get_quality_trends() with custom hours parameter (default=24)
    - Test quality_history filtering using datetime comparison
    - Test trend dictionary creation with all quality metrics
    - Test timestamp formatting to ISO format
    - Test recent_history filtering and data extraction
    
  - **Quality Alert Detection Tests:**
    - Test detect_quality_alerts() with threshold comparison
    - Test detect_quality_alerts() with various DataQualityMetrics input
    - Test completeness alert creation (QUAL001, ErrorSeverity.MEDIUM)
    - Test consistency alert creation (QUAL002, ErrorSeverity.MEDIUM)
    - Test accuracy alert creation (QUAL003, ErrorSeverity.HIGH)
    - Test timeliness alert creation (QUAL004, ErrorSeverity.MEDIUM)
    - Test alert threshold comparison logic
    - Test PatternAnomaly object creation for each alert type
    - Test statistical_measures population with quality scores
    - Test confidence score assignment for different alert types

- Integration Tests:
  - **Statistical Analysis Integration:**
    - Test complete sensor behavior analysis workflow with real sensor data
    - Test baseline establishment and comparison across multiple analysis cycles
    - Test sensor malfunction detection with realistic trigger patterns
    - Test scipy/sklearn integration when available vs fallback behavior
    
  - **Corruption Detection Integration:**
    - Test comprehensive corruption detection across all detection methods
    - Test ValidationError aggregation and prioritization
    - Test corruption detection with mixed valid and corrupted events
    - Test error reporting integration with logging system
    
  - **Quality Monitoring Integration:**
    - Test real-time quality monitoring with streaming sensor data
    - Test quality metrics calculation with expected sensor sets
    - Test alert generation and escalation workflows
    - Test quality trend analysis over extended time periods
    - Test integration with StatisticalPatternAnalyzer and CorruptionDetector
    
  - **Cross-Component Integration:**
    - Test pattern detection integration with event validation systems
    - Test quality metrics integration with monitoring and alerting systems
    - Test baseline management across system restarts and persistence
    - Test error propagation to upstream system components

- Edge Cases:
  - **Statistical Analysis Edge Cases:**
    - Test sensor behavior analysis with extreme interval variations
    - Test statistical calculations with very small datasets (<5 events)
    - Test timestamp edge cases (leap years, daylight saving time transitions)
    - Test sensor data with identical timestamps across multiple events
    - Test mathematical edge cases (division by zero in coefficient calculations)
    - Test memory management with very large event sequences
    - Test sensor_baselines growth with thousands of sensors
    
  - **Corruption Detection Edge Cases:**
    - Test timestamp corruption with edge timezone formats
    - Test state corruption with Unicode edge cases and emoji characters
    - Test ID corruption with very long strings and international characters
    - Test encoding corruption with mixed character sets
    - Test corruption detection with partially corrupted events
    - Test regex pattern matching with extreme string lengths
    
  - **Quality Monitoring Edge Cases:**
    - Test quality calculation with empty expected sensor sets
    - Test quality monitoring with sensors that never report
    - Test quality trends with sparse historical data
    - Test alert detection at exact threshold boundaries
    - Test deque overflow behavior with quality_history and sensor_quality
    - Test concurrent access to quality monitoring data structures
    
  - **Performance Edge Cases:**
    - Test pattern detection performance with very large event datasets
    - Test memory usage with extensive sensor baselines and history
    - Test statistical calculations with extreme outlier values
    - Test corruption detection performance with complex regex patterns

- Error Handling:
  - **Statistical Analysis Errors:**
    - Test exception handling in timestamp parsing across all methods
    - Test mathematical errors in statistical calculations (NaN, infinity)
    - Test memory errors with large statistical datasets
    - Test scipy/sklearn import failures and graceful fallback
    - Test baseline corruption recovery and re-establishment
    
  - **Corruption Detection Errors:**
    - Test ValidationError creation failures and error recovery
    - Test regex pattern matching failures with malformed patterns
    - Test encoding detection failures with unusual character sets
    - Test timestamp parsing edge cases and error accumulation
    - Test error context preservation across detection methods
    
  - **Quality Monitoring Errors:**
    - Test quality calculation failures with invalid sensor data
    - Test alert generation failures and error recovery
    - Test deque operations failures and data consistency
    - Test trend analysis failures with corrupted history data
    - Test integration failures with external analyzer components
    - Test threshold configuration errors and validation
    
  - **Resource Management Errors:**
    - Test memory management with pattern cache overflow
    - Test disk space issues with quality history persistence
    - Test concurrent access errors and thread safety issues
    - Test cleanup failures during system shutdown

- Coverage Target: 85%+

### src/data/storage/database_compatibility.py - Database Compatibility Layer
**Classes Found:** No classes - utility functions only
**Methods Analyzed:** is_sqlite_engine(), is_postgresql_engine(), configure_sensor_event_model(), create_database_specific_models(), configure_sqlite_for_testing(), configure_database_on_first_connect(), patch_models_for_sqlite_compatibility(), get_database_specific_table_args()

**Required Tests:**
- Unit Tests:
  - **is_sqlite_engine()**: Test detection with various URL formats (sqlite:///, file://, memory databases, uppercase/lowercase, URL objects vs strings)
  - **is_postgresql_engine()**: Test detection with various PostgreSQL URL formats (postgresql://, postgres://, URL objects vs strings, different host/port combinations)
  - **configure_sensor_event_model()**: Test SQLite configuration (autoincrement setup, primary key modifications, unique constraints), PostgreSQL configuration (composite primary keys, column modifications)
  - **create_database_specific_models()**: Test SensorEvent model configuration, test passthrough for other model types, test with empty and populated model dictionaries
  - **patch_models_for_sqlite_compatibility()**: Test model patching behavior, test with existing table attributes, test primary key and autoincrement modifications
  - **get_database_specific_table_args()**: Test SQLite table arguments generation, test PostgreSQL table arguments generation, test index creation with proper names and configurations

- Integration Tests:
  - **SQLAlchemy Event Listeners**: Test configure_sqlite_for_testing() with real database connections, test configure_database_on_first_connect() with both SQLite and PostgreSQL
  - **Model Configuration Integration**: Test complete model setup workflow with SQLite engine, test complete model setup workflow with PostgreSQL engine
  - **Database Pragma Settings**: Test SQLite PRAGMA execution (foreign_keys, journal_mode, synchronous), verify settings actually applied to connection
  - **Index Creation**: Test that generated indexes are properly created in database, verify PostgreSQL-specific index features (partial indexes, USING clause)

- Edge Cases:
  - **URL Format Variations**: Test with malformed URLs, empty URLs, None values, URLs with special characters or encoding
  - **Engine Object Variations**: Test with different SQLAlchemy engine types, test with mock engine objects, test with engines lacking URL attribute
  - **Model Table Scenarios**: Test models without __table__ attribute, test models with corrupted table metadata, test models with pre-existing constraints
  - **Database Connection Edge Cases**: Test connection failures during event listener execution, test with read-only database connections
  - **Constraint Naming Conflicts**: Test unique constraint creation with existing constraints, test index creation with name conflicts

- Error Handling:
  - **Database Import Errors**: Test behavior when src.data.storage.models import fails, test graceful degradation when SensorEvent model unavailable
  - **SQLAlchemy Operation Errors**: Test constraint addition failures, test primary key modification failures, test when table columns don't exist
  - **PRAGMA Execution Errors**: Test SQLite PRAGMA command failures, test cursor operation failures, test database connection errors during configuration
  - **Model Configuration Errors**: Test when model class modification fails, test when autoincrement setting fails, test when constraint objects cannot be created
  - **URL Parsing Errors**: Test with invalid URL formats that cause string conversion errors, test with URL objects that don't support string conversion

- Coverage Target: 85%+

### src/data/storage/dialect_utils.py - Database Dialect Utilities
**Classes Found:** DatabaseDialectUtils, StatisticalFunctions, QueryBuilder, CompatibilityManager
**Methods Analyzed:** get_dialect_name, is_postgresql, is_sqlite, percentile_cont, _sqlite_median, _sqlite_quartile, _sqlite_percentile_approx, stddev_samp, extract_epoch_from_interval, build_percentile_query, build_statistics_query, get_instance, initialize, get_compatibility_manager, percentile_cont (global), stddev_samp (global), extract_epoch_interval (global)

**Required Tests:**
- Unit Tests:
  - **DatabaseDialectUtils Class:**
    - Test get_dialect_name() with PostgreSQL engine (should return "postgresql")
    - Test get_dialect_name() with SQLite engine (should return "sqlite")
    - Test get_dialect_name() with mock engines having different dialect names
    - Test is_postgresql() returns True for PostgreSQL engines, False for others
    - Test is_sqlite() returns True for SQLite engines, False for others
    - Test both Engine and AsyncEngine types for all dialect detection methods

  - **StatisticalFunctions Class:**
    - Test percentile_cont() with PostgreSQL engine using different percentiles (0.25, 0.5, 0.75, 0.9)
    - Test percentile_cont() with PostgreSQL engine with order_desc=True and order_desc=False
    - Test percentile_cont() with SQLite engine for median (0.5) calling _sqlite_median()
    - Test percentile_cont() with SQLite engine for quartiles (0.25, 0.75) calling _sqlite_quartile()
    - Test percentile_cont() with SQLite engine for other percentiles calling _sqlite_percentile_approx()
    - Test _sqlite_median() returns sql_func.avg(column)
    - Test _sqlite_quartile() with 0.25 percentile returns min + 25% of range
    - Test _sqlite_quartile() with 0.75 percentile returns min + 75% of range
    - Test _sqlite_quartile() with invalid percentile returns avg
    - Test _sqlite_percentile_approx() returns linear interpolation between min and max
    - Test stddev_samp() with PostgreSQL engine returns sql_func.stddev_samp()
    - Test stddev_samp() with SQLite engine returns sqrt approximation formula
    - Test extract_epoch_from_interval() with PostgreSQL returns extract("epoch", interval)
    - Test extract_epoch_from_interval() with SQLite returns strftime difference calculation

  - **QueryBuilder Class:**
    - Test __init__() stores engine and gets dialect name correctly
    - Test build_percentile_query() with single percentile creates correct labeled expression
    - Test build_percentile_query() with multiple percentiles (e.g., [0.25, 0.5, 0.75])
    - Test build_percentile_query() with order_desc=True passes parameter correctly
    - Test build_statistics_query() includes count, mean, min, max, stddev expressions
    - Test build_statistics_query() with include_percentiles=True adds q1, median, q3
    - Test build_statistics_query() with include_percentiles=False excludes percentiles
    - Test both methods return Select objects with correct column expressions

  - **CompatibilityManager Class:**
    - Test __init__() initializes all components and sets class variables
    - Test get_instance() returns same instance when initialized
    - Test get_instance() raises RuntimeError when not initialized
    - Test initialize() creates new instance and returns it
    - Test initialize() sets singleton instance correctly
    - Test is_postgresql() delegates to DatabaseDialectUtils.is_postgresql()
    - Test is_sqlite() delegates to DatabaseDialectUtils.is_sqlite()
    - Test get_dialect_name() delegates to DatabaseDialectUtils.get_dialect_name()

  - **Global Utility Functions:**
    - Test get_compatibility_manager() returns CompatibilityManager.get_instance()
    - Test percentile_cont() with engine parameter delegates to StatisticalFunctions
    - Test percentile_cont() without engine uses compatibility manager engine
    - Test percentile_cont() with RuntimeError fallback for median (0.5)
    - Test percentile_cont() with RuntimeError fallback for quartiles (0.25, 0.75)
    - Test percentile_cont() with RuntimeError fallback for other percentiles
    - Test stddev_samp() with engine parameter delegates to StatisticalFunctions
    - Test stddev_samp() without engine uses compatibility manager engine
    - Test stddev_samp() with RuntimeError fallback returns approximation
    - Test extract_epoch_interval() with engine parameter delegates to StatisticalFunctions
    - Test extract_epoch_interval() without engine uses compatibility manager
    - Test extract_epoch_interval() with RuntimeError fallback uses SQLite approach

- Integration Tests:
  - **Cross-Database Compatibility:**
    - Test StatisticalFunctions.percentile_cont() with real PostgreSQL connection produces correct percentile values
    - Test StatisticalFunctions.percentile_cont() with real SQLite connection produces reasonable approximations
    - Test stddev_samp() with real databases produces mathematically valid standard deviations
    - Test extract_epoch_from_interval() with real databases produces correct time differences
    - Test QueryBuilder with real databases generates executable queries
    - Test CompatibilityManager with different database engines maintains correct state

  - **End-to-End Query Generation:**
    - Test build_percentile_query() generates executable PostgreSQL queries
    - Test build_percentile_query() generates executable SQLite queries  
    - Test build_statistics_query() generates executable queries on both database types
    - Test generated queries return expected result structure and data types
    - Test complex queries with multiple statistical functions work correctly

- Edge Cases:
  - **Parameter Validation:**
    - Test percentile_cont() with percentile values at boundaries (0.0, 1.0)
    - Test percentile_cont() with invalid percentile values (<0.0, >1.0)
    - Test percentile_cont() with very precise percentile values (0.123456789)
    - Test functions with None engine parameters
    - Test functions with mock engines having unknown dialect names
    - Test QueryBuilder with empty percentile lists
    - Test statistical functions with null/empty column parameters

  - **SQLite Approximation Accuracy:**
    - Test _sqlite_quartile() edge cases where min equals max (zero range)
    - Test _sqlite_percentile_approx() with extreme percentile values (0.001, 0.999)
    - Test _sqlite_median() behavior with even vs odd number of records
    - Test stddev_samp() SQLite approximation with datasets having zero variance
    - Test mathematical edge cases in sqrt() calculations (negative values)

  - **Singleton Pattern Edge Cases:**
    - Test CompatibilityManager multiple initialization attempts
    - Test get_instance() after explicit _instance = None assignment
    - Test concurrent initialization of CompatibilityManager
    - Test initialize() with different engines overwrites previous instance

- Error Handling:
  - **Database Connection Errors:**
    - Test all functions gracefully handle engine.dialect access errors
    - Test functions with engines that have malformed dialect attributes
    - Test CompatibilityManager initialization with invalid engines
    - Test query execution errors with malformed SQL expressions

  - **Statistical Calculation Errors:**
    - Test stddev_samp() SQLite approximation with mathematical errors (NaN, infinity)
    - Test sqrt() calculations with negative variance approximations
    - Test percentile calculations with empty datasets
    - Test epoch extraction with invalid timestamp formats
    - Test division by zero scenarios in statistical approximations

  - **Global Function Fallback Errors:**
    - Test global functions when CompatibilityManager is not initialized
    - Test RuntimeError handling in all global utility functions
    - Test fallback calculations produce valid SQL expressions
    - Test error propagation from StatisticalFunctions methods
    - Test engine parameter validation in global functions

  - **SQLAlchemy Integration Errors:**
    - Test with invalid ColumnElement types
    - Test with Select objects that cannot be modified
    - Test sql_func method calls with unsupported parameters
    - Test label() method failures on generated expressions

- Coverage Target: 85%+

## Summary

This comprehensive data layer testing requirements document covers all 10+ data layer components with detailed testing specifications including:

- **Event Processing Pipeline**: MovementSequence classification, event validation, and pattern detection
- **Database Models**: SQLAlchemy models with TimescaleDB integration
- **Home Assistant Client**: WebSocket and REST API integration with rate limiting
- **Database Management**: Connection pooling, health monitoring, and query optimization
- **Event Validation**: Security validation, schema validation, and integrity checking
- **Bulk Data Import**: Historical data import with progress tracking and error handling
- **Pattern Detection**: Statistical analysis, corruption detection, and quality monitoring
- **Database Compatibility**: Cross-database abstraction layer
- **Database Utilities**: Dialect-specific statistical functions and query building

Each component includes comprehensive unit tests, integration tests, edge cases, error handling scenarios, and specific coverage targets of 85%+ to ensure robust data layer functionality.