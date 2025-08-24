# Testing Requirements Analysis

## Overview
This document contains detailed testing requirements for the ha-ml-predictor project components to achieve 85%+ test coverage. Each component has been analyzed for actual implementation details and specific testing scenarios.

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

### src/core/constants.py - System Constants and Enums
**Classes Found:** SensorType, SensorState, EventType, ModelType, PredictionType
**Constants/Enums Analyzed:** SensorType enum (PRESENCE, DOOR, CLIMATE, LIGHT, MOTION), SensorState enum (ON, OFF, OPEN, CLOSED, UNKNOWN, UNAVAILABLE), EventType enum (STATE_CHANGE, PREDICTION, MODEL_UPDATE, ACCURACY_UPDATE), ModelType enum (LSTM, XGBOOST, HMM, GAUSSIAN_PROCESS, GP, ENSEMBLE), PredictionType enum (NEXT_OCCUPIED, NEXT_VACANT, OCCUPANCY_DURATION, VACANCY_DURATION), PRESENCE_STATES, ABSENCE_STATES, DOOR_OPEN_STATES, DOOR_CLOSED_STATES, INVALID_STATES, MIN_EVENT_SEPARATION, MAX_SEQUENCE_GAP, DEFAULT_CONFIDENCE_THRESHOLD, TEMPORAL_FEATURE_NAMES, SEQUENTIAL_FEATURE_NAMES, CONTEXTUAL_FEATURE_NAMES, MQTT_TOPICS, DB_TABLES, API_ENDPOINTS, DEFAULT_MODEL_PARAMS, HUMAN_MOVEMENT_PATTERNS, CAT_MOVEMENT_PATTERNS

**Required Tests:**
- Unit Tests: Enum value verification tests, constant list content tests, numeric constant validation tests, dictionary structure tests, model parameter validation tests, movement pattern parameter tests, format string validation tests, immutability tests, type checking tests, backward compatibility tests for parameter aliases
- Integration Tests: Enum-constant consistency tests, cross-constant validation tests, format string functionality tests, model type parameter alignment tests
- Edge Cases: Enum modification attempts, constant list/dict modification attempts, format string edge cases, parameter value boundary testing, memory efficiency testing, performance testing for large feature lists
- Error Handling: Invalid enum member access, format string validation errors, import error scenarios, type validation errors
- Coverage Target: 85%+

### src/core/environment.py - Environment Management
**Classes Found:** Environment, SecretConfig, EnvironmentSettings, SecretsManager, EnvironmentManager
**Methods Analyzed:** Environment.from_string(), SecretsManager.__init__(), SecretsManager._get_or_create_key(), SecretsManager.encrypt_secret(), SecretsManager.decrypt_secret(), SecretsManager.store_secret(), SecretsManager.get_secret(), SecretsManager.list_secrets(), SecretsManager.rotate_encryption_key(), EnvironmentManager.__init__(), EnvironmentManager._detect_environment(), EnvironmentManager.get_environment_settings(), EnvironmentManager.get_config_file_path(), EnvironmentManager.load_environment_config(), EnvironmentManager._apply_environment_overrides(), EnvironmentManager._inject_secrets(), EnvironmentManager.get_secret(), EnvironmentManager.set_secret(), EnvironmentManager.validate_configuration(), EnvironmentManager._validate_config_structure(), EnvironmentManager._validate_production_config(), EnvironmentManager._validate_staging_config(), EnvironmentManager.setup_environment_secrets(), EnvironmentManager.export_environment_template(), get_environment_manager()

**Required Tests:**

#### Unit Tests:
1. **Environment Enum Tests:**
   - `test_environment_enum_values()` - Test all enum values match expected strings
   - `test_from_string_exact_matches()` - Test exact string matches (development, testing, staging, production)
   - `test_from_string_short_forms()` - Test short form mappings (dev, test, stage, prod)
   - `test_from_string_case_insensitive()` - Test case insensitive matching
   - `test_from_string_unknown_default()` - Test unknown strings default to DEVELOPMENT
   - `test_from_string_empty_string()` - Test empty string defaults to DEVELOPMENT

2. **SecretConfig Dataclass Tests:**
   - `test_secret_config_required_only()` - Test minimal creation with key only
   - `test_secret_config_full_fields()` - Test creation with all fields
   - `test_secret_config_defaults()` - Test default values (required=True, encrypted=False)
   - `test_secret_config_optional_fields()` - Test Optional field handling

3. **EnvironmentSettings Dataclass Tests:**
   - `test_environment_settings_defaults()` - Test all default values
   - `test_environment_settings_field_assignment()` - Test field modification
   - `test_environment_settings_boolean_fields()` - Test boolean field handling
   - `test_environment_settings_numeric_fields()` - Test integer field validation

4. **SecretsManager Initialization Tests:**
   - `test_secrets_manager_init_default_dir()` - Test default secrets directory creation
   - `test_secrets_manager_init_custom_dir()` - Test custom directory path
   - `test_secrets_manager_init_existing_dir()` - Test existing directory handling
   - `test_secrets_manager_key_creation()` - Test encryption key initialization
   - `test_secrets_manager_cipher_setup()` - Test Fernet cipher creation

5. **Encryption Key Management Tests:**
   - `test_get_or_create_key_new()` - Test new key generation
   - `test_get_or_create_key_existing()` - Test existing key loading
   - `test_key_file_permissions()` - Test key file permissions (0o600)
   - `test_key_file_base64_encoding()` - Test key file base64 encoding/decoding
   - `test_key_generation_uniqueness()` - Test multiple key generations are unique

6. **Secret Encryption/Decryption Tests:**
   - `test_encrypt_secret_string()` - Test string encryption
   - `test_decrypt_secret_success()` - Test successful decryption
   - `test_encrypt_decrypt_roundtrip()` - Test encryption/decryption roundtrip
   - `test_decrypt_invalid_data()` - Test decryption failure with invalid data
   - `test_encrypt_empty_string()` - Test empty string encryption
   - `test_encrypt_unicode_string()` - Test Unicode string handling

7. **Secret Storage Tests:**
   - `test_store_secret_new_file()` - Test storing secret in new file
   - `test_store_secret_existing_file()` - Test adding secret to existing file
   - `test_store_secret_encrypted()` - Test encrypted secret storage
   - `test_store_secret_plaintext()` - Test plaintext secret storage
   - `test_store_secret_file_permissions()` - Test secrets file permissions (0o600)
   - `test_store_secret_json_format()` - Test JSON format with encrypted flag

8. **Secret Retrieval Tests:**
   - `test_get_secret_existing()` - Test retrieving existing secret
   - `test_get_secret_missing_default()` - Test default return for missing secret
   - `test_get_secret_missing_file()` - Test behavior when secrets file doesn't exist
   - `test_get_secret_encrypted()` - Test retrieving encrypted secret
   - `test_get_secret_plaintext()` - Test retrieving plaintext secret
   - `test_get_secret_decryption_failure()` - Test decryption failure handling

9. **Secret Listing Tests:**
   - `test_list_secrets_existing()` - Test listing secrets from existing file
   - `test_list_secrets_missing_file()` - Test empty list when file missing
   - `test_list_secrets_empty_file()` - Test empty secrets file handling
   - `test_list_secrets_multiple_environments()` - Test secrets across environments

10. **Encryption Key Rotation Tests:**
    - `test_rotate_encryption_key_success()` - Test successful key rotation
    - `test_rotate_encryption_key_multiple_envs()` - Test rotation across environments
    - `test_rotate_encryption_key_mixed_secrets()` - Test rotation with encrypted/plaintext mix
    - `test_rotate_encryption_key_corruption_handling()` - Test handling corrupted secrets during rotation
    - `test_rotate_encryption_key_new_key_generation()` - Test new key generation during rotation

11. **EnvironmentManager Initialization Tests:**
    - `test_environment_manager_init_defaults()` - Test initialization with default paths
    - `test_environment_manager_init_custom_paths()` - Test custom config/secrets directories
    - `test_environment_manager_secrets_manager_creation()` - Test SecretsManager initialization
    - `test_environment_manager_environment_detection()` - Test current environment detection

12. **Environment Detection Tests:**
    - `test_detect_environment_env_var()` - Test ENVIRONMENT variable detection
    - `test_detect_environment_env_var_precedence()` - Test ENV variable as fallback
    - `test_detect_environment_config_files()` - Test detection via config file presence
    - `test_detect_environment_docker()` - Test Docker environment detection (/.dockerenv)
    - `test_detect_environment_default()` - Test default to DEVELOPMENT
    - `test_detect_environment_priority()` - Test detection priority order

13. **Configuration Loading Tests:**
    - `test_get_config_file_path_environment_specific()` - Test env-specific config file
    - `test_get_config_file_path_fallback()` - Test fallback to base config
    - `test_load_environment_config_success()` - Test successful config loading
    - `test_load_environment_config_missing_file()` - Test FileNotFoundError for missing config
    - `test_load_environment_config_invalid_yaml()` - Test invalid YAML handling
    - `test_load_environment_config_empty_yaml()` - Test empty YAML file handling

14. **Configuration Override Tests:**
    - `test_apply_environment_overrides_logging()` - Test logging level override
    - `test_apply_environment_overrides_database()` - Test database pool size override
    - `test_apply_environment_overrides_api()` - Test API debug flag override
    - `test_apply_environment_overrides_environment_section()` - Test environment section creation
    - `test_apply_environment_overrides_missing_sections()` - Test creating missing config sections

15. **Secret Injection Tests:**
    - `test_inject_secrets_ha_token()` - Test Home Assistant token injection
    - `test_inject_secrets_database_password()` - Test database password injection in connection string
    - `test_inject_secrets_redis_password()` - Test Redis password injection
    - `test_inject_secrets_api_secret_key()` - Test API secret key injection
    - `test_inject_secrets_missing_sections()` - Test creating missing config sections for secrets

16. **Secret Retrieval with Environment Variables Tests:**
    - `test_get_secret_env_var_priority()` - Test environment variable takes priority
    - `test_get_secret_fallback_to_manager()` - Test fallback to secrets manager
    - `test_get_secret_case_sensitive_env()` - Test case sensitivity of env var names
    - `test_get_secret_env_var_uppercase()` - Test automatic uppercase conversion

17. **Configuration Validation Tests:**
    - `test_validate_configuration_success()` - Test successful validation
    - `test_validate_configuration_missing_secrets()` - Test missing required secrets detection
    - `test_validate_config_structure_ha_missing()` - Test missing Home Assistant config
    - `test_validate_config_structure_database_invalid()` - Test invalid database config
    - `test_validate_config_structure_mqtt_missing()` - Test missing MQTT config
    - `test_validate_production_config_debug()` - Test production debug mode validation
    - `test_validate_production_config_monitoring()` - Test production monitoring requirements
    - `test_validate_staging_config()` - Test staging-specific validation

18. **Interactive Setup Tests:**
    - `test_setup_environment_secrets_new()` - Test setting up new secrets
    - `test_setup_environment_secrets_existing()` - Test updating existing secrets
    - `test_setup_environment_secrets_defaults()` - Test using default values
    - `test_setup_environment_secrets_required()` - Test required secret prompting
    - `test_setup_environment_secrets_skip_existing()` - Test skipping configured secrets

19. **Template Export Tests:**
    - `test_export_environment_template()` - Test template export functionality
    - `test_export_template_structure()` - Test exported template structure
    - `test_export_template_secrets_format()` - Test secrets format in template
    - `test_export_template_settings_inclusion()` - Test environment settings inclusion

20. **Global Manager Tests:**
    - `test_get_environment_manager_singleton()` - Test singleton behavior
    - `test_get_environment_manager_creation()` - Test manager creation on first call

#### Integration Tests:
1. **File System Integration:**
   - `test_real_secrets_directory_creation()` - Test actual directory creation
   - `test_real_file_operations()` - Test actual file read/write operations
   - `test_secrets_file_permissions_integration()` - Test file permissions in real filesystem
   - `test_config_file_loading_integration()` - Test loading actual config files

2. **Cross-Environment Integration:**
   - `test_multiple_environment_secrets()` - Test secrets across all environments
   - `test_environment_switching()` - Test switching between environments
   - `test_environment_specific_configs()` - Test different configs per environment

3. **Encryption Integration:**
   - `test_full_encryption_workflow()` - Test complete encryption workflow
   - `test_key_rotation_workflow()` - Test complete key rotation process
   - `test_cross_session_encryption()` - Test encryption across different manager instances

#### Edge Cases:
1. **File System Edge Cases:**
   - `test_secrets_directory_permission_denied()` - Test directory creation permission errors
   - `test_corrupted_secrets_file()` - Test handling corrupted JSON files
   - `test_corrupted_key_file()` - Test handling corrupted key files
   - `test_concurrent_file_access()` - Test concurrent access to secrets files
   - `test_disk_full_scenarios()` - Test behavior when disk is full

2. **Encryption Edge Cases:**
   - `test_encryption_with_special_characters()` - Test encrypting special characters
   - `test_encryption_with_large_secrets()` - Test encrypting very large secrets
   - `test_decryption_with_wrong_key()` - Test decryption failure with wrong key
   - `test_key_rotation_partial_failure()` - Test partial failure during key rotation

3. **Environment Detection Edge Cases:**
   - `test_environment_detection_case_variations()` - Test case variations in env vars
   - `test_multiple_config_files_present()` - Test priority when multiple config files exist
   - `test_docker_file_edge_cases()` - Test edge cases in Docker detection
   - `test_environment_var_empty_string()` - Test empty string environment variables

4. **Configuration Edge Cases:**
   - `test_deeply_nested_config_structures()` - Test very nested configuration structures
   - `test_config_with_null_values()` - Test configuration with null/None values
   - `test_config_circular_references()` - Test handling potential circular references
   - `test_very_large_config_files()` - Test loading very large configuration files

#### Error Handling:
1. **File System Errors:**
   - `test_secrets_directory_creation_failure()` - Test directory creation failures
   - `test_key_file_read_permission_denied()` - Test key file permission errors
   - `test_secrets_file_write_failure()` - Test secrets file write failures
   - `test_config_file_missing_error()` - Test missing configuration file errors

2. **Encryption Errors:**
   - `test_encryption_key_generation_failure()` - Test key generation failures
   - `test_decryption_invalid_token_error()` - Test Fernet InvalidToken errors
   - `test_encryption_memory_errors()` - Test encryption with memory constraints
   - `test_base64_decode_errors()` - Test base64 decoding failures

3. **Configuration Validation Errors:**
   - `test_validation_missing_required_config()` - Test validation failures
   - `test_validation_invalid_config_types()` - Test type validation failures
   - `test_validation_production_security_errors()` - Test production security validation

4. **Environment Variable Errors:**
   - `test_invalid_environment_variable_values()` - Test invalid env var values
   - `test_environment_variable_parsing_errors()` - Test env var parsing failures

#### Performance Tests:
1. **Encryption Performance:**
   - `test_encryption_performance_large_secrets()` - Test encryption speed with large data
   - `test_decryption_performance_benchmarks()` - Test decryption performance
   - `test_key_rotation_performance()` - Test key rotation performance with many secrets

2. **File Operations Performance:**
   - `test_secrets_file_loading_performance()` - Test file loading speed
   - `test_concurrent_secret_access_performance()` - Test concurrent access performance
   - `test_large_config_loading_performance()` - Test large configuration loading

**Coverage Target:** 85%+

**Mock Requirements:**
- Mock os.getenv() for environment variable testing
- Mock Path.exists() and Path.mkdir() for filesystem operations
- Mock open() and file read/write operations
- Mock Fernet key generation for deterministic testing
- Mock yaml.safe_load() for configuration loading
- Mock input() for interactive setup testing
- Mock datetime.now() for time-dependent tests

**Test Fixtures Needed:**
- Sample environment configurations for all environments
- Mock secrets files with various encryption states
- Sample YAML configuration files
- Encryption key fixtures for testing
- Environment variable fixtures
- File permission testing fixtures

**Special Testing Considerations:**
- Environment detection requires careful mocking of filesystem and environment variables
- Encryption testing needs deterministic key generation for reproducible tests  
- File operations require proper cleanup in test teardown
- Interactive setup testing requires input mocking
- Cross-platform file permission testing considerations
- Thread safety testing for concurrent secret access
- Memory usage testing for large secrets and configurations
- Security testing to ensure encrypted secrets remain secure

### src/integration/mqtt_publisher.py - MQTT Publisher
**Classes Found:** MQTTConnectionStatus, MQTTPublishResult, MQTTPublisher, MQTTPublisherError
**Methods Analyzed:** MQTTConnectionStatus (dataclass), MQTTPublishResult (dataclass), MQTTPublisher.__init__(), MQTTPublisher.initialize(), MQTTPublisher.start_publisher(), MQTTPublisher.stop_publisher(), MQTTPublisher.publish(), MQTTPublisher.publish_json(), MQTTPublisher.get_connection_status(), MQTTPublisher.get_publisher_stats(), MQTTPublisher._connect_to_broker(), MQTTPublisher._disconnect_from_broker(), MQTTPublisher._connection_monitoring_loop(), MQTTPublisher._message_queue_processing_loop(), MQTTPublisher._process_message_queue(), MQTTPublisher._on_connect(), MQTTPublisher._on_disconnect(), MQTTPublisher._on_publish(), MQTTPublisher._on_message(), MQTTPublisher._on_log(), MQTTPublisherError.__init__()

**Required Tests:**

**Unit Tests:**
- **MQTTConnectionStatus Dataclass Tests:**
  - Test default initialization with connected=False
  - Test field assignment and modification
  - Test Optional datetime field handling (last_connected, last_disconnected)
  - Test numeric field validation (connection_attempts, reconnect_count, uptime_seconds)

- **MQTTPublishResult Dataclass Tests:**
  - Test successful result creation with all required fields
  - Test failure result creation with error_message
  - Test optional message_id field handling
  - Test payload_size calculation accuracy
  - Test publish_time timestamp generation

- **MQTTPublisher Initialization Tests:**
  - Test __init__ with MQTTConfig parameter
  - Test client_id auto-generation when None provided
  - Test client_id custom assignment
  - Test callback parameter handling (on_connect, on_disconnect, on_message)
  - Test initialization of internal state (connection_status, message_queue, locks)
  - Test statistics initialization (counters, timestamps)

- **MQTT Client Setup Tests:**
  - Test initialize() method with publishing enabled
  - Test initialize() method with publishing disabled
  - Test MQTT client creation with correct parameters
  - Test callback assignment (_on_connect, _on_disconnect, etc.)
  - Test authentication configuration with username/password
  - Test TLS configuration for port 8883
  - Test keepalive and connection timeout settings

- **Connection Management Tests:**
  - Test _connect_to_broker() successful connection
  - Test _connect_to_broker() with retry logic
  - Test connection timeout handling
  - Test max_reconnect_attempts configuration
  - Test _disconnect_from_broker() graceful disconnection
  - Test connection status updates during connect/disconnect

- **Publisher Lifecycle Tests:**
  - Test start_publisher() background task creation
  - Test start_publisher() with publishing disabled
  - Test start_publisher() when already active
  - Test stop_publisher() graceful shutdown
  - Test shutdown_event signaling
  - Test background task cleanup
  - Test remaining message processing on shutdown

- **Message Publishing Tests:**
  - Test publish() with string payload
  - Test publish() with dict payload (JSON conversion)
  - Test publish() with bytes payload
  - Test publish() with various QoS levels (0, 1, 2)
  - Test publish() with retain flag
  - Test publish_json() convenience method
  - Test payload size calculation
  - Test message ID assignment

- **Message Queuing Tests:**
  - Test message queuing when client disconnected
  - Test message queue size limits (max_queue_size=1000)
  - Test queue overflow handling (oldest message removal)
  - Test _process_message_queue() batch processing
  - Test queue processing with connection status
  - Test queued message timestamp tracking

- **Background Task Loop Tests:**
  - Test _connection_monitoring_loop() connection checking
  - Test _connection_monitoring_loop() reconnection attempts
  - Test _message_queue_processing_loop() processing cycle
  - Test background loop exception handling
  - Test shutdown_event respecting in loops
  - Test loop timeout and interval behavior

- **MQTT Callback Tests:**
  - Test _on_connect() successful connection handling
  - Test _on_connect() connection failure handling
  - Test _on_connect() callback invocation (sync and async)
  - Test _on_disconnect() clean vs unexpected disconnection
  - Test _on_disconnect() callback invocation
  - Test _on_publish() success and failure scenarios
  - Test _on_message() message reception
  - Test _on_log() MQTT logging integration

- **Statistics and Status Tests:**
  - Test get_connection_status() current status reporting
  - Test get_connection_status() uptime calculation
  - Test get_publisher_stats() comprehensive statistics
  - Test statistics tracking (messages published/failed, bytes)
  - Test last_publish_time updating

- **Error Handling Tests:**
  - Test MQTTPublisherError exception creation
  - Test publish() error scenarios
  - Test connection error handling
  - Test callback exception handling
  - Test background task error recovery

**Integration Tests:**
- **MQTT Broker Integration:**
  - Test connection to real MQTT broker (test environment)
  - Test message publishing to real broker
  - Test subscription and message reception
  - Test authentication with real credentials
  - Test TLS connection establishment

- **Configuration Integration:**
  - Test integration with MQTTConfig dataclass
  - Test various broker configurations (localhost, remote, ports)
  - Test authentication configurations
  - Test timeout and retry configurations

- **Async Task Integration:**
  - Test background tasks with asyncio event loop
  - Test concurrent message publishing
  - Test task cancellation during shutdown
  - Test integration with other async components

- **TrackingManager Integration:**
  - Test MQTT publisher integration with tracking system
  - Test automatic message publishing from tracking events
  - Test callback integration for system events

**Edge Cases:**
- **Network Edge Cases:**
  - Test network disconnection during publishing
  - Test network reconnection scenarios
  - Test broker unavailability
  - Test connection timeout edge cases
  - Test intermittent connectivity issues

- **Message Edge Cases:**
  - Test publishing empty messages
  - Test very large message payloads
  - Test invalid JSON in dict payload
  - Test Unicode characters in messages
  - Test binary data handling

- **Queue Edge Cases:**
  - Test message queue overflow scenarios
  - Test queue processing with repeated failures
  - Test queue corruption handling
  - Test concurrent queue access

- **Threading Edge Cases:**
  - Test thread safety of message queue
  - Test concurrent callback execution
  - Test race conditions in connection status
  - Test lock contention scenarios

- **Configuration Edge Cases:**
  - Test invalid broker hostnames
  - Test invalid port numbers
  - Test missing authentication credentials
  - Test malformed configuration parameters

**Error Handling:**
- **Connection Error Handling:**
  - Test connection failures with proper exception propagation
  - Test authentication failures
  - Test network timeout handling
  - Test SSL/TLS certificate errors

- **Publishing Error Handling:**
  - Test publish failures with return code analysis
  - Test message serialization errors
  - Test broker rejection scenarios
  - Test QoS level handling failures

- **Callback Error Handling:**
  - Test callback exception isolation
  - Test async callback error handling
  - Test callback timeout scenarios
  - Test malformed callback parameters

- **Resource Error Handling:**
  - Test memory exhaustion scenarios
  - Test file descriptor limits
  - Test thread pool exhaustion
  - Test system resource constraints

**Coverage Target: 85%+**

**Mock Requirements:**
- Mock paho.mqtt.client for MQTT client operations
- Mock asyncio.create_task for background task creation
- Mock datetime.utcnow for timestamp testing
- Mock ssl.create_default_context for TLS testing
- Mock threading.RLock for lock testing
- Mock json.dumps for payload serialization testing

**Test Fixtures Needed:**
- Sample MQTTConfig objects for various scenarios
- Mock MQTT client responses for different scenarios
- Sample message payloads (string, dict, bytes)
- Connection status scenarios (connected, disconnected, error)
- Background task lifecycle fixtures

**Special Testing Considerations:**
- MQTT client mocking requires careful callback simulation
- Background task testing requires proper asyncio event loop management
- Connection status testing needs time-based validation
- Message queue testing requires thread safety validation
- Integration tests need careful broker setup/teardown
- Network simulation for connection failure scenarios

### src/data/storage/models.py - SQLAlchemy Models
**Classes Found:** SensorEvent, RoomState, Prediction, ModelAccuracy, FeatureStore, PredictionAudit
**Methods Analyzed:** SensorEvent.__init__(), SensorEvent.get_recent_events(), SensorEvent.get_state_changes(), SensorEvent.get_transition_sequences(), SensorEvent.get_predictions(), SensorEvent.get_advanced_analytics(), SensorEvent.get_sensor_efficiency_metrics(), SensorEvent.get_temporal_patterns(), SensorEvent._calculate_efficiency_score(), RoomState.get_current_state(), RoomState.get_occupancy_history(), RoomState.get_predictions(), RoomState.get_occupancy_sessions(), RoomState.get_precision_occupancy_metrics(), Prediction.__init__(), Prediction.get_pending_validations(), Prediction.get_accuracy_metrics(), Prediction.get_triggering_event(), Prediction.get_room_state(), Prediction.get_predictions_with_events(), Prediction.get_predictions_with_full_context(), Prediction._extract_top_features(), Prediction._categorize_features(), Prediction._analyze_confidence_spread(), Prediction.add_extended_metadata(), FeatureStore.get_latest_features(), FeatureStore.get_all_features(), PredictionAudit.create_audit_entry(), PredictionAudit.get_audit_trail_with_relationships(), PredictionAudit.analyze_json_details(), PredictionAudit._calculate_json_complexity(), PredictionAudit.update_validation_metrics(), plus utility functions _is_sqlite_engine(), _get_database_specific_column_config(), _get_json_column_type(), create_timescale_hypertables(), optimize_database_performance(), get_bulk_insert_query()

**Required Tests:**
- Unit Tests: Model initialization tests, class method functionality, JSON field handling, data validation, relationship handling, query method tests, utility function tests, static method tests, configuration tests
- Integration Tests: Database operations with real SQLAlchemy sessions, TimescaleDB integration, cross-model relationships, bulk operations, async session handling
- Edge Cases: Large JSON payloads, extreme timestamp values, Unicode data, null/None handling, database-specific behavior differences
- Error Handling: Constraint violations, foreign key errors, JSON parsing errors, database connection failures, TimescaleDB unavailability
- Coverage Target: 85%+

### src/core/exceptions.py - Custom Exception Classes
**Classes Found:** ErrorSeverity, OccupancyPredictionError, ConfigurationError, ConfigFileNotFoundError, ConfigValidationError, MissingConfigSectionError, ConfigParsingError, HomeAssistantError, HomeAssistantConnectionError, HomeAssistantAuthenticationError, HomeAssistantAPIError, EntityNotFoundError, WebSocketError, WebSocketConnectionError, WebSocketAuthenticationError, WebSocketValidationError, DatabaseError, DatabaseConnectionError, DatabaseQueryError, DatabaseMigrationError, DatabaseIntegrityError, FeatureEngineeringError, FeatureExtractionError, InsufficientDataError, FeatureValidationError, FeatureStoreError, ModelError, ModelTrainingError, ModelPredictionError, InsufficientTrainingDataError, ModelNotFoundError, ModelVersionMismatchError, MissingFeatureError, ModelValidationError, DataProcessingError, DataCorruptionError, IntegrationError, DataValidationError, MQTTError, MQTTConnectionError, MQTTPublishError, MQTTSubscriptionError, APIServerError, SystemInitializationError, SystemResourceError, SystemError, ResourceExhaustionError, ServiceUnavailableError, MaintenanceModeError, APIError, APIAuthenticationError, RateLimitExceededError, APIAuthorizationError, APISecurityError, APIResourceNotFoundError
**Methods Analyzed:** OccupancyPredictionError.__init__(), OccupancyPredictionError.__str__(), ConfigurationError.__init__(), HomeAssistantAuthenticationError.__init__(), DatabaseConnectionError._mask_password(), DatabaseQueryError.__init__(), DataValidationError.__init__(), validate_room_id(), validate_entity_id()

**Required Tests:**
- Unit Tests: Base exception functionality, error severity enumeration, context and cause handling, string formatting, inheritance hierarchy, specialized exception parameters, password masking, validation functions
- Integration Tests: Exception chaining, error propagation through system layers, configuration error scenarios, logging integration
- Edge Cases: Very long error messages, Unicode characters, nested context data, circular references, extreme parameter values
- Error Handling: Exception creation failures, serialization issues, logging failures
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

### src/core/config.py - Configuration Management
**Classes Found:** HomeAssistantConfig, DatabaseConfig, MQTTConfig, PredictionConfig, FeaturesConfig, LoggingConfig, TrackingConfig, JWTConfig, APIConfig, SensorConfig, RoomConfig, SystemConfig, ConfigLoader
**Methods Analyzed:** DatabaseConfig.__post_init__(), RoomConfig.get_all_entity_ids(), RoomConfig.get_sensors_by_type(), SystemConfig.get_all_entity_ids(), SystemConfig.get_room_by_entity_id(), JWTConfig.__post_init__(), APIConfig.__post_init__(), TrackingConfig.__post_init__(), ConfigLoader.__init__(), ConfigLoader.load_config(), ConfigLoader._load_yaml(), ConfigLoader._create_system_config(), get_config(), reload_config()

**Required Tests:**

**Unit Tests:**
- **Dataclass Configuration Objects:**
  - Test HomeAssistantConfig initialization with required/optional parameters
  - Test DatabaseConfig.__post_init__() with DATABASE_URL environment variable override
  - Test MQTTConfig initialization with extensive MQTT configuration options
  - Test PredictionConfig, FeaturesConfig, LoggingConfig with default values
  - Test TrackingConfig.__post_init__() with default alert_thresholds creation
  - Test SensorConfig with entity_id, sensor_type, room_id parameters

- **JWT Configuration Tests:**
  - Test JWTConfig.__post_init__() with JWT_ENABLED environment variable ("false", "0", "no", "off")
  - Test JWTConfig secret key loading from JWT_SECRET_KEY environment variable
  - Test JWTConfig test environment fallback with default test secret key
  - Test JWTConfig secret key length validation (minimum 32 characters)
  - Test JWTConfig ValueError for missing secret key in non-test environments
  - Test JWTConfig environment detection (test, CI environments)

- **API Configuration Tests:**
  - Test APIConfig.__post_init__() environment variable loading (API_ENABLED, API_HOST, API_PORT, API_DEBUG)
  - Test APIConfig CORS configuration with CORS_ENABLED, CORS_ALLOW_ORIGINS
  - Test APIConfig API key configuration (API_KEY, API_KEY_ENABLED)
  - Test APIConfig rate limiting settings (API_RATE_LIMIT_ENABLED, API_RATE_LIMIT_PER_MINUTE, API_RATE_LIMIT_BURST)
  - Test APIConfig background tasks settings (API_BACKGROUND_TASKS_ENABLED, HEALTH_CHECK_INTERVAL_SECONDS)
  - Test APIConfig logging settings (API_ACCESS_LOG, API_LOG_REQUESTS, API_LOG_RESPONSES)
  - Test APIConfig documentation settings (API_INCLUDE_DOCS)
  - Test APIConfig ValueError for missing API key when enabled
  - Test APIConfig default CORS origins splitting on comma
  - Test APIConfig nested JWTConfig field initialization

- **Room Configuration Tests:**
  - Test RoomConfig.get_all_entity_ids() with nested dictionary sensor structures
  - Test RoomConfig.get_all_entity_ids() with list-based sensor structures
  - Test RoomConfig.get_all_entity_ids() entity ID extraction (binary_sensor., sensor. prefixes)
  - Test RoomConfig.get_sensors_by_type() with dictionary sensor type
  - Test RoomConfig.get_sensors_by_type() with string sensor type
  - Test RoomConfig.get_sensors_by_type() with missing sensor type
  - Test RoomConfig recursive extraction from nested objects

- **System Configuration Tests:**
  - Test SystemConfig.get_all_entity_ids() aggregation from all rooms
  - Test SystemConfig.get_all_entity_ids() duplicate removal (set conversion)
  - Test SystemConfig.get_room_by_entity_id() entity lookup across rooms
  - Test SystemConfig.get_room_by_entity_id() with non-existent entity (None return)

- **ConfigLoader Tests:**
  - Test ConfigLoader.__init__() with valid config directory
  - Test ConfigLoader.__init__() with missing config directory (FileNotFoundError)
  - Test ConfigLoader._load_yaml() with valid YAML files
  - Test ConfigLoader._load_yaml() with missing files (FileNotFoundError)
  - Test ConfigLoader._load_yaml() with invalid YAML content
  - Test ConfigLoader._load_yaml() return type handling (dict vs non-dict)

- **Configuration Loading Tests:**
  - Test ConfigLoader.load_config() with environment parameter
  - Test ConfigLoader.load_config() environment-specific config loading (config.{environment}.yaml)
  - Test ConfigLoader.load_config() fallback to base config.yaml when environment config missing
  - Test ConfigLoader.load_config() rooms.yaml integration
  - Test ConfigLoader.load_config() nested room structure handling (hallways example)
  - Test ConfigLoader.load_config() regular room structure processing
  - Test ConfigLoader.load_config() dataclass object creation for all config sections

- **Global Configuration Tests:**
  - Test get_config() singleton behavior with global _config_instance
  - Test get_config() environment manager integration (when available)
  - Test get_config() ImportError fallback to direct ConfigLoader
  - Test reload_config() forced reload behavior
  - Test reload_config() environment manager vs direct loading paths

**Integration Tests:**
- **File System Integration:**
  - Test complete configuration loading from actual YAML files
  - Test environment-specific configuration override behavior
  - Test configuration loading with complex nested room structures
  - Test environment variable integration across all configuration objects
  - Test configuration validation with environment manager integration

- **Cross-Configuration Integration:**
  - Test SystemConfig with all nested configuration objects
  - Test entity ID aggregation across multiple room configurations
  - Test room lookup functionality with realistic sensor mappings
  - Test JWT and API configuration interaction

- **Environment Manager Integration:**
  - Test get_config() with environment manager secret injection
  - Test configuration processing through environment manager
  - Test fallback behavior when environment manager unavailable

**Edge Cases:**
- **Configuration File Edge Cases:**
  - Test configuration loading with empty YAML files
  - Test configuration loading with malformed YAML syntax
  - Test configuration loading with missing required sections
  - Test configuration loading with Unicode characters in configuration values
  - Test configuration loading with very large configuration files
  - Test configuration loading with deeply nested room structures

- **Environment Variable Edge Cases:**
  - Test environment variable handling with empty string values
  - Test environment variable handling with case sensitivity variations
  - Test environment variable parsing with invalid values (non-numeric for numeric fields)
  - Test environment variable boolean parsing with various true/false representations
  - Test environment variable list parsing (comma-separated values)

- **Room Configuration Edge Cases:**
  - Test room configuration with no sensors defined
  - Test room configuration with empty sensor dictionaries
  - Test room configuration with mixed sensor type definitions (strings vs dicts)
  - Test room configuration with circular references in nested structures
  - Test room configuration with null/None values in sensor definitions

- **Configuration Object Edge Cases:**
  - Test configuration objects with extreme parameter values (very large timeouts, pools)
  - Test configuration objects with boundary values (minimum/maximum allowed)
  - Test configuration objects with conflicting parameter combinations
  - Test configuration objects with missing optional parameters

**Error Handling:**
- **File System Errors:**
  - Test ConfigLoader behavior with permission denied errors
  - Test ConfigLoader behavior with corrupted YAML files
  - Test ConfigLoader behavior with symbolic link issues
  - Test ConfigLoader behavior when config directory becomes unavailable

- **Configuration Validation Errors:**
  - Test JWTConfig validation failures (short secret key, missing secret)
  - Test APIConfig validation failures (missing API key when enabled)
  - Test DatabaseConfig validation with invalid connection strings
  - Test configuration object validation with invalid parameter types

- **Environment Integration Errors:**
  - Test environment manager import failures
  - Test environment manager configuration processing failures
  - Test secret injection failures in environment manager integration

- **Data Processing Errors:**
  - Test room configuration processing with corrupted sensor data
  - Test entity ID extraction with malformed entity identifiers
  - Test configuration object creation with invalid parameter combinations

**Coverage Target: 85%+

**Mock Requirements:**
- Mock Path.exists() and Path() for filesystem operations
- Mock open() and file operations for YAML loading
- Mock yaml.safe_load() for configuration parsing
- Mock os.getenv() for environment variable testing
- Mock environment manager import and methods
- Mock print() for test environment JWT warning messages

**Test Fixtures Needed:**
- Sample configuration YAML files (base and environment-specific)
- Sample rooms.yaml with various room structures (simple, nested, hallways)
- Environment variable fixtures for all configuration objects
- Invalid YAML content fixtures for error testing
- Complex sensor configuration fixtures for room testing

**Special Testing Considerations:**
- Test dataclass field validation and default values
- Test __post_init__ method behavior across all configuration classes
- Test singleton pattern behavior for global configuration
- Test recursive sensor extraction from nested room structures
- Test environment-specific configuration override behavior
- Test secret key security validation in JWT configuration
- Test comprehensive environment variable parsing across all config objects

### src/models/base/predictor.py - Base Predictor Interface
**Classes Found:** PredictionResult (dataclass), TrainingResult (dataclass), BasePredictor (abstract base class)
**Methods Analyzed:** PredictionResult.to_dict(), TrainingResult.to_dict(), BasePredictor.__init__(), BasePredictor.train() (abstract), BasePredictor.predict() (abstract), BasePredictor.get_feature_importance() (abstract), BasePredictor.predict_single(), BasePredictor.save_model(), BasePredictor.load_model(), BasePredictor.get_model_info(), BasePredictor.get_training_history(), BasePredictor.get_prediction_accuracy(), BasePredictor.clear_prediction_history(), BasePredictor.validate_features(), BasePredictor._record_prediction(), BasePredictor._generate_model_version(), BasePredictor.__str__(), BasePredictor.__repr__()

**Required Tests:**
- Unit Tests: 
  - **PredictionResult Tests:** Test dataclass initialization with all field combinations, test to_dict() serialization with datetime ISO formatting, test optional field handling (None values), test alternatives list serialization, test prediction_interval tuple handling, test all possible field value combinations
  - **TrainingResult Tests:** Test dataclass initialization with required/optional fields, test to_dict() serialization with all field types, test success/failure states, test feature_importance dictionary handling, test training_metrics dictionary serialization
  - **BasePredictor Initialization Tests:** Test __init__ with ModelType enum, room_id assignment, config parameter handling, test default values for state variables (is_trained=False, model_version="v1.0"), test training_date None initialization, test empty lists initialization (feature_names, training_history, prediction_history)
  - **Model Serialization Tests:** Test save_model() with pickle serialization, test model data dictionary structure, test file path handling (Path and string), test model persistence with all state variables, test load_model() deserialization, test model state restoration, test training history restoration from dictionaries, test file error handling in save/load operations
  - **Model Information Tests:** Test get_model_info() dictionary structure, test model state reflection in info, test feature count calculations, test feature names truncation (first 10), test training statistics calculations, test get_training_history() list conversion, test history ordering and completeness
  - **Prediction Management Tests:** Test predict_single() DataFrame conversion from dict, test feature dictionary to DataFrame transformation, test prediction result extraction (first result), test prediction history recording via _record_prediction(), test prediction history memory management (1000 limit, 500 truncation), test get_prediction_accuracy() with time filtering, test clear_prediction_history() functionality
  - **Feature Validation Tests:** Test validate_features() with trained vs untrained models, test feature name matching against self.feature_names, test missing features detection and error reporting, test extra features warning generation, test empty feature_names handling, test DataFrame column validation
  - **Version Management Tests:** Test _generate_model_version() with empty history, test version incrementation logic (v1.0 -> v1.1), test version parsing and float conversion, test fallback to base version on parse errors, test version string format validation
  - **String Representation Tests:** Test __str__() format with/without room_id, test trained/untrained status display, test __repr__() detailed format with all parameters, test model_type enum value display

- Integration Tests:
  - **Abstract Method Integration:** Test that BasePredictor cannot be instantiated directly (TypeError), test that concrete subclasses must implement train(), predict(), get_feature_importance() methods, test abstract method signature validation, test inheritance behavior with multiple concrete implementations
  - **Model Persistence Integration:** Test complete save/load cycle with real file system, test persistence across different Python sessions, test model state consistency after load, test error recovery from corrupted pickle files
  - **Configuration Integration:** Test BasePredictor with SystemConfig integration, test config parameter propagation to subclasses, test room_id validation with config room definitions
  - **Feature Store Integration:** Test feature validation with actual feature engineering outputs, test DataFrame compatibility with feature store formats, test feature name consistency across system components

- Edge Cases:
  - **Data Type Edge Cases:** Test PredictionResult with extreme datetime values (timezone aware), test alternatives with empty lists vs None, test prediction_interval with same start/end times, test confidence_score boundary values (0.0, 1.0), test transition_type validation with edge strings
  - **Model State Edge Cases:** Test BasePredictor with very large training histories (memory management), test prediction_history with concurrent access patterns, test model version generation with malformed version strings, test feature names with special characters and Unicode
  - **Serialization Edge Cases:** Test pickle serialization with very large model objects, test model saving with insufficient disk space, test loading models created with different library versions, test corrupted pickle file recovery
  - **Validation Edge Cases:** Test validate_features with empty DataFrames, test feature validation with duplicate column names, test DataFrame with mixed data types, test very large feature sets (>1000 features)

- Error Handling:
  - **Pickle Serialization Errors:** Test save_model() with pickle errors, test load_model() with corrupted files, test exception handling and logging for file operations, test graceful failure with informative error messages
  - **Abstract Method Errors:** Test TypeError when instantiating BasePredictor directly, test NotImplementedError propagation from abstract methods, test proper error messages for missing implementations
  - **Feature Validation Errors:** Test ModelPredictionError in predict_single() with empty predictions, test feature validation errors with detailed missing feature reporting, test validation failure recovery and logging
  - **File System Errors:** Test model persistence with permission denied errors, test file not found errors in load_model(), test disk full scenarios during model saving, test network drive issues for model storage
  - **Memory Management Errors:** Test prediction history overflow handling, test memory constraints with large model objects, test garbage collection of cleared histories

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

### src/core/config_validator.py - Configuration Validation
**Classes Found:** ValidationResult, HomeAssistantConfigValidator, DatabaseConfigValidator, MQTTConfigValidator, RoomsConfigValidator, SystemRequirementsValidator, ConfigurationValidator
**Methods Analyzed:** ValidationResult.__init__(), ValidationResult.add_error(), ValidationResult.add_warning(), ValidationResult.add_info(), ValidationResult.merge(), ValidationResult.__str__(), HomeAssistantConfigValidator.validate(), HomeAssistantConfigValidator._is_valid_url(), HomeAssistantConfigValidator.test_connection(), DatabaseConfigValidator.validate(), DatabaseConfigValidator.test_connection(), MQTTConfigValidator.validate(), MQTTConfigValidator.test_connection(), RoomsConfigValidator.validate(), RoomsConfigValidator._validate_room(), RoomsConfigValidator._count_sensors(), RoomsConfigValidator._is_valid_entity_id(), SystemRequirementsValidator.validate(), ConfigurationValidator.__init__(), ConfigurationValidator.validate_configuration(), ConfigurationValidator.validate_config_files()

**Required Tests:**

**Unit Tests:**

- **ValidationResult Class Tests:**
  - Test ValidationResult initialization with is_valid=True, empty error/warning/info lists
  - Test add_error() method updating errors list and setting is_valid=False
  - Test add_warning() method appending to warnings list without affecting is_valid
  - Test add_info() method appending to info list without affecting is_valid
  - Test merge() method combining two ValidationResult objects (errors, warnings, info)
  - Test merge() method setting is_valid=False when merging invalid results
  - Test __str__() formatting with valid status (✅ VALID) and section counts
  - Test __str__() formatting with invalid status (❌ INVALID) and error details
  - Test __str__() emoji formatting for errors (❌), warnings (⚠️), and info (ℹ️)
  - Test __str__() with empty result sections (no errors/warnings/info)
  - Test __str__() with mixed valid/invalid scenarios and comprehensive output formatting

- **HomeAssistantConfigValidator Tests:**
  - Test validate() with complete home_assistant configuration (url, token, timeouts)
  - Test validate() with missing home_assistant section (empty dict handling)
  - Test validate() with missing URL (required field validation)
  - Test validate() with invalid URL formats using _is_valid_url() method
  - Test validate() with missing token (required field validation)
  - Test validate() with short token (<180 characters) generating warning
  - Test validate() with proper token length (≥180 characters) generating info
  - Test validate() websocket_timeout validation (warnings for <10, >300 seconds)
  - Test validate() api_timeout validation (warnings for <5, >60 seconds)
  - Test validate() with default timeout values (websocket_timeout=30, api_timeout=10)
  - Test _is_valid_url() with valid URLs (http/https schemes with netloc)
  - Test _is_valid_url() with invalid URLs (missing scheme, netloc, malformed)
  - Test _is_valid_url() exception handling for malformed URL parsing
  - Test test_connection() with missing URL/token (cannot test connection error)
  - Test test_connection() successful API connection (status 200) with version info
  - Test test_connection() authentication failure (status 401) with specific error
  - Test test_connection() other API failures (non-200/401 status codes)
  - Test test_connection() timeout scenarios with configurable timeout values
  - Test test_connection() connection errors (network unreachable)
  - Test test_connection() SSL verification disabled (verify=False) for self-signed certificates
  - Test test_connection() exception handling for unexpected errors

- **DatabaseConfigValidator Tests:**
  - Test validate() with complete database configuration (connection_string, pool settings)
  - Test validate() with missing database section (empty dict handling)
  - Test validate() with missing connection_string (required field validation)
  - Test validate() with PostgreSQL connection string validation
  - Test validate() with non-PostgreSQL connection strings (error generation)
  - Test validate() TimescaleDB detection in connection string (warning if missing)
  - Test validate() pool_size validation (warnings for <2, >50)
  - Test validate() max_overflow validation relative to pool_size (<50% warning)
  - Test validate() default pool settings (pool_size=10, max_overflow=20)
  - Test validate() pool settings info message formatting
  - Test test_connection() with missing connection_string (cannot test error)
  - Test test_connection() successful database connection with version extraction
  - Test test_connection() connection string conversion (postgresql+asyncpg:// -> postgresql://)
  - Test test_connection() TimescaleDB extension detection and version reporting
  - Test test_connection() TimescaleDB extension not found warning
  - Test test_connection() authentication failures (InvalidAuthorizationSpecificationError)
  - Test test_connection() database not exists error (InvalidCatalogNameError)
  - Test test_connection() connection timeout handling (asyncio.TimeoutError)
  - Test test_connection() asyncpg import error graceful handling
  - Test test_connection() general exception handling with informative messages

- **MQTTConfigValidator Tests:**
  - Test validate() with complete MQTT configuration (broker, port, auth, QoS)
  - Test validate() with missing mqtt section (empty dict handling)
  - Test validate() with missing broker (required field validation)
  - Test validate() with broker info message generation
  - Test validate() port validation (errors for <1, >65535)
  - Test validate() non-standard port warnings (not 1883 or 8883)
  - Test validate() default port handling (port=1883)
  - Test validate() topic_prefix validation (empty warning, leading/trailing slash warnings)
  - Test validate() QoS level validation (prediction_qos, system_qos: 0,1,2 valid)
  - Test validate() invalid QoS levels (outside 0-2 range) error generation
  - Test validate() discovery settings with discovery_enabled=True
  - Test validate() discovery_prefix validation when discovery enabled
  - Test validate() default discovery settings (enabled=True, prefix="homeassistant")
  - Test test_connection() with missing broker (cannot test error)
  - Test test_connection() successful MQTT connection with on_connect callback
  - Test test_connection() connection failure scenarios (return codes 1-5)
  - Test test_connection() connection timeout (5 second wait with 0.1s intervals)
  - Test test_connection() authentication configuration (username/password)
  - Test test_connection() paho-mqtt import error graceful handling
  - Test test_connection() connection error messages for different return codes
  - Test test_connection() async connection setup and cleanup
  - Test test_connection() callback-based connection result tracking

- **RoomsConfigValidator Tests:**
  - Test validate() with complete rooms configuration (multiple rooms with sensors)
  - Test validate() with missing rooms section (empty dict, no rooms error)
  - Test validate() room count and sensor count statistics reporting
  - Test validate() low sensor count warning (<5 sensors total)
  - Test validate() room validation integration using _validate_room()
  - Test validate() result merging from individual room validations
  - Test _validate_room() with complete room configuration (name, sensors)
  - Test _validate_room() with missing room name (warning generation)
  - Test _validate_room() with no sensors configured (warning and early return)
  - Test _validate_room() sensor validation with dictionary and string formats
  - Test _validate_room() entity ID validation using _is_valid_entity_id()
  - Test _validate_room() essential sensor type checking (motion, occupancy, door)
  - Test _validate_room() missing essential sensor types warning
  - Test _validate_room() sensor count and type statistics reporting
  - Test _count_sensors() with dictionary sensor configurations
  - Test _count_sensors() with string sensor configurations
  - Test _count_sensors() mixed sensor type counting
  - Test _is_valid_entity_id() with valid entity ID patterns (domain.entity format)
  - Test _is_valid_entity_id() regex pattern validation for supported domains
  - Test _is_valid_entity_id() with invalid entity ID formats
  - Test _is_valid_entity_id() with edge case entity IDs (underscores, numbers)

- **SystemRequirementsValidator Tests:**
  - Test validate() Python version checking and reporting
  - Test validate() Python 3.9+ requirement (error for <3.9)
  - Test validate() Python 3.11+ recommendation (warning for 3.9-3.10)
  - Test validate() current Python version info reporting
  - Test validate() required package availability checking
  - Test validate() package list validation (asyncio, aiohttp, asyncpg, etc.)
  - Test validate() missing packages error reporting
  - Test validate() all packages available success message
  - Test validate() disk space checking using shutil.disk_usage()
  - Test validate() disk space requirements (error <1GB, warning <5GB)
  - Test validate() disk space info reporting in GB
  - Test validate() disk space check exception handling
  - Test validate() memory checking using psutil.virtual_memory()
  - Test validate() total memory requirements (error <2GB, warning <4GB)
  - Test validate() available memory warning (<0.5GB)
  - Test validate() memory info reporting (total and available GB)
  - Test validate() psutil import error graceful handling
  - Test validate() memory check general exception handling

- **ConfigurationValidator Tests:**
  - Test __init__() validator initialization (all sub-validators created)
  - Test validate_configuration() complete validation orchestration
  - Test validate_configuration() section validation execution (Home Assistant, Database, MQTT, Rooms, System)
  - Test validate_configuration() test_connections parameter behavior
  - Test validate_configuration() strict_mode parameter behavior (warnings -> errors)
  - Test validate_configuration() result merging from all sections
  - Test validate_configuration() section validation exception handling
  - Test validate_configuration() connection testing when enabled
  - Test validate_configuration() connection test exception handling for each service
  - Test validate_configuration() final validation summary (success/failure)
  - Test validate_configuration() strict mode warning promotion to errors
  - Test validate_configuration() comprehensive logging integration
  - Test validate_config_files() file path resolution for environments
  - Test validate_config_files() environment-specific config file selection
  - Test validate_config_files() fallback to base config.yaml when env config missing
  - Test validate_config_files() YAML file loading (config.yaml, rooms.yaml)
  - Test validate_config_files() file not found error handling
  - Test validate_config_files() YAML parsing error handling
  - Test validate_config_files() integration with validate_configuration()

**Integration Tests:**

- **End-to-End Validation Workflow:**
  - Test complete configuration validation with real config files
  - Test validation with environment-specific configurations
  - Test validation with connection testing enabled for all services
  - Test validation result aggregation across all validator components
  - Test strict mode behavior with comprehensive warning-to-error promotion

- **Cross-Validator Integration:**
  - Test validation consistency across multiple configuration sections
  - Test result merging behavior with mixed valid/invalid sections
  - Test error propagation and context preservation across validators
  - Test info/warning aggregation and comprehensive reporting

- **Real Service Connection Testing:**
  - Test Home Assistant connection validation with live API endpoints
  - Test database connection validation with real PostgreSQL/TimescaleDB
  - Test MQTT broker connection validation with real MQTT brokers
  - Test connection failure scenarios and error reporting
  - Test timeout handling in real network conditions

- **Configuration File Processing:**
  - Test YAML loading with complex nested configuration structures
  - Test environment variable integration in configuration validation
  - Test configuration file resolution across different environments
  - Test error recovery from malformed or missing configuration files

**Edge Cases:**

- **Configuration Data Edge Cases:**
  - Test validation with completely empty configuration objects
  - Test validation with null/None values in configuration sections
  - Test validation with extremely large configuration files
  - Test validation with deeply nested configuration structures
  - Test validation with Unicode characters in configuration values
  - Test validation with circular references in configuration data

- **Network and Connection Edge Cases:**
  - Test connection validation with network timeouts and interruptions
  - Test connection validation with SSL/TLS certificate issues
  - Test connection validation with proxy configurations
  - Test connection validation with rate limiting and throttling
  - Test connection validation with DNS resolution failures

- **System Resource Edge Cases:**
  - Test system validation on resource-constrained environments
  - Test system validation with missing system dependencies
  - Test system validation with permissions issues
  - Test system validation with disk space exhaustion scenarios
  - Test system validation with memory pressure conditions

- **File System Edge Cases:**
  - Test configuration file loading with permission denied scenarios
  - Test configuration file loading with symbolic link issues
  - Test configuration file loading with concurrent file modifications
  - Test configuration file loading with filesystem corruption
  - Test configuration file loading with very large files

- **Validation Logic Edge Cases:**
  - Test validation with boundary values for all numeric parameters
  - Test validation with extreme timeout values and connection parameters
  - Test validation with malformed URLs and connection strings
  - Test validation with invalid entity ID patterns and formats
  - Test validation with missing required vs optional configuration fields

**Error Handling:**

- **Configuration Loading Errors:**
  - Test FileNotFoundError handling for missing configuration files
  - Test YAML parsing errors with informative error messages
  - Test configuration structure validation with missing required sections
  - Test configuration type validation with invalid data types

- **Service Connection Errors:**
  - Test Home Assistant connection errors (authentication, network, API)
  - Test database connection errors (credentials, network, permissions)
  - Test MQTT broker connection errors (authentication, network, firewall)
  - Test service-specific error categorization and reporting

- **System Validation Errors:**
  - Test Python version validation errors with version incompatibility
  - Test package availability errors with missing dependencies
  - Test system resource validation errors (disk, memory, permissions)
  - Test import errors for optional dependencies (psutil, asyncpg)

- **Validation Framework Errors:**
  - Test exception handling in individual validator components
  - Test error propagation through validation result merging
  - Test validation failure recovery and partial result reporting
  - Test validation timeout handling for long-running checks

- **String Representation Errors:**
  - Test ValidationResult.__str__() with malformed error messages
  - Test formatting errors with special characters and Unicode
  - Test string representation with very large error/warning lists
  - Test formatting consistency across different result states

**Coverage Target: 85%+**

**Mock Requirements:**
- Mock requests.get() for Home Assistant API connection testing
- Mock asyncpg.connect() for database connection validation
- Mock paho.mqtt.client for MQTT broker connection testing
- Mock sys.version_info for Python version testing
- Mock shutil.disk_usage() for disk space validation
- Mock psutil.virtual_memory() for memory validation
- Mock Path.exists() and file operations for configuration loading
- Mock yaml.safe_load() for YAML parsing testing
- Mock __import__() for package availability testing
- Mock urlparse() for URL validation testing

**Test Fixtures Needed:**
- Sample configuration dictionaries for all validator types
- Sample YAML configuration files (valid and invalid)
- Mock API responses for Home Assistant connection testing
- Mock database connection responses and error scenarios
- Mock MQTT client connection scenarios and callbacks
- System resource fixtures (memory, disk space variations)
- Python version fixtures for compatibility testing
- Network error simulation fixtures for connection testing

**Special Testing Considerations:**
- Configuration validation requires comprehensive mocking of external services
- Connection testing needs proper timeout handling and async operation testing
- System requirements validation requires careful mocking of system resources
- File system operations need proper cleanup and exception simulation
- Validator integration testing requires realistic configuration scenarios
- Error message formatting and string representation need Unicode handling
- Network simulation for connection failure and timeout scenarios
- Cross-platform system validation considerations (Windows/Linux differences)
- Performance testing for validation of large configuration files
- Security testing to ensure sensitive information (passwords, tokens) is properly masked

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

### src/core/backup_manager.py - Backup Management System
**Classes Found:** BackupMetadata (dataclass), DatabaseBackupManager, ModelBackupManager, ConfigurationBackupManager, BackupManager  
**Methods Analyzed:** BackupMetadata.to_dict(), BackupMetadata.from_dict(), DatabaseBackupManager.__init__(), DatabaseBackupManager.create_backup(), DatabaseBackupManager.restore_backup(), DatabaseBackupManager._save_backup_metadata(), DatabaseBackupManager._load_backup_metadata(), ModelBackupManager.__init__(), ModelBackupManager.create_backup(), ModelBackupManager.restore_backup(), ModelBackupManager._save_backup_metadata(), ModelBackupManager._load_backup_metadata(), ConfigurationBackupManager.__init__(), ConfigurationBackupManager.create_backup(), ConfigurationBackupManager._save_backup_metadata(), BackupManager.__init__(), BackupManager.run_scheduled_backups(), BackupManager.cleanup_expired_backups(), BackupManager.list_backups(), BackupManager.get_backup_info(), BackupManager.restore_database_backup(), BackupManager.restore_models_backup(), BackupManager.create_disaster_recovery_package()

**Required Tests:**

**Unit Tests:**
- **BackupMetadata Dataclass Tests:**
  - Test BackupMetadata initialization with all required fields (backup_id, backup_type, timestamp, size_bytes, compressed)
  - Test BackupMetadata with optional fields (checksum, retention_date, tags)
  - Test to_dict() serialization with ISO timestamp formatting
  - Test to_dict() with None values for optional fields (retention_date, checksum)
  - Test to_dict() with empty tags dictionary default
  - Test from_dict() deserialization with complete data
  - Test from_dict() with missing optional fields
  - Test from_dict() timestamp parsing with datetime.fromisoformat()
  - Test from_dict() with None retention_date handling
  - Test from_dict() with missing tags defaulting to empty dict
  - Test roundtrip serialization/deserialization consistency

- **DatabaseBackupManager Tests:**
  - Test __init__() with backup directory creation and database config storage
  - Test create_backup() with auto-generated backup ID (db_YYYYMMDD_HHMMSS format)
  - Test create_backup() with custom backup_id parameter
  - Test create_backup() with compression enabled/disabled
  - Test database connection string parsing (postgresql+asyncpg://user:pass@host:port/dbname)
  - Test connection string parsing with missing components (default values)
  - Test pg_dump command construction with proper parameters (--verbose, --clean, --if-exists, --create)
  - Test pg_dump execution with environment variable PGPASSWORD setting
  - Test subprocess.run() with proper timeout (3600 seconds) and error handling
  - Test backup file compression with gzip when compress=True
  - Test temporary file cleanup after compression
  - Test backup metadata creation with proper fields (type="database", tags with database/host)
  - Test backup size calculation and metadata storage
  - Test backup cleanup on failure scenarios
  - Test restore_backup() with metadata loading and validation
  - Test restore_backup() with compressed/uncompressed file handling
  - Test psql command construction for restoration
  - Test SQL content reading (compressed vs uncompressed files)
  - Test database restoration with proper error handling
  - Test _save_backup_metadata() JSON file creation with proper formatting
  - Test _load_backup_metadata() with existing/missing metadata files

- **ModelBackupManager Tests:**
  - Test __init__() with backup and models directory setup
  - Test create_backup() with tar archive creation (models_YYYYMMDD_HHMMSS format)
  - Test create_backup() with compression (.tar vs .tar.gz)
  - Test models directory existence check and creation
  - Test tar command construction with proper parameters (-cf, -czf, -C options)
  - Test tar execution with timeout (1800 seconds) and error handling
  - Test backup metadata creation with type="models" and proper tags
  - Test restore_backup() with existing models directory backup
  - Test restore_backup() with tar extraction command construction
  - Test existing models directory backup with timestamp naming
  - Test shutil.move() for existing models backup
  - Test tar extraction with proper directory handling
  - Test backup cleanup on creation failures
  - Test metadata file operations for models

- **ConfigurationBackupManager Tests:**
  - Test __init__() with backup and config directory setup
  - Test create_backup() with config archive creation (config_YYYYMMDD_HHMMSS format)
  - Test tar command for configuration files with timeout (300 seconds)
  - Test configuration backup size calculation (bytes to KB conversion)
  - Test backup metadata creation with type="config"
  - Test configuration-specific backup parameters
  - Test _save_backup_metadata() for configuration backups

- **BackupManager Orchestration Tests:**
  - Test __init__() with sub-manager initialization (db, model, config managers)
  - Test backup configuration parameter extraction from config dict
  - Test run_scheduled_backups() with enabled/disabled backup configuration
  - Test scheduled backup interval logic (interval_hours, retention_days configuration)
  - Test database backup scheduling with configurable intervals
  - Test model backup scheduling (less frequent, model_backup_interval_hours)
  - Test configuration backup scheduling (daily at 2 AM)
  - Test retention date calculation for different backup types
  - Test cleanup_expired_backups() execution after backup creation
  - Test backup loop with asyncio.sleep() for interval management
  - Test backup failure handling and error logging
  - Test cleanup_expired_backups() with metadata file discovery (rglob pattern)
  - Test backup expiration logic with retention_date comparison
  - Test expired backup file removal (multiple extensions: .sql, .sql.gz, .tar, .tar.gz)
  - Test metadata file cleanup after backup removal
  - Test cleanup statistics tracking (count, size freed)
  - Test list_backups() with optional backup_type filtering
  - Test list_backups() sorting by timestamp (newest first)
  - Test get_backup_info() with specific backup_id lookup
  - Test restore_database_backup() delegation to DatabaseBackupManager
  - Test restore_models_backup() delegation to ModelBackupManager
  - Test create_disaster_recovery_package() with complete backup package creation
  - Test disaster recovery manifest creation with system info
  - Test disaster recovery package cleanup on failures

**Integration Tests:**
- **Complete Backup Workflow Integration:**
  - Test end-to-end database backup creation and restoration cycle
  - Test model backup with real file system operations and tar archives
  - Test configuration backup with actual config directory archiving
  - Test scheduled backup execution with real timer intervals (shortened for testing)
  - Test backup cleanup integration with file system operations
  - Test disaster recovery package creation with all backup types

- **File System Integration:**
  - Test backup directory creation with proper permissions
  - Test backup file operations (creation, compression, cleanup)
  - Test metadata file JSON operations with real file I/O
  - Test concurrent backup operations and file locking
  - Test disk space monitoring during backup operations
  - Test backup verification after creation (file existence, size validation)

- **Database Integration:**
  - Test pg_dump integration with actual PostgreSQL database
  - Test psql restoration with real database operations
  - Test TimescaleDB-specific backup considerations
  - Test database backup with large datasets and proper timeout handling
  - Test backup integrity verification with checksum validation

- **Configuration Integration:**
  - Test backup manager configuration with SystemConfig integration
  - Test backup scheduling configuration validation
  - Test environment-specific backup settings
  - Test backup retention policy enforcement across environments

**Edge Cases:**
- **File System Edge Cases:**
  - Test backup operations with insufficient disk space
  - Test backup directory permission issues (read-only, no write access)
  - Test backup file corruption scenarios and detection
  - Test very large backup files (multi-GB) and timeout handling
  - Test network storage backup locations and connectivity issues
  - Test concurrent backup operations and file locking conflicts
  - Test backup operations with special characters in file paths

- **Database Edge Cases:**
  - Test database backup with very large databases (timeout scenarios)
  - Test database connection failures during backup process
  - Test pg_dump/psql command failures with detailed error messages
  - Test database backup with active connections and lock conflicts
  - Test connection string parsing edge cases (malformed URLs, missing components)
  - Test database restoration with schema conflicts and version mismatches
  - Test backup/restore with TimescaleDB-specific features (hypertables, compression)

- **Compression Edge Cases:**
  - Test compression with very large files and memory constraints
  - Test compression failures and fallback to uncompressed backups
  - Test corrupted compressed files and error detection
  - Test compression ratio analysis and storage optimization

- **Scheduling Edge Cases:**
  - Test backup scheduling with system clock changes
  - Test backup scheduling with very short intervals (edge timing)
  - Test backup scheduling during system shutdown/restart
  - Test backup schedule conflicts (overlapping backup operations)
  - Test backup scheduling with timezone changes and daylight saving time

- **Retention Policy Edge Cases:**
  - Test retention policy with extreme date values (very old/future dates)
  - Test retention policy with backup metadata corruption
  - Test retention policy enforcement with partial cleanup failures
  - Test retention policy with mixed backup types and different retention periods

**Error Handling:**
- **Subprocess Error Handling:**
  - Test pg_dump subprocess failures with detailed error capture
  - Test tar command failures with proper error propagation
  - Test subprocess timeout handling (DatabaseError, backup cleanup)
  - Test subprocess permission errors and error context
  - Test command not found errors (pg_dump, psql, tar not installed)

- **File System Error Handling:**
  - Test file creation errors during backup process
  - Test file deletion errors during cleanup operations
  - Test JSON metadata file corruption and recovery
  - Test backup directory creation failures
  - Test file permission errors during backup/restore operations
  - Test disk full scenarios during backup creation

- **Database Error Handling:**
  - Test database connection failures with proper error context
  - Test authentication failures during database backup
  - Test database lock errors during backup/restore operations
  - Test schema validation errors during restoration
  - Test TimescaleDB extension errors and fallback behavior

- **Configuration Error Handling:**
  - Test invalid backup configuration parameter handling
  - Test missing backup directory configuration
  - Test invalid retention policy configuration
  - Test backup scheduling configuration validation errors

- **Disaster Recovery Error Handling:**
  - Test disaster recovery package creation failures
  - Test partial backup failures in disaster recovery
  - Test manifest file creation errors
  - Test disaster recovery package corruption detection

**Coverage Target: 85%+**

**Mock Requirements:**
- Mock subprocess.run() for pg_dump, psql, and tar command execution
- Mock Path.exists(), Path.mkdir(), Path.stat() for file system operations
- Mock open() and file operations for metadata and SQL file handling
- Mock gzip.open() for compression operations
- Mock shutil.move(), shutil.copyfileobj(), shutil.rmtree() for file operations
- Mock datetime.now() for timestamp generation and testing
- Mock os.environ for PGPASSWORD environment variable testing
- Mock asyncio.sleep() for scheduled backup loop testing

**Test Fixtures Needed:**
- Sample database configuration with connection string variations
- Mock pg_dump/psql output for successful and failed operations
- Sample backup metadata JSON files for various backup types
- Mock tar command output for successful and failed operations
- Sample disaster recovery package manifest structures
- File system permission testing fixtures
- Backup scheduling configuration fixtures

**Special Testing Considerations:**
- Subprocess testing requires careful command line argument validation
- File system testing needs proper cleanup in test teardown
- Backup scheduling testing requires time manipulation for realistic testing
- Database backup testing should use test databases to avoid data loss
- Compression testing needs to verify actual file compression ratios
- Disaster recovery testing requires comprehensive package validation
- Error scenarios must test proper cleanup and resource management
- Concurrent backup testing requires thread safety validation

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

**Mock Requirements:**
- Mock HomeAssistantClient and all its methods (connect, disconnect, get_entity_history)
- Mock EventProcessor and process_event_batch method
- Mock get_db_session context manager and session operations
- Mock pickle.load and pickle.dump for resume functionality
- Mock Path.exists, Path.mkdir, and file operations
- Mock datetime.utcnow for deterministic timestamp testing
- Mock asyncio.Semaphore, asyncio.gather, and asyncio.sleep
- Mock psutil.Process for memory usage monitoring
- Mock get_config() for system configuration
- Mock SensorEvent model and bulk insert operations

**Test Fixtures Needed:**
- Sample ImportConfig objects with various parameter combinations
- Mock Home Assistant historical data responses (valid and invalid)
- Sample SystemConfig with room and entity configurations
- Mock database sessions and query results
- Sample HAEvent and SensorEvent objects for conversion testing
- Resume data file fixtures for testing serialization
- Progress callback fixtures (sync and async functions)
- Exception scenarios for all error handling paths

**Special Testing Considerations:**
- Bulk import requires careful mocking of large data volumes without actual data transfer
- Asynchronous operations need proper asyncio event loop management in tests
- Database operations require transaction testing and rollback behavior validation
- Progress tracking needs time-based validation with controlled datetime mocking
- Resume functionality requires filesystem operation mocking and state persistence testing
- Memory usage monitoring needs psutil mocking for cross-platform compatibility
- Error propagation testing requires comprehensive exception chaining validation
- Performance optimization testing needs realistic throughput simulation
- Large file backup testing needs memory usage monitoring
- Cross-platform testing for subprocess command compatibility (Windows/Unix)

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

**Mock Requirements:**
- Mock datetime.now() and datetime.fromisoformat() for timestamp testing
- Mock statistics.mean(), statistics.median(), statistics.stdev() for deterministic testing
- Mock scipy.stats.shapiro() for normality testing (when SCIPY_AVAILABLE)
- Mock numpy operations when SKLEARN_AVAILABLE=False
- Mock logging.getLogger() for error logging validation
- Mock defaultdict and deque for collection behavior testing

**Test Fixtures Needed:**
- Sample sensor event datasets with various patterns (normal, anomalous, corrupted)
- Timestamp fixtures with edge cases (timezones, formats, edge dates)
- State distribution fixtures with different sensor types and patterns
- Corruption examples for each detection method (timestamp, state, ID, encoding)
- Quality metrics fixtures with known expected outcomes
- Sensor baseline fixtures for malfunction detection testing
- Expected sensor sets for completeness calculation testing

**Special Testing Considerations:**
- Pattern detection requires careful statistical validation with known datasets
- Corruption detection needs comprehensive test data covering all corruption types
- Quality monitoring requires time-based testing with proper datetime mocking
- Statistical calculations need edge case testing for mathematical stability
- Memory management testing for long-running pattern detection scenarios
- Performance testing with realistic sensor data volumes
- Integration testing with scipy/sklearn availability scenarios

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

**Mock Requirements:**
- Mock SQLAlchemy Engine objects with various URL types and attributes
- Mock model classes with __table__ attributes and column objects
- Mock database connection objects (dbapi_connection) for event listener testing
- Mock cursor objects for PRAGMA execution testing
- Mock src.data.storage.models import for error handling scenarios
- Mock SQLAlchemy constraint and index classes (UniqueConstraint, Index)
- Mock connection_record objects with info dictionaries for event listeners

**Test Fixtures Needed:**
- Sample engine objects for SQLite, PostgreSQL, and unknown database types
- Mock model classes with various table configurations
- Sample URL strings in different formats (file paths, network URLs, memory databases)
- Sample constraint and index objects for table argument testing
- Database connection mock objects with proper cursor interfaces
- Model table metadata fixtures with various column configurations

**Special Testing Considerations:**
- Test event listener registration and execution with SQLAlchemy's event system
- Verify that database-specific configurations actually work with real database connections
- Test thread safety of model configuration modifications
- Test compatibility with different SQLAlchemy versions for event handling
- Test monkey-patching behavior and side effects on global model state
- Performance testing for table argument generation with complex schemas
- Test that PostgreSQL-specific features (partial indexes, USING clause) are properly generated

### src/features/sequential.py - Sequential Pattern Features
**Classes Found:** SequentialFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_room_transition_features, _extract_velocity_features, _extract_sensor_sequence_features, _extract_cross_room_features, _extract_movement_classification_features, _extract_ngram_features, _create_sequences_for_classification, _create_movement_sequence, _get_default_features, get_feature_names, clear_cache

**Required Tests:**

- Unit Tests: 
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

- Integration Tests:
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

- Edge Cases:
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

- Error Handling:
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

- Coverage Target: 85%+

**Mock Requirements:**
- Mock MovementPatternClassifier for classification testing
- Mock numpy operations for mathematical edge case testing  
- Mock statistics functions for deterministic calculations
- Mock logging.getLogger() for error logging validation
- Mock datetime operations for timestamp testing
- Mock Counter and defaultdict for collection behavior testing

**Test Fixtures Needed:**
- Sample SensorEvent sequences with various patterns (room transitions, velocity patterns, sensor sequences)
- RoomConfig fixtures with different sensor configurations
- SystemConfig fixtures with and without MovementPatternClassifier
- Timestamp fixtures for target_time and lookback testing
- Movement pattern fixtures matching HUMAN_MOVEMENT_PATTERNS and CAT_MOVEMENT_PATTERNS constants
- N-gram pattern fixtures for sequence analysis testing
- Cross-room correlation fixtures with multi-room scenarios

**Special Testing Considerations:**
- Sequential feature extraction requires careful temporal ordering validation
- Mathematical operations need comprehensive edge case testing for stability
- Pattern matching requires validation against movement constants
- Memory usage testing for large event sequences and caching behavior
- Performance testing with realistic sensor data volumes and complex room configurations
- Integration testing with all dependent modules and external libraries

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

### src/features/temporal.py - Temporal Feature Extraction
**Classes Found:** TemporalFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_time_since_features, _extract_duration_features, _extract_generic_sensor_features, _extract_cyclical_features, _extract_historical_patterns, _extract_transition_timing_features, _extract_room_state_features, _get_default_features, get_feature_names, validate_feature_names, clear_cache, extract_batch_features

**Required Tests:**
- Unit Tests: 
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

- Integration Tests:
  - Test integration with SensorEvent model attributes and methods
  - Test integration with RoomState model and database objects
  - Test integration with TEMPORAL_FEATURE_NAMES constants
  - Test integration with FeatureExtractionError exception handling
  - Test pandas and numpy dependency integration
  - Test timezone handling with real datetime objects
  - Test feature extraction pipeline with realistic sensor data sequences
  - Test batch processing performance with large event datasets

- Edge Cases:
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

- Error Handling:
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

- Coverage Target: 85%+

**Test Fixtures Needed:**
- Multi-day sensor event sequences with various state patterns
- Events with mixed sensor types (motion, door, presence)
- Events with complex attribute structures (nested dicts, mixed types)
- Room state sequences with occupancy confidence values
- Events spanning timezone changes and daylight saving transitions
- Malformed event data for error handling validation
- Mock objects simulating database model behavior
- Statistical edge case datasets (identical values, extreme outliers)
- Chronologically disordered event sequences for sorting validation
- Large batch datasets for performance and memory testing

**Special Testing Considerations:**
- Temporal feature extraction requires precise datetime handling and timezone awareness
- Statistical calculations need validation against known datasets with expected outcomes
- Pandas/numpy integration requires mocking for error scenario testing
- Cyclical encoding tests need mathematical validation of sin/cos transformations
- Historical pattern analysis requires time-series data with predictable statistical properties
- Batch processing tests need memory usage monitoring and performance benchmarking
- Integration with database models requires proper mock setup for attributes and methods

### src/features/contextual.py - Contextual Features
**Classes Found:** ContextualFeatureExtractor
**Methods Analyzed:** __init__, extract_features, _extract_environmental_features, _extract_door_state_features, _extract_multi_room_features, _extract_seasonal_features, _extract_sensor_correlation_features, _extract_room_context_features, _extract_numeric_values, _is_realistic_value, _calculate_trend, _calculate_change_rate, _calculate_room_activity_correlation, _calculate_room_state_correlation, _calculate_natural_light_score, _calculate_light_change_rate, _get_default_features, get_feature_names, clear_cache, _filter_environmental_events, _filter_door_events

**Required Tests:**
- Unit Tests:
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

- Integration Tests:
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

- Edge Cases:
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

- Error Handling:
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

- Coverage Target: 85%+

**Test Fixtures Needed:**
- Multi-room sensor event sequences with temperature, humidity, light, and door sensors
- Events with various SensorType enum values and edge case sensor types
- Room state histories with occupancy transitions and confidence values
- Environmental sensor data with values at threshold boundaries
- Door sensor events with "on"/"off" and "open"/"closed" state variations
- Seasonal test data covering all months and holiday periods
- Correlation test datasets with known statistical relationships
- Malformed sensor events for error handling validation
- Time-series data spanning different temporal scales (minutes to months)
- Mock SystemConfig and RoomConfig objects with realistic sensor mappings

**Special Testing Considerations:**
- Contextual feature extraction requires extensive environmental sensor simulation
- Statistical correlation analysis needs mathematically validated test datasets
- Multi-room occupancy patterns require realistic behavioral modeling
- Environmental thresholds need validation against real-world sensor ranges
- Seasonal features require time-aware test fixtures spanning calendar years
- Door state tracking requires proper binary sensor state modeling
- Sliding window algorithms need performance testing with large datasets
- Natural light scoring requires time-of-day pattern validation
- SensorType enum integration requires comprehensive sensor type coverage
- Feature name consistency requires validation against default feature dictionary
- Error handling tests must cover all exception paths with proper logging validation

### src/models/base/xgboost_predictor.py - XGBoost Gradient Boosting
**Classes Found:** XGBoostPredictor
**Methods Analyzed:** __init__, train, predict, get_feature_importance, get_feature_importance_plot_data, _prepare_targets, _determine_transition_type, _calculate_confidence, _get_feature_contributions, get_learning_curve_data, get_model_complexity, save_model, load_model

**Required Tests:**
- Unit Tests:
  - Test __init__ with default parameters and custom kwargs override
  - Test __init__ with room_id parameter and DEFAULT_MODEL_PARAMS integration
  - Test __init__ model_params validation with all XGBoost parameters
  - Test train with minimum required parameters (features, targets)
  - Test train with validation data provided and eval_set creation
  - Test train with insufficient training data (< 10 samples) raises ModelTrainingError
  - Test train feature scaling with StandardScaler fit_transform
  - Test train XGBRegressor creation with all model parameters
  - Test train model fitting with eval_set and verbose=False
  - Test train feature importance extraction and storage
  - Test train training metrics calculation (MAE, RMSE, R2)
  - Test train validation metrics calculation when validation data provided
  - Test train TrainingResult creation with all metrics
  - Test train training_history appending and logging
  - Test predict with untrained model raises ModelPredictionError
  - Test predict with invalid features raises ModelPredictionError
  - Test predict feature scaling and DataFrame transformation
  - Test predict XGBoost model prediction and result processing
  - Test predict time clipping (60-86400 seconds) and datetime calculation
  - Test predict transition type determination logic
  - Test predict confidence calculation with various scenarios
  - Test predict feature contributions calculation
  - Test predict PredictionResult creation with all metadata
  - Test predict prediction recording for accuracy tracking
  - Test get_feature_importance with trained and untrained model
  - Test get_feature_importance_plot_data sorting and tuple format
  - Test _prepare_targets with time_until_transition_seconds column
  - Test _prepare_targets with next_transition_time and target_time calculation
  - Test _prepare_targets with default single column fallback
  - Test _prepare_targets value clipping (60-86400 bounds)
  - Test _determine_transition_type with occupied/vacant/unknown states
  - Test _determine_transition_type feature-based inference logic
  - Test _determine_transition_type time-based fallback logic
  - Test _calculate_confidence with various model states
  - Test _calculate_confidence with extreme predictions and adjustments
  - Test _calculate_confidence with feature variance calculations
  - Test _calculate_confidence with prediction value normalization
  - Test _get_feature_contributions with feature importance multiplication
  - Test _get_feature_contributions top 10 sorting by absolute value
  - Test get_learning_curve_data with eval_results_ processing
  - Test get_model_complexity with all model parameters
  - Test save_model pickle serialization with all components
  - Test load_model pickle deserialization and state restoration
  - Test load_model TrainingResult reconstruction from dictionaries

- Integration Tests:
  - Test integration with BasePredictor inheritance and methods
  - Test integration with ModelType.XGBOOST enum and constants
  - Test integration with DEFAULT_MODEL_PARAMS configuration
  - Test integration with StandardScaler from sklearn.preprocessing
  - Test integration with XGBRegressor from xgboost library
  - Test integration with ModelTrainingError and ModelPredictionError exceptions
  - Test integration with PredictionResult and TrainingResult dataclasses
  - Test integration with pandas DataFrame operations throughout
  - Test integration with numpy array operations and clipping
  - Test integration with datetime timezone handling
  - Test feature validation through BasePredictor.validate_features
  - Test prediction recording through BasePredictor._record_prediction
  - Test model versioning through BasePredictor._generate_model_version
  - Test logging integration with proper room_id context

- Edge Cases:
  - Test with empty features DataFrame
  - Test with single feature column
  - Test with features containing NaN or infinite values
  - Test with targets containing negative or zero values
  - Test with extremely large target values (> 24 hours)
  - Test with features having duplicate column names
  - Test with mismatched features/targets row counts
  - Test with validation data having different feature columns
  - Test with model parameters containing invalid values
  - Test with save_model using invalid file paths or permissions
  - Test with load_model using non-existent or corrupted files
  - Test with pickle serialization failures
  - Test with XGBoost model creation failures
  - Test with feature scaling failures (constant features, etc.)
  - Test with confidence calculation edge cases (NaN values, extreme features)
  - Test with feature importance calculation when model not fitted
  - Test with very short or very long time predictions
  - Test with prediction at timezone boundaries
  - Test with feature contributions when importance is empty

- Error Handling:
  - Test ModelTrainingError raised with proper model_type and room_id
  - Test ModelPredictionError raised with proper context information
  - Test exception handling during XGBRegressor creation and fitting
  - Test exception handling during feature scaling operations
  - Test exception handling during prediction generation
  - Test exception handling during confidence calculation
  - Test exception handling during feature contribution calculation
  - Test exception handling during model save/load operations
  - Test logging of error messages with appropriate levels
  - Test graceful handling of pickle import/export failures
  - Test handling of corrupted model files during loading
  - Test handling of version compatibility issues in saved models
  - Test exception propagation from BasePredictor methods

- Coverage Target: 85%+

**Test Fixtures Needed:**
- Sample feature DataFrames with various column types and sizes
- Target DataFrames with different column structures and time ranges
- Mock XGBRegressor objects with controllable fit/predict behavior
- Sample training/validation datasets for realistic model testing
- Corrupted pickle files for load failure testing
- Edge case datasets with extreme values, NaN, and infinite values
- Mock StandardScaler objects for testing scaling failures
- Datetime objects spanning various timezones and time ranges
- Feature importance dictionaries with varying distribution patterns
- Training history objects for confidence calculation testing

**Special Testing Considerations:**
- XGBoost integration requires proper mocking of xgb.XGBRegressor methods
- Feature scaling tests need StandardScaler behavior validation
- Confidence calculation involves complex mathematical operations requiring precise testing
- Model serialization tests need temporary file handling and cleanup
- Training result validation requires checking all metric calculations
- Prediction clipping and datetime arithmetic need timezone-aware testing
- Feature contribution calculation requires understanding of SHAP-like interpretability
- Integration with BasePredictor requires testing inherited functionality
- Error propagation testing needs validation of custom exception types

### src/features/store.py - Feature Store Management
**Classes Found:** FeatureRecord, FeatureCache, FeatureStore
**Methods Analyzed:** FeatureRecord.to_dict(), FeatureRecord.from_dict(), FeatureRecord.is_valid(), FeatureCache._make_key(), FeatureCache.get(), FeatureCache.put(), FeatureCache.clear(), FeatureCache.get_stats(), FeatureStore.__init__(), FeatureStore.initialize(), FeatureStore.get_features(), FeatureStore.get_batch_features(), FeatureStore.compute_training_data(), FeatureStore._compute_features(), FeatureStore._get_data_for_features(), FeatureStore._get_features_from_db(), FeatureStore._persist_features_to_db(), FeatureStore._compute_data_hash(), FeatureStore.get_stats(), FeatureStore.clear_cache(), FeatureStore.reset_stats(), FeatureStore.health_check(), FeatureStore.get_statistics(), FeatureStore.__aenter__(), FeatureStore.__aexit__()

**Required Tests:**
- Unit Tests: 
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

- Integration Tests:
  - Test FeatureStore with real DatabaseManager integration for data persistence
  - Test FeatureStore with FeatureEngineeringEngine integration for feature computation
  - Test end-to-end feature computation from database query to cached result
  - Test batch feature extraction with real database queries and concurrent processing
  - Test training data generation with actual sensor events and room states
  - Test feature cache persistence across FeatureStore lifecycle
  - Test FeatureStore with multiple rooms and overlapping time windows
  - Test async database operations with connection pooling and error recovery

- Edge Cases:
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

- Error Handling:
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

- Coverage Target: 85%+

### src/models/base/lstm_predictor.py - LSTM Neural Networks
**Classes Found:** LSTMPredictor
**Methods Analyzed:** __init__, train, predict, get_feature_importance, _create_sequences, _calculate_confidence, get_model_complexity, save_model, load_model, incremental_update

**Required Tests:**
- Unit Tests:
  - **LSTMPredictor.__init__() Tests:**
    - Test initialization with default parameters (no room_id, no kwargs)
    - Test initialization with room_id parameter
    - Test initialization with custom kwargs overriding DEFAULT_MODEL_PARAMS
    - Test hidden_units parameter handling (int vs list conversion)
    - Test model_params dictionary structure and all parameter aliases (hidden_size, lstm_units, etc.)
    - Test sequence_length and sequence_step initialization
    - Test feature_scaler (StandardScaler) and target_scaler (MinMaxScaler) initialization
    - Test training statistics lists initialization (training_loss_history, validation_loss_history)
    - Test parameter alias handling (dropout vs dropout_rate)
    - Test MLPRegressor model initialization as None
    - Test BasePredictor parent class initialization with ModelType.LSTM

  - **train() Method Core Logic Tests:**
    - Test training with valid features and targets DataFrames
    - Test adaptive sequence length reduction for small datasets (<200 samples)
    - Test sequence_step adjustment for small datasets (step = 1)
    - Test minimum sequence requirement validation (< 2 sequences raises ModelTrainingError)
    - Test _create_sequences() integration and sequence generation logging
    - Test feature and target scaling with StandardScaler and MinMaxScaler
    - Test validation data preparation when provided (validation_features, validation_targets)
    - Test MLPRegressor creation with proper parameters from model_params
    - Test model training with fit() and random_state=42, warm_start=False
    - Test feature_names storage from input DataFrame columns
    - Test training metrics calculation (MAE, RMSE, R²) with inverse_transform
    - Test validation metrics calculation when validation data provided
    - Test model state updates (is_trained=True, training_date, model_version)
    - Test TrainingResult creation with comprehensive training metrics
    - Test training_history append operation
    - Test success logging with training time and scores
    - Test sequence length restoration in finally block

  - **train() Error Handling Tests:**
    - Test ModelTrainingError raising with proper model_type and room_id
    - Test error handling with malformed features/targets DataFrames
    - Test insufficient sequence data error handling
    - Test scaler fitting failures and error propagation
    - Test MLPRegressor training failures and exception handling
    - Test TrainingResult creation for failed training with error_message
    - Test training_history append for failed attempts
    - Test sequence length restoration after training failures

  - **predict() Method Core Logic Tests:**
    - Test prediction validation (is_trained and model existence checks)
    - Test feature validation with validate_features() integration
    - Test training_sequence_length usage for prediction consistency
    - Test sequence creation for prediction with padding for insufficient history
    - Test feature sequence flattening for MLPRegressor input format
    - Test feature scaling with transform() using fitted scaler
    - Test model prediction with predict() method
    - Test target inverse_transform for actual time values
    - Test time bounds clipping (60 seconds to 86400 seconds)
    - Test predicted_time calculation with timedelta addition
    - Test transition_type determination based on current_state
    - Test default transition_type logic based on hour of day (6-22 vs nighttime)
    - Test confidence calculation with _calculate_confidence()
    - Test PredictionResult creation with comprehensive metadata
    - Test prediction recording with _record_prediction()
    - Test multiple predictions for DataFrame with multiple rows

  - **predict() Error Handling Tests:**
    - Test ModelPredictionError for untrained model
    - Test ModelPredictionError for failed feature validation
    - Test exception handling in prediction loop
    - Test error logging and ModelPredictionError raising with cause

  - **get_feature_importance() Tests:**
    - Test feature importance calculation with trained model
    - Test empty dict return for untrained model
    - Test neural network weight analysis using model.coefs_[0]
    - Test feature importance calculation across sequence timesteps
    - Test importance normalization to sum to 1
    - Test input layer weight averaging across hidden units
    - Test feature name mapping with sequence length consideration
    - Test exception handling and warning logging for calculation failures

  - **_create_sequences() Core Logic Tests:**
    - Test sequence creation with valid features and targets DataFrames
    - Test input validation (equal length features/targets, minimum sequence_length)
    - Test target value extraction from different column formats (time_until_transition_seconds, next_transition_time/target_time, default)
    - Test target value validation and numeric conversion with pd.to_numeric()
    - Test sequence generation with corrected bounds checking (sequence_length to len(features))
    - Test sequence step handling with self.sequence_step
    - Test X_seq flattening for MLPRegressor compatibility
    - Test target value bounds filtering (60 to 86400 seconds)
    - Test sequence validation and array creation
    - Test final validation of X_array and y_array shapes
    - Test sequence generation logging

  - **_create_sequences() Error Handling Tests:**
    - Test ValueError for mismatched feature/target lengths
    - Test ValueError for insufficient data length
    - Test ValueError for non-numeric target values
    - Test ValueError for no valid sequences generated
    - Test target value calculation edge cases and error handling

  - **_calculate_confidence() Tests:**
    - Test confidence calculation using training history validation scores
    - Test fallback to training_score when validation_score unavailable
    - Test default confidence (0.7) when no training history
    - Test prediction reasonableness adjustment (extreme values reduce confidence)
    - Test confidence clipping to range [0.1, 0.95]
    - Test target_scaler inverse_transform integration for prediction evaluation
    - Test exception handling with default confidence return (0.7)

  - **Model Persistence Tests:**
    - Test save_model() with valid file path and model state
    - Test save_model() pickle serialization of complete model state
    - Test save_model() inclusion of all necessary components (model, scalers, metadata)
    - Test save_model() error handling and logging for save failures
    - Test load_model() with valid pickle file
    - Test load_model() complete state restoration (model, scalers, parameters, history)
    - Test load_model() TrainingResult reconstruction from history data
    - Test load_model() error handling and logging for load failures

  - **get_model_complexity() Tests:**
    - Test complexity information for trained models
    - Test parameter counting from model.coefs_ and model.intercepts_
    - Test complexity dictionary structure with all required fields
    - Test empty dict return for untrained models
    - Test total_parameters calculation accuracy

  - **incremental_update() Core Logic Tests:**
    - Test incremental update with valid new features/targets
    - Test fallback to full training for untrained models
    - Test minimum data requirement validation (< 5 samples error)
    - Test sequence creation from new data with _create_sequences()
    - Test feature scaling using existing fitted scalers
    - Test MLPRegressor parameter adjustment for incremental learning
    - Test warm_start=True and reduced max_iter for incremental updates
    - Test performance calculation on new data with metrics
    - Test model version updating with timestamp suffix
    - Test TrainingResult creation for incremental updates
    - Test training_history append for incremental results

  - **incremental_update() Error Handling Tests:**
    - Test ModelTrainingError for insufficient new data
    - Test ModelTrainingError for sequence creation failures
    - Test exception handling during incremental training
    - Test error logging and TrainingResult creation for failures

- Integration Tests:
  - **Model Training Integration:**
    - Test complete training workflow with realistic sensor event data
    - Test training with actual pandas DataFrames from feature extraction
    - Test integration with DEFAULT_MODEL_PARAMS from core.constants
    - Test integration with ModelType enum and BasePredictor inheritance
    - Test training with various DataFrame column formats and structures
    - Test validation split and cross-validation integration

  - **Model Prediction Integration:**
    - Test prediction pipeline with real feature DataFrames
    - Test integration with PredictionResult and TrainingResult classes
    - Test prediction recording and accuracy tracking integration
    - Test confidence calculation with real model performance data
    - Test transition type logic with actual room occupancy patterns

  - **Persistence Integration:**
    - Test model saving/loading with complete training state
    - Test pickle serialization compatibility across sessions
    - Test model version tracking and history persistence
    - Test scaler state preservation and restoration

  - **Error Integration:**
    - Test ModelTrainingError and ModelPredictionError integration
    - Test error propagation to parent BasePredictor methods
    - Test logging integration with structured error messages

- Edge Cases:
  - **Data Quality Edge Cases:**
    - Test with features/targets having NaN or infinite values
    - Test with features having zero variance (constant columns)
    - Test with targets having extreme outliers or impossible values
    - Test with very small datasets (< 10 samples) and sequence adaptation
    - Test with very large datasets and memory management
    - Test with features having different column counts across calls
    - Test with non-numeric data in features requiring error handling

  - **Sequence Processing Edge Cases:**
    - Test sequence creation with edge sequence_length values (1, 2, very large)
    - Test sequence creation with sequence_step larger than data length
    - Test sequence generation with identical timestamps
    - Test sequence flattening with different feature dimensionalities
    - Test target value extraction with edge timestamp formats

  - **Model Training Edge Cases:**
    - Test MLPRegressor with edge case parameters (single neuron, large networks)
    - Test training convergence with difficult optimization landscapes
    - Test scaler fitting with constant or near-constant data
    - Test model training with perfect correlations or rank-deficient data
    - Test adaptive sequence length with boundary conditions

  - **Prediction Edge Cases:**
    - Test prediction with feature sequences shorter than training sequence length
    - Test prediction with features having different column orders
    - Test prediction confidence at extreme ends of trained data range
    - Test transition type logic at hour boundaries (6am, 10pm)
    - Test time clipping at exact boundaries (60 seconds, 24 hours)

  - **Persistence Edge Cases:**
    - Test save/load with corrupted pickle files
    - Test save/load with models having no training history
    - Test save/load with different Python/library versions compatibility
    - Test file system edge cases (permissions, disk space, long paths)

- Error Handling:
  - **Training Error Scenarios:**
    - Test ModelTrainingError with various underlying causes
    - Test scaler fitting failures with appropriate error propagation
    - Test MLPRegressor initialization failures
    - Test sequence generation failures with informative error messages
    - Test training timeout or memory exhaustion scenarios

  - **Prediction Error Scenarios:**
    - Test ModelPredictionError for various failure modes
    - Test feature validation failures and error messaging
    - Test scaler transform failures on new data
    - Test model prediction failures with corrupted model state
    - Test confidence calculation failures with fallback handling

  - **Data Validation Errors:**
    - Test comprehensive input validation for all public methods
    - Test pandas DataFrame structure validation
    - Test timestamp and numeric data validation
    - Test feature-target alignment validation
    - Test sequence bounds checking and error reporting

  - **Resource Management Errors:**
    - Test memory allocation failures during training/prediction
    - Test file I/O failures during model persistence
    - Test cleanup after various failure scenarios
    - Test state consistency after partial failures

- Coverage Target: 85%+

**Mock Requirements:**
- Mock MLPRegressor from sklearn.neural_network for training behavior testing
- Mock StandardScaler and MinMaxScaler for deterministic scaling behavior
- Mock pandas DataFrame and numpy array operations for edge case testing
- Mock pickle operations for save/load error scenario testing
- Mock datetime.now() for consistent timestamp testing
- Mock logging.getLogger() for error logging validation
- Mock DEFAULT_MODEL_PARAMS for parameter testing without external dependencies

**Test Fixtures Needed:**
- Sample feature/target DataFrames with various formats and structures
- Training data with known sequence patterns for validation
- Mock model states for testing persistence operations
- Error scenario datasets (corrupted, invalid, edge case data)
- Performance benchmarking datasets for large-scale testing
- Cross-validation datasets for model evaluation
- Timestamp and occupancy pattern fixtures for prediction testing

**Special Testing Considerations:**
- LSTM predictor uses MLPRegressor instead of actual LSTM, requiring tests to validate this architectural choice
- Sequence processing requires careful validation of flattening and padding logic
- Neural network feature importance approximation needs mathematical validation
- Incremental learning simulation needs testing against actual online learning scenarios
- Model persistence needs comprehensive state validation to ensure complete restoration
- Performance testing with realistic sensor data volumes and sequence lengths
- Memory usage monitoring for large sequence processing and model training
- Integration testing with complete feature extraction pipeline and prediction workflows

### src/features/engineering.py - Feature Engineering Pipeline
**Classes Found:** FeatureEngineeringEngine
**Methods Analyzed:** __init__, extract_features, extract_batch_features, _extract_features_parallel, _extract_features_sequential, _add_metadata_features, get_feature_names, create_feature_dataframe, _get_default_features, get_extraction_stats, reset_stats, clear_caches, validate_configuration, compute_feature_correlations, analyze_feature_importance, _validate_configuration, compute_feature_statistics, _calculate_skewness, _calculate_kurtosis, _calculate_entropy, _count_outliers, __del__

**Required Tests:**
- Unit Tests:
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

- Integration Tests:
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

- Edge Cases:
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

- Error Handling:
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

- Coverage Target: 85%+

**Test Fixtures Needed:**
- Mock SystemConfig with various room configurations
- Mock SensorEvent and RoomState objects with realistic attributes
- Sample extraction request tuples for batch processing
- Feature dictionaries with consistent and inconsistent key sets
- Pandas DataFrames with correlation test data and edge cases
- Target data with datetime and numeric format variations
- ThreadPoolExecutor mock for testing parallel execution paths
- Statistical test datasets for skewness, kurtosis, entropy validation
- Large-scale batch data for performance and memory testing
- Error-inducing data for exception handling validation

**Special Testing Considerations:**
- Feature engineering pipeline requires comprehensive integration testing with all three extractor types
- Parallel processing tests need careful asyncio and ThreadPoolExecutor mock management
- Statistical calculations require mathematical validation against known datasets
- Configuration validation needs testing of both None and valid config scenarios
- Pandas/numpy integration requires extensive edge case testing with invalid data
- Correlation and importance analysis need datasets with known statistical properties
- Memory usage monitoring critical for batch processing tests
- Error propagation testing must cover all async/parallel execution paths
- Integration with individual extractors requires consistent mock interface implementation

### src/models/ensemble.py - Ensemble Model Architecture
**Classes Found:** OccupancyEnsemble, utility functions (_ensure_timezone_aware, _safe_time_difference)
**Methods Analyzed:** __init__, train, predict, get_feature_importance, incremental_update, _train_base_models_cv, _train_meta_learner, _train_base_models_final, _calculate_model_weights, _create_meta_features, _select_important_features, _predict_ensemble, _combine_predictions, _calculate_ensemble_confidence, _prepare_targets, _validate_training_data, get_ensemble_info, save_model, load_model

**Required Tests:**
- Unit Tests: 
  * OccupancyEnsemble.__init__() - test initialization with various parameters, default params, tracking manager integration
  * train() - test complete training pipeline, insufficient data handling, cross-validation, meta-learner training, validation scenarios
  * predict() - test ensemble prediction generation, base model failure handling, meta-feature creation, result combination
  * get_feature_importance() - test weighted feature importance calculation from base models
  * incremental_update() - test online learning updates, model weight recalculation, dimension handling
  * _train_base_models_cv() - test cross-validation training, fold processing, error handling for failed models
  * _train_meta_learner() - test meta-learner training, feature scaling, dimension alignment, NaN handling
  * _train_base_models_final() - test concurrent base model training, performance metric collection
  * _calculate_model_weights() - test weight calculation based on prediction consistency and accuracy
  * _create_meta_features() - test meta-feature creation, dimension alignment, feature scaling edge cases
  * _select_important_features() - test feature selection strategies, column renaming, dimension limits
  * _predict_ensemble() - test ensemble prediction generation for training evaluation
  * _combine_predictions() - test combination of base and meta-learner predictions, confidence calculation
  * _calculate_ensemble_confidence() - test confidence calculation with GP uncertainty, prediction agreement
  * _prepare_targets() - test target value extraction from different DataFrame formats
  * _validate_training_data() - test comprehensive data validation, error conditions
  * get_ensemble_info() - test ensemble metadata retrieval
  * save_model()/load_model() - test model persistence and restoration
  * _ensure_timezone_aware()/_safe_time_difference() - test timezone handling utilities
- Integration Tests:
  * Full training pipeline with real base models (LSTM, XGBoost, HMM, GP)
  * End-to-end prediction workflow with feature engineering integration
  * Tracking manager integration for automatic accuracy tracking
  * Model serialization/deserialization with complex ensemble state
  * Concurrent base model training with various failure scenarios
  * Cross-validation performance across different data distributions
  * Meta-learner training with different learner types (RandomForest, LinearRegression)
- Edge Cases:
  * Training with insufficient data (< 50 samples)
  * All base models failing during prediction
  * Meta-feature dimension mismatches during scaling
  * NaN values in features, targets, and meta-features
  * Empty or zero-length prediction arrays
  * Timezone-aware/naive datetime mixing in predictions
  * Model weight calculation with zero or negative scores
  * Feature importance with untrained base models
  * Incremental updates without prior training
  * Validation data with inconsistent column structures
- Error Handling:
  * ModelTrainingError for insufficient data, base model failures, validation errors
  * ModelPredictionError for untrained models, invalid features, all base model failures
  * ValueError for invalid parameters, data format issues, dimension mismatches
  * Exception propagation from base model training/prediction failures
  * Graceful degradation when subset of base models fail
  * Error recovery in incremental update scenarios
- Coverage Target: 85%+

**Test Data Requirements for Ensemble:**
- Mock base predictors (LSTM, XGBoost, HMM, GP) with configurable training/prediction behavior
- Feature DataFrames with 50+ samples and various column counts for training
- Target DataFrames with time_until_transition_seconds, transition_type, target_time columns
- Cross-validation datasets with different fold configurations
- Meta-feature matrices with aligned/misaligned dimensions
- Prediction results from base models with various confidence scores and metadata
- Tracking manager mock for accuracy integration testing
- Model persistence test files and directories
- Timezone-aware and naive datetime objects for time utility testing
- Validation datasets with missing columns, NaN values, wrong data types
- Base model training results with success/failure scenarios
- Large feature sets for dimension reduction and scaling testing

**Special Testing Considerations:**
- Ensemble requires complex async coordination between multiple base models
- Cross-validation testing needs careful fold management and data consistency
- Meta-learner training involves feature scaling and dimension alignment challenges
- Prediction combination requires careful handling of variable-length result arrays
- Model weight calculations need mathematical validation against known performance metrics
- Confidence calculation incorporates GP uncertainty quantification requiring specialized test data
- Incremental update testing requires partially trained models and realistic update scenarios
- Model persistence testing must handle complex nested object serialization
- Threading and async behavior in concurrent base model training needs careful mock management
- Timezone handling utilities need comprehensive datetime edge case testing

### src/models/training_integration.py - Training Integration Logic
**Classes Found:** TrainingIntegrationManager
**Methods Analyzed:** __init__, initialize, shutdown, _start_background_tasks, _register_tracking_callbacks, _on_accuracy_degradation, _on_drift_detected, _on_performance_change, _queue_retraining_request, _calculate_priority, _can_retrain_room, _get_cooldown_remaining, _training_queue_processor, _process_training_queue, _execute_training_request, _select_training_profile_for_strategy, _handle_training_completion, _update_model_registration, _handle_training_failure, _periodic_maintenance, _resource_monitor, _cleanup_old_data, _check_scheduled_training, _update_performance_baselines, get_active_training_rooms, _check_resource_usage, _adjust_training_capacity, request_manual_training, get_integration_status, get_training_queue_status, set_training_capacity, set_cooldown_period, integrate_training_with_tracking_manager

**Required Tests:**
- Unit Tests: 
  - **TrainingIntegrationManager.__init__() Tests:**
    - Test initialization with all required parameters (tracking_manager, training_pipeline, config_manager)
    - Test initialization with optional config_manager (defaults to get_training_config_manager())
    - Test all instance variable initialization (_active_training_requests, _training_queue, _integration_active)
    - Test trigger and condition dictionaries initialization (_accuracy_triggers, _drift_triggers, _last_training_times)
    - Test resource management parameters (_max_concurrent_training=2, _training_cooldown_hours=12)
    - Test background task and shutdown event initialization
    - Test logging initialization message

  - **initialize() Method Tests:**
    - Test successful initialization flow setting _integration_active=True
    - Test background task startup via _start_background_tasks()
    - Test callback registration via _register_tracking_callbacks()
    - Test exception handling and error logging during initialization
    - Test initialization failure cleanup and error propagation

  - **shutdown() Method Tests:**
    - Test graceful shutdown setting _integration_active=False and _shutdown_event
    - Test background task cleanup with asyncio.gather and return_exceptions=True
    - Test _background_tasks.clear() cleanup
    - Test exception handling during shutdown with error logging
    - Test logging of successful shutdown completion

  - **Background Task Management Tests:**
    - Test _start_background_tasks() creates three async tasks (queue processor, maintenance, resource monitor)
    - Test task creation with asyncio.create_task() and proper task tracking
    - Test exception handling during background task startup
    - Test logging of successful background task startup

  - **Callback Registration Tests:**
    - Test _register_tracking_callbacks() with hasattr() checks for tracking manager methods
    - Test accuracy callback registration (add_accuracy_callback with _on_accuracy_degradation)
    - Test drift callback registration (add_drift_callback with _on_drift_detected)  
    - Test performance callback registration (add_performance_callback with _on_performance_change)
    - Test exception handling with warning logging for failed registrations

  - **Accuracy Degradation Handler Tests:**
    - Test _on_accuracy_degradation() with accuracy metrics processing
    - Test accuracy threshold evaluation (accuracy_rate vs min_accuracy_threshold * 100)
    - Test error threshold evaluation (mean_error_minutes vs max_error_threshold_minutes)
    - Test should_retrain logic with both accuracy and error conditions
    - Test retraining request queueing with proper trigger reason and priority calculation
    - Test logging of accuracy degradation events and threshold comparisons
    - Test exception handling in accuracy degradation processing

  - **Drift Detection Handler Tests:**
    - Test _on_drift_detected() with drift metrics processing (severity, score, recommendation)
    - Test retraining strategy selection based on drift severity (CRITICAL/MAJOR -> full_retrain, others -> adaptive)
    - Test priority assignment based on drift severity (CRITICAL/MAJOR -> 1, others -> 3)
    - Test retraining request queueing with drift-specific metadata
    - Test handling when retraining_recommended=False
    - Test exception handling in drift detection processing

  - **Performance Change Handler Tests:**
    - Test _on_performance_change() basic logging functionality
    - Test performance metrics processing and logging
    - Test exception handling in performance change processing

  - **Retraining Request Queue Management Tests:**
    - Test _queue_retraining_request() with all parameters (room_id, trigger_reason, priority, strategy, metadata)
    - Test duplicate request prevention (room already in _active_training_requests)
    - Test cooldown period checking with _can_retrain_room()
    - Test request creation with proper timestamp and metadata structure
    - Test queue priority ordering (priority, then requested_at)
    - Test logging of queued requests and cooldown scenarios
    - Test exception handling during request queueing

  - **Priority Calculation Tests:**
    - Test _calculate_priority() with various current_value/threshold ratios
    - Test critical priority (ratio < 0.5 -> priority 1)
    - Test high priority (ratio < 0.7 -> priority 2) 
    - Test medium priority (ratio < 0.9 -> priority 3)
    - Test low priority (ratio >= 0.9 -> priority 4)
    - Test edge case with current_value <= 0 (returns priority 1)

  - **Cooldown Management Tests:**
    - Test _can_retrain_room() with no previous training (returns True)
    - Test _can_retrain_room() with recent training within cooldown (returns False)
    - Test _can_retrain_room() with training outside cooldown period (returns True)
    - Test _get_cooldown_remaining() calculations with elapsed time
    - Test cooldown calculations using _training_cooldown_hours and timedelta

  - **Training Queue Processing Tests:**
    - Test _training_queue_processor() background task loop with shutdown event
    - Test _process_training_queue() with empty queue (early return)
    - Test capacity checking against _max_concurrent_training
    - Test request processing in priority order with duplicate prevention
    - Test cooldown checking during request processing
    - Test request execution via _execute_training_request()
    - Test processed request cleanup from queue
    - Test exception handling in queue processing with error logging and retry delays

  - **Training Request Execution Tests:**
    - Test _execute_training_request() with all request metadata (room_id, trigger_reason, strategy)
    - Test training type mapping (full_retrain -> FULL_RETRAIN, incremental -> INCREMENTAL, adaptive -> ADAPTATION)
    - Test training profile selection via _select_training_profile_for_strategy()
    - Test config manager profile setting with set_current_profile()
    - Test pipeline task creation with asyncio.create_task() and proper parameters
    - Test active training tracking with generated pipeline_id
    - Test training completion handling via _handle_training_completion()
    - Test exception handling with active training cleanup and error propagation

  - **Training Profile Selection Tests:**
    - Test _select_training_profile_for_strategy() mappings (full_retrain -> COMPREHENSIVE, incremental -> QUICK, adaptive -> PRODUCTION)
    - Test default strategy fallback to PRODUCTION profile

  - **Training Completion Handling Tests:**
    - Test _handle_training_completion() active training cleanup
    - Test last training time updating with datetime.utcnow()
    - Test success detection with progress.stage.value == "completed"
    - Test tracking manager notification via on_model_retrained()
    - Test model registration updates via _update_model_registration()
    - Test training failure handling via _handle_training_failure()
    - Test exception handling during completion processing

  - **Model Registration Tests:**
    - Test _update_model_registration() with model registry access
    - Test model key generation (room_id + best_model)
    - Test tracking manager model registration with proper parameters
    - Test logging of successful model registration updates
    - Test exception handling during model registration

  - **Training Failure Handling Tests:**
    - Test _handle_training_failure() retry logic with failure_count increments
    - Test retry request modification (strategy -> "quick", priority increase)
    - Test maximum retry limit (3 attempts) with permanent failure handling
    - Test tracking manager failure notification via on_training_failure()
    - Test exception handling during failure processing

  - **Maintenance Tasks Tests:**
    - Test _periodic_maintenance() background task loop with 3600-second intervals
    - Test maintenance task execution (_cleanup_old_data, _check_scheduled_training, _update_performance_baselines)
    - Test exception handling with 300-second retry delays
    - Test asyncio cancellation handling

  - **Resource Monitoring Tests:**
    - Test _resource_monitor() background task loop with 300-second intervals
    - Test resource usage checking via _check_resource_usage()
    - Test capacity adjustment via _adjust_training_capacity()
    - Test exception handling with 60-second retry delays
    - Test asyncio cancellation handling

  - **Cleanup and Maintenance Tests:**
    - Test _cleanup_old_data() with 24-hour cutoff filtering
    - Test old request removal from _training_queue with timestamp comparison
    - Test cleanup count logging
    - Test _check_scheduled_training() placeholder implementation
    - Test _update_performance_baselines() placeholder implementation

  - **Resource Management Tests:**
    - Test get_active_training_rooms() returns set of active room IDs
    - Test _check_resource_usage() with active room logging
    - Test _adjust_training_capacity() placeholder implementation

  - **Public API Tests:**
    - Test request_manual_training() with all parameters and successful queueing
    - Test get_integration_status() comprehensive status dictionary with all fields
    - Test get_training_queue_status() with request metadata and waiting time calculations
    - Test set_training_capacity() validation and parameter updates
    - Test set_cooldown_period() validation and parameter updates
    - Test API methods exception handling and return value validation

  - **Global Integration Function Tests:**
    - Test integrate_training_with_tracking_manager() function with all parameters
    - Test TrainingIntegrationManager creation and initialization
    - Test successful integration logging
    - Test exception handling with ModelTrainingError raising

- Integration Tests:
  - **Tracking Manager Integration:**
    - Test integration with real tracking manager callback registration
    - Test accuracy degradation callbacks with realistic metrics
    - Test drift detection callbacks with various severity levels
    - Test performance change callbacks with model metrics
    - Test model registration and failure notifications

  - **Training Pipeline Integration:**
    - Test integration with ModelTrainingPipeline for actual training execution
    - Test training request parameter passing and result handling
    - Test training type coordination between integration manager and pipeline
    - Test training completion and failure scenarios with real pipeline

  - **Configuration Manager Integration:**
    - Test integration with training configuration manager
    - Test profile selection and configuration updates
    - Test environment configuration access for thresholds
    - Test configuration manager initialization and error handling

  - **Background Task Integration:**
    - Test complete background task lifecycle (startup, running, shutdown)
    - Test task coordination and resource sharing
    - Test asyncio event loop integration and proper cleanup
    - Test task exception isolation and system resilience

- Edge Cases:
  - **Initialization Edge Cases:**
    - Test initialization with None tracking_manager or training_pipeline
    - Test initialization with invalid config_manager
    - Test initialization with missing callback methods in tracking manager
    - Test initialization failure recovery and state cleanup

  - **Queue Management Edge Cases:**
    - Test queue behavior with maximum concurrent training reached
    - Test queue ordering with identical priorities and timestamps
    - Test queue processing with all requests in cooldown
    - Test queue cleanup with very large numbers of old requests
    - Test memory usage with thousands of queued requests

  - **Training Execution Edge Cases:**
    - Test training request execution with pipeline failures
    - Test training execution with invalid room configurations
    - Test training execution timeout scenarios
    - Test concurrent training limit boundary conditions
    - Test pipeline task cancellation during shutdown

  - **Callback Processing Edge Cases:**
    - Test callback execution with malformed metric dictionaries
    - Test callback execution with missing required metric fields
    - Test callback execution with extreme threshold values
    - Test rapid callback invocation and queue saturation
    - Test callback execution during system shutdown

  - **Resource Management Edge Cases:**
    - Test resource monitoring with system resource exhaustion
    - Test capacity adjustment with negative or zero values
    - Test cooldown period edge cases (exactly at boundary, leap years)
    - Test active training tracking with ID collisions
    - Test background task resource cleanup during exceptions

- Error Handling:
  - **Initialization Errors:**
    - Test ModelTrainingError propagation from integration function
    - Test exception handling during background task startup
    - Test callback registration failures with graceful degradation
    - Test configuration manager initialization failures

  - **Training Execution Errors:**
    - Test training pipeline execution failures and error recovery
    - Test model registration failures during completion handling
    - Test asyncio task creation and execution failures
    - Test training timeout and resource exhaustion scenarios

  - **Queue Processing Errors:**
    - Test queue processing with corrupted request data
    - Test priority calculation errors with invalid inputs
    - Test request execution failures and queue cleanup
    - Test concurrent access errors and data consistency

  - **Background Task Errors:**
    - Test maintenance task failures and system resilience
    - Test resource monitoring failures and fallback behavior
    - Test asyncio cancellation handling and proper cleanup
    - Test task exception isolation and error logging

  - **API Operation Errors:**
    - Test public API method failures with proper error responses
    - Test parameter validation errors and user feedback
    - Test concurrent API access and thread safety
    - Test integration status collection failures

- Coverage Target: 85%+

**Mock Requirements:**
- Mock tracking_manager with callback registration methods (add_accuracy_callback, add_drift_callback, add_performance_callback)
- Mock training_pipeline with run_retraining_pipeline method and model registry
- Mock config_manager with profile management and environment configuration
- Mock asyncio operations (create_task, gather, wait_for, Event, sleep)
- Mock datetime.utcnow() for deterministic timestamp testing
- Mock logging.getLogger() for error logging validation
- Mock TrainingProfile and TrainingType enums for strategy testing
- Mock get_training_config_manager() for default configuration

**Test Fixtures Needed:**
- Sample accuracy metrics dictionaries with various threshold scenarios
- Sample drift metrics with different severity levels and recommendations
- Sample performance metrics for callback testing
- Training request objects with various priorities and metadata
- Mock training pipeline results (success and failure scenarios)
- Mock tracking manager objects with controllable callback behavior
- Configuration manager fixtures with different profile configurations
- Background task scenarios for asyncio testing

**Special Testing Considerations:**
- Training integration requires comprehensive async operation testing with proper event loop management
- Background task testing needs careful coordination of asyncio tasks and shutdown events
- Callback system testing requires mock objects with realistic tracking manager behavior
- Queue processing requires testing of priority ordering and concurrent access scenarios
- Error handling must cover all async execution paths and ensure proper cleanup
- Resource management testing needs validation of concurrent training limits and cooldown periods
- Integration testing requires coordination between multiple complex components (tracking, training, configuration)
- Performance testing with realistic training request volumes and background task load
- Memory usage monitoring for long-running background tasks and queue management
- Thread safety validation for concurrent API access and background task operations

### src/models/training_config.py - Training Configuration
**Classes Found:** TrainingProfile (Enum), OptimizationLevel (Enum), ResourceLimits (dataclass), QualityThresholds (dataclass), OptimizationConfig (dataclass), TrainingEnvironmentConfig (dataclass), TrainingConfigManager (main class)
**Methods Analyzed:** TrainingProfile._missing_(), TrainingProfile.from_string(), ResourceLimits.validate(), QualityThresholds.validate(), TrainingConfigManager.__init__(), _initialize_default_profiles(), _load_config_file(), _dict_to_environment_config(), get_training_config(), _get_lookback_days_for_profile(), set_current_profile(), get_current_profile(), get_environment_config(), validate_configuration(), get_optimization_config(), update_profile_config(), save_config_to_file(), get_profile_comparison(), recommend_profile_for_use_case(), get_training_config_manager(), get_training_config()

**Required Tests:**
- Unit Tests:
  - Test TrainingProfile enum values (DEVELOPMENT, PRODUCTION, TESTING, RESEARCH, QUICK, COMPREHENSIVE)
  - Test TrainingProfile._missing_() with frame inspection logic for different caller contexts
  - Test TrainingProfile.from_string() success and custom error messages
  - Test OptimizationLevel enum values (NONE, BASIC, STANDARD, INTENSIVE)
  - Test ResourceLimits.validate() with positive, negative, zero, and None values for all fields
  - Test ResourceLimits validation error messages for each field type
  - Test QualityThresholds.validate() with boundary conditions (0.0, 1.0) for percentage fields
  - Test QualityThresholds validation for positive number requirements
  - Test OptimizationConfig default factory functions for search spaces
  - Test OptimizationConfig with all parameter combinations
  - Test TrainingEnvironmentConfig.validate() calling sub-validators
  - Test TrainingEnvironmentConfig.validate() Path conversion and error handling
  - Test TrainingConfigManager.__init__() with default and custom config paths
  - Test TrainingConfigManager.__init__() config file existence checking
  - Test _initialize_default_profiles() creating all six profile configurations
  - Test _initialize_default_profiles() profile-specific parameters (resource limits, quality thresholds)
  - Test _load_config_file() YAML parsing and profile creation
  - Test _load_config_file() default profile setting from config
  - Test _load_config_file() error handling for malformed YAML
  - Test _dict_to_environment_config() nested dataclass conversion
  - Test _dict_to_environment_config() enum conversion and Path handling
  - Test get_training_config() with profile parameter and None (current profile)
  - Test get_training_config() profile not found fallback to production
  - Test get_training_config() TrainingConfig field mapping from environment config
  - Test _get_lookback_days_for_profile() for all profile types
  - Test _get_lookback_days_for_profile() default value for unknown profiles
  - Test set_current_profile() validation and success cases
  - Test set_current_profile() error handling for unavailable profiles
  - Test get_current_profile() returning current profile
  - Test get_environment_config() with profile parameter and current profile default
  - Test get_environment_config() error handling for missing profiles
  - Test validate_configuration() calling env_config.validate()
  - Test validate_configuration() handling missing profiles
  - Test get_optimization_config() delegation to environment config
  - Test update_profile_config() with valid and invalid configuration keys
  - Test update_profile_config() error handling for missing profiles
  - Test save_config_to_file() YAML serialization with enum and Path conversion
  - Test save_config_to_file() directory creation and file writing
  - Test save_config_to_file() error handling and logging
  - Test get_profile_comparison() metrics extraction for all profiles
  - Test get_profile_comparison() comparison data structure creation
  - Test recommend_profile_for_use_case() with various use case strings
  - Test recommend_profile_for_use_case() case-insensitive matching and default
  - Test get_training_config_manager() singleton pattern
  - Test get_training_config() convenience function delegation

- Integration Tests:
  - Test TrainingConfigManager integration with SystemConfig via get_config()
  - Test config file loading with real YAML files containing all profile types
  - Test _dict_to_environment_config() with complex nested configuration structures
  - Test get_training_config() creating valid TrainingConfig objects for all profiles
  - Test profile switching integration (set_current_profile → get_training_config)
  - Test save_config_to_file() → _load_config_file() round-trip preservation
  - Test update_profile_config() → validate_configuration() integration
  - Test singleton pattern integration with multiple get_training_config_manager() calls
  - Test recommend_profile_for_use_case() → set_current_profile() workflow
  - Test config validation integration across all nested dataclasses

- Edge Cases:
  - Test TrainingProfile._missing_() frame inspection with complex call stacks
  - Test TrainingProfile._missing_() with AttributeError during frame traversal
  - Test ResourceLimits.validate() with extremely large positive values
  - Test QualityThresholds.validate() with floating-point precision edge cases
  - Test TrainingConfigManager.__init__() with non-existent config path
  - Test _load_config_file() with empty YAML file
  - Test _load_config_file() with YAML containing only partial profile data
  - Test _dict_to_environment_config() with missing required nested fields
  - Test _dict_to_environment_config() with invalid enum string values
  - Test get_training_config() with profile conversion edge cases
  - Test save_config_to_file() with unserializable configuration objects
  - Test save_config_to_file() with filesystem permission errors
  - Test update_profile_config() with configuration objects requiring deep copying
  - Test get_profile_comparison() with profiles having None values
  - Test recommend_profile_for_use_case() with empty strings and special characters

- Error Handling:
  - Test TrainingProfile._missing_() ValueError raising with context-specific messages
  - Test TrainingProfile.from_string() ValueError handling with custom message
  - Test ResourceLimits.validate() comprehensive issue collection
  - Test QualityThresholds.validate() all validation rule combinations
  - Test TrainingEnvironmentConfig.validate() error aggregation from sub-validators
  - Test TrainingConfigManager._load_config_file() exception handling with logging
  - Test _dict_to_environment_config() exception handling during dataclass creation
  - Test set_current_profile() ValueError raising for unavailable profiles
  - Test get_environment_config() ValueError raising for missing profiles
  - Test update_profile_config() ValueError raising for missing profiles
  - Test update_profile_config() warning logging for unknown configuration keys
  - Test save_config_to_file() comprehensive exception handling and re-raising
  - Test validation errors propagation through get_training_config()
  - Test YAML parsing errors in _load_config_file() with proper logging
  - Test filesystem errors during config file operations

- Coverage Target: 85%+

### src/models/base/gp_predictor.py - Gaussian Process Models
**Classes Found:** GaussianProcessPredictor
**Methods Analyzed:** __init__, _create_kernel, train, predict, get_feature_importance, _select_inducing_points, _calibrate_uncertainty, _calculate_confidence_intervals, _calculate_confidence_score, _generate_alternative_scenarios, _estimate_epistemic_uncertainty, _determine_transition_type, _prepare_targets, get_uncertainty_metrics, incremental_update, save_model, load_model, get_model_complexity

**Required Tests:**

- **Unit Tests:**
  - **Initialization Tests:**
    - Test GaussianProcessPredictor initialization with default parameters
    - Test initialization with custom kernel types ('rbf', 'matern', 'periodic', 'rational_quadratic', 'composite')
    - Test initialization with different confidence intervals and uncertainty parameters
    - Test initialization with sparse GP parameters and max inducing points
  
  - **Kernel Creation Tests:**
    - Test _create_kernel() with different kernel types (rbf, matern, periodic, rational_quadratic, composite)
    - Test fallback behavior when PeriodicKernel is not available
    - Test composite kernel creation with multiple components (local, trend, daily, weekly, noise)
    - Test kernel parameter bounds and initialization values
  
  - **Training Tests:**
    - Test successful training with valid features and targets DataFrames
    - Test training with validation data provided
    - Test training with insufficient data (< 10 samples) raises ModelTrainingError
    - Test sparse GP activation when data exceeds max_inducing_points threshold
    - Test feature scaling with StandardScaler fit_transform
    - Test kernel parameter optimization and log marginal likelihood calculation
    - Test training history recording and TrainingResult creation
  
  - **Prediction Tests:**
    - Test predict() with trained model returns PredictionResult list
    - Test prediction with uncertainty quantification (mean, std)
    - Test confidence interval calculation for different confidence levels (68%, 95%, 99%)
    - Test prediction without trained model raises ModelPredictionError
    - Test prediction with invalid features raises ModelPredictionError
    - Test alternative scenario generation based on uncertainty
    - Test transition type determination based on current state and time
  
  - **Feature Importance Tests:**
    - Test get_feature_importance() with ARD kernel (individual length scales)
    - Test feature importance with single length scale kernel
    - Test fallback uniform importance when kernel parameters unavailable
    - Test handling of missing or invalid kernel parameters
  
  - **Uncertainty Quantification Tests:**
    - Test _calibrate_uncertainty() with validation data
    - Test uncertainty calibration curve calculation and storage
    - Test confidence score calculation based on prediction uncertainty
    - Test epistemic uncertainty estimation using training point distances
    - Test get_uncertainty_metrics() returning comprehensive uncertainty information
  
  - **Sparse GP Tests:**
    - Test _select_inducing_points() with KMeans clustering
    - Test inducing point selection with different dataset sizes
    - Test duplicate removal and sorting in inducing point selection
    - Test sparse GP training vs full GP training paths
  
  - **Incremental Update Tests:**
    - Test incremental_update() with new training data
    - Test incremental update without trained model falls back to full training
    - Test inducing points update in sparse GP mode
    - Test combined data size limiting to prevent memory issues
    - Test kernel parameter reuse with warm start
  
  - **Persistence Tests:**
    - Test save_model() serializes all model components correctly
    - Test load_model() restores complete model state
    - Test model saving/loading with different file paths and formats
    - Test handling of missing or corrupted model files
  
  - **Target Preparation Tests:**
    - Test _prepare_targets() with different DataFrame column formats
    - Test target value clipping to reasonable bounds (60-86400 seconds)
    - Test handling of time_until_transition_seconds columns
    - Test datetime calculation from target_time and next_transition_time columns

- **Integration Tests:**
  - **GP Model Training Integration:**
    - Test full training pipeline with realistic occupancy data
    - Test integration with different kernel types and their effect on predictions
    - Test validation data processing and accuracy metric calculation
    - Test sparse GP vs full GP performance comparison with large datasets
  
  - **Scikit-learn Integration:**
    - Test compatibility with different scikit-learn versions
    - Test PeriodicKernel availability detection and fallback mechanisms
    - Test GaussianProcessRegressor parameter passing and optimization
    - Test StandardScaler integration with feature preprocessing
  
  - **BasePredictor Integration:**
    - Test inheritance from BasePredictor and interface compliance
    - Test validate_features() integration and error handling
    - Test _record_prediction() integration with prediction tracking
    - Test model_version generation and training history management
  
  - **Configuration Integration:**
    - Test model parameter loading from DEFAULT_MODEL_PARAMS
    - Test room_id assignment and usage throughout prediction pipeline
    - Test ModelType.GP enum integration and error messaging

- **Edge Cases:**
  - **Data Edge Cases:**
    - Test training with exactly 10 samples (boundary condition)
    - Test prediction with single-row DataFrame input
    - Test handling of NaN or infinite values in features/targets
    - Test empty DataFrame inputs and appropriate error handling
    - Test features with zero variance or constant values
  
  - **Kernel Edge Cases:**
    - Test kernel creation with very small or very large feature dimensions
    - Test handling of kernel parameter optimization failures
    - Test kernel bounds validation and constraint enforcement
    - Test composite kernel with disabled periodic components
  
  - **Uncertainty Edge Cases:**
    - Test confidence intervals with extremely low/high uncertainty values
    - Test calibration with insufficient validation data
    - Test uncertainty calculations with edge case statistical distributions
    - Test handling of negative or zero standard deviations
  
  - **Sparse GP Edge Cases:**
    - Test inducing point selection with very small datasets
    - Test KMeans clustering failure scenarios
    - Test inducing points exceeding maximum limits
    - Test sparse GP with single inducing point
  
  - **Time/Date Edge Cases:**
    - Test transition time calculations at day boundaries
    - Test timezone-aware datetime handling in predictions
    - Test prediction intervals crossing midnight or day boundaries
    - Test handling of daylight saving time transitions

- **Error Handling:**
  - **Training Errors:**
    - Test ModelTrainingError with insufficient data
    - Test training failures due to kernel optimization issues
    - Test memory errors with extremely large datasets
    - Test feature scaling failures with invalid data
    - Test validation data shape mismatches
  
  - **Prediction Errors:**
    - Test ModelPredictionError when model not trained
    - Test prediction failures with mismatched feature dimensions
    - Test handling of scikit-learn internal errors during prediction
    - Test feature scaling errors during prediction preprocessing
  
  - **File I/O Errors:**
    - Test model saving failures due to permissions or disk space
    - Test model loading failures with corrupted files
    - Test pickle serialization errors with complex kernel objects
    - Test file path validation and error messaging
  
  - **Numerical Errors:**
    - Test handling of numerical instability in GP calculations
    - Test matrix inversion failures in GP optimization
    - Test overflow/underflow in uncertainty calculations
    - Test division by zero in confidence score calculations
  
  - **Configuration Errors:**
    - Test invalid kernel type specifications
    - Test invalid confidence interval values
    - Test negative or invalid hyperparameter values
    - Test inconsistent sparse GP configuration parameters

- **Coverage Target:** 85%+

**Special GP Testing Considerations:**
- Gaussian Process models require careful numerical validation due to matrix operations
- Kernel parameter optimization can be stochastic, requiring statistical test validation
- Uncertainty quantification needs validation against known statistical properties
- Sparse GP approximation introduces additional complexity requiring accuracy validation
- Incremental learning requires careful state management testing
- Scikit-learn version compatibility requires fallback mechanism testing
- Memory usage critical for large-scale GP operations requiring performance testing

### src/models/base/hmm_predictor.py - Hidden Markov Models
**Classes Found:** HMMPredictor (extends BasePredictor)
**Methods Analyzed:** __init__, train, predict, get_feature_importance, _prepare_targets, _analyze_states, _assign_state_label, _build_transition_matrix, _train_state_duration_models, _predict_durations, _predict_single_duration, _determine_transition_type_from_states, _calculate_confidence, get_state_info, save_model, load_model, incremental_update, get_model_complexity, _predict_hmm_internal

**Required Tests:**
- Unit Tests:
  - Test HMMPredictor.__init__ with default parameters and parameter aliases (n_states/n_components)
  - Test HMMPredictor.__init__ with n_iter/max_iter synchronization logic
  - Test HMMPredictor.__init__ with model_params initialization and default values
  - Test HMMPredictor.__init__ with GaussianMixture component initialization
  - Test train method with valid features and targets DataFrames
  - Test train method with insufficient training data (< 20 samples) raising ModelTrainingError
  - Test train method with feature scaling using StandardScaler.fit_transform
  - Test train method with KMeans pre-clustering for GMM initialization
  - Test train method with GaussianMixture fitting and state prediction
  - Test train method with state analysis, transition matrix building, and duration model training
  - Test train method with training metrics calculation (R², MAE, RMSE)
  - Test train method with validation data processing and metrics
  - Test train method with TrainingResult creation and training_history update
  - Test predict method with untrained model raising ModelPredictionError
  - Test predict method with invalid features raising ModelPredictionError
  - Test predict method with feature scaling and state probability prediction
  - Test predict method with duration prediction and transition time calculation
  - Test predict method with confidence calculation and PredictionResult creation
  - Test predict method with prediction recording for accuracy tracking
  - Test get_feature_importance with state discrimination power calculation
  - Test get_feature_importance with covariance matrix analysis (full, diag, tied, spherical)
  - Test get_feature_importance with importance score normalization
  - Test get_feature_importance with untrained model returning empty dict
  - Test _prepare_targets with time_until_transition_seconds column
  - Test _prepare_targets with next_transition_time and target_time columns
  - Test _prepare_targets with fallback to first column and clipping (60-86400)
  - Test _prepare_targets with exception handling returning default values
  - Test _analyze_states with state characteristics calculation and duration analysis
  - Test _analyze_states with state probability analysis and confidence metrics
  - Test _analyze_states with state labeling and reliability assessment
  - Test _assign_state_label with duration-based heuristic labeling
  - Test _build_transition_matrix with state transition counting and probability calculation
  - Test _build_transition_matrix with uniform distribution for unobserved transitions
  - Test _train_state_duration_models with LinearRegression for each state
  - Test _train_state_duration_models with insufficient samples falling back to average
  - Test _predict_durations with state identification and duration prediction
  - Test _predict_single_duration with average and regression model types
  - Test _predict_single_duration with default prediction for missing states
  - Test _determine_transition_type_from_states with current occupancy inference
  - Test _determine_transition_type_from_states with state characteristics analysis
  - Test _calculate_confidence with state confidence and entropy adjustment
  - Test _calculate_confidence with prediction reasonableness adjustment
  - Test get_state_info returning n_states, labels, characteristics, transition_matrix
  - Test save_model with pickle serialization of all model components
  - Test save_model with file writing and success/failure handling
  - Test load_model with pickle deserialization and component restoration
  - Test load_model with training history reconstruction from dictionaries
  - Test load_model with exception handling and logging
  - Test incremental_update with existing trained model
  - Test incremental_update with untrained model falling back to full training
  - Test incremental_update with insufficient data raising ModelTrainingError
  - Test incremental_update with new GMM training and state analysis
  - Test incremental_update with transition model updates and performance calculation
  - Test get_model_complexity with trained model returning component information
  - Test get_model_complexity with untrained model returning default values
  - Test _predict_hmm_internal with state predictions and transition model usage

- Integration Tests:
  - Test integration with BasePredictor, PredictionResult, and TrainingResult classes
  - Test integration with ModelType.HMM and DEFAULT_MODEL_PARAMS
  - Test integration with StandardScaler for feature normalization
  - Test integration with GaussianMixture for hidden state modeling
  - Test integration with KMeans for initial clustering
  - Test integration with LinearRegression for state duration modeling
  - Test integration with ModelTrainingError and ModelPredictionError exceptions
  - Test integration with pandas DataFrame operations and numpy array processing
  - Test integration with scikit-learn metrics (r2_score, mean_absolute_error, mean_squared_error)
  - Test integration with pickle for model serialization/deserialization
  - Test integration with logging for training and prediction monitoring
  - Test end-to-end workflow from training to prediction with realistic sensor data

- Edge Cases:
  - Test with n_components = 1 (single state HMM)
  - Test with n_components > number of samples (overfitting scenario)
  - Test with all samples having identical features (zero variance)
  - Test with features containing NaN or infinite values
  - Test with extremely short or long prediction durations
  - Test with empty state_durations dictionary
  - Test with transition_matrix as None (no transitions observed)
  - Test with covariance_type variations and singular matrices
  - Test with state_labels assignment edge cases
  - Test with model convergence failures (n_iter reached)
  - Test with feature_scaler not fitted properly
  - Test with corrupted pickle files during load_model
  - Test with mismatched feature names between training and prediction
  - Test with extremely large or small confidence scores
  - Test with state_model predict_proba returning edge probabilities
  - Test with prediction_time and datetime timezone handling
  - Test with feature importance calculation edge cases (zero covariance)
  - Test with transition model training with < 5 samples per state
  - Test with incremental update on completely different data distribution

- Error Handling:
  - Test ModelTrainingError raising with proper model_type, room_id, and cause
  - Test ModelPredictionError raising for untrained model scenarios
  - Test exception handling in train method with proper TrainingResult error recording
  - Test exception handling in predict method with detailed error logging
  - Test exception handling in get_feature_importance with warning logs
  - Test exception handling in _prepare_targets with default value fallback
  - Test exception handling in save_model/load_model with file I/O errors
  - Test exception handling in incremental_update with ModelTrainingError
  - Test GaussianMixture fitting failures and recovery
  - Test KMeans clustering failures with random_state handling
  - Test LinearRegression training failures for individual states
  - Test feature scaling errors with StandardScaler
  - Test numpy array operations with invalid shapes or types
  - Test transition matrix calculation with division by zero
  - Test confidence calculation with logarithm domain errors
  - Test pickle serialization/deserialization errors

- Coverage Target: 85%+

### src/adaptation/retrainer.py - Adaptive Retraining
**Classes Found:** RetrainingTrigger (Enum), RetrainingStrategy (Enum), RetrainingStatus (Enum), RetrainingRequest (dataclass), RetrainingProgress (dataclass), RetrainingHistory (dataclass), AdaptiveRetrainer, RetrainingError (Exception)
**Methods Analyzed:** AdaptiveRetrainer.__init__, initialize, shutdown, evaluate_retraining_need, request_retraining, get_retraining_status, get_retraining_progress, cancel_retraining, get_retrainer_stats, RetrainingRequest.to_dict, RetrainingProgress.update_progress, RetrainingHistory.add_retraining_record, _analyze_accuracy_trend, get_success_rate, get_recent_performance, to_dict, plus 50+ private methods

**Required Tests:**
- Unit Tests:
  - Test RetrainingTrigger enum values (ACCURACY_DEGRADATION, ERROR_THRESHOLD_EXCEEDED, CONCEPT_DRIFT, SCHEDULED_UPDATE, MANUAL_REQUEST, PERFORMANCE_ANOMALY)
  - Test RetrainingStrategy enum values (INCREMENTAL, FULL_RETRAIN, FEATURE_REFRESH, ENSEMBLE_REBALANCE)
  - Test RetrainingStatus enum values (PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED)
  - Test RetrainingRequest dataclass initialization with all fields including complex defaults
  - Test RetrainingRequest.__post_init__ with legacy field mapping from retraining_parameters
  - Test RetrainingRequest.__lt__ priority queue comparison (higher priority first)
  - Test RetrainingRequest.to_dict with comprehensive serialization including nested objects
  - Test RetrainingProgress initialization and update_progress method with estimated completion
  - Test RetrainingHistory initialization with defaultdict and datetime fields
  - Test RetrainingHistory.add_retraining_record with completed/failed status handling
  - Test RetrainingHistory._analyze_accuracy_trend with linear regression analysis
  - Test RetrainingHistory.get_success_rate calculation and percentage conversion
  - Test RetrainingHistory.get_recent_performance with time-based filtering
  - Test AdaptiveRetrainer.__init__ with all optional dependencies and configuration
  - Test AdaptiveRetrainer.initialize starting background tasks and checking enabled status
  - Test AdaptiveRetrainer.shutdown gracefully stopping tasks and clearing resources
  - Test evaluate_retraining_need with accuracy threshold trigger detection
  - Test evaluate_retraining_need with error threshold trigger detection
  - Test evaluate_retraining_need with concept drift trigger detection
  - Test evaluate_retraining_need with performance anomaly trigger detection
  - Test evaluate_retraining_need with cooldown period checking
  - Test evaluate_retraining_need with strategy selection logic
  - Test request_retraining with manual trigger and parameter building
  - Test request_retraining with enum/string model type handling
  - Test get_retraining_status for specific request_id with progress information
  - Test get_retraining_status for all requests with category classification
  - Test cancel_retraining for pending requests with status updates
  - Test cancel_retraining for active requests with cleanup
  - Test get_retrainer_stats with comprehensive statistics calculation
  - Test _queue_retraining_request with priority queue management and deduplication
  - Test _select_retraining_strategy with accuracy/drift based selection
  - Test _is_in_cooldown with datetime comparison and timezone handling
  - Test _retraining_processor_loop background processing with shutdown handling
  - Test _trigger_checker_loop with configurable check intervals
  - Test _start_retraining with resource tracking and progress initialization
  - Test _perform_retraining with full pipeline execution and phase tracking
  - Test _prepare_retraining_data with train_test_split and temporal considerations
  - Test _extract_features_for_retraining with feature engine integration
  - Test _retrain_model with strategy-specific training and optimization
  - Test _validate_and_deploy_retrained_model with validation thresholds
  - Test _handle_retraining_success with cooldown updates and statistics
  - Test _handle_retraining_failure with error tracking and cleanup
  - Test _notify_retraining_event with callback execution
  - Test _classify_drift_severity with fallback classification
  - Test _validate_retraining_predictions with PredictionValidator integration

- Integration Tests:
  - Test integration with TrackingConfig for retraining settings
  - Test integration with ModelOptimizer for hyperparameter optimization
  - Test integration with ConceptDriftDetector for drift analysis
  - Test integration with PredictionValidator for accuracy validation
  - Test integration with model registry for model retrieval and updates
  - Test integration with feature engineering engine for data preparation
  - Test integration with notification callbacks for event handling
  - Test integration with AccuracyMetrics and DriftMetrics data structures
  - Test integration with TrainingResult and PredictionResult from model training
  - Test integration with sklearn.model_selection.train_test_split
  - Test complete retraining workflow from trigger to deployment
  - Test concurrent retraining requests with resource management
  - Test background task lifecycle management and proper shutdown
  - Test cross-component integration with tracking manager
  - Test MQTT/API integration for retraining status reporting

- Edge Cases:
  - Test with adaptive_retraining_enabled=False (disabled system)
  - Test with no model registry entries for target model
  - Test with empty training data preparation
  - Test with validation_split=0 (no validation data)
  - Test with max_concurrent_retrains=0 (no concurrent limit)
  - Test with retraining_cooldown_hours=0 (no cooldown)
  - Test with missing feature_engineering_engine
  - Test with missing drift_detector or prediction_validator
  - Test with empty notification_callbacks list
  - Test with model that doesn't support incremental_update
  - Test with model that doesn't support ensemble rebalancing
  - Test with extremely high/low priority values
  - Test with duplicate request_ids in queue
  - Test with corrupted progress tracking data
  - Test with request cancellation during active training
  - Test with background task failures and recovery
  - Test with optimization timeout scenarios
  - Test with training data validation failures
  - Test with model deployment validation failures
  - Test with callback notification failures
  - Test with timezone-aware datetime handling across different regions
  - Test with memory constraints during large model retraining
  - Test with network failures during distributed training
  - Test with long-running retraining operations and timeout handling
  - Test with retraining history overflow (deque maxlen=1000)

- Error Handling:
  - Test RetrainingError raising with proper error_code and severity
  - Test initialization failure with RetrainingError in initialize method
  - Test queue management errors with proper error logging
  - Test resource lock contention and deadlock prevention
  - Test training failure handling with status updates and cleanup
  - Test validation failure handling with proper error propagation
  - Test callback failure handling without affecting main operation
  - Test background task exception handling and logging
  - Test model registry access failures
  - Test feature extraction failures with fallback mechanisms
  - Test optimization failure handling with default parameter fallback
  - Test data preparation failures with empty DataFrame handling
  - Test model saving/loading failures during deployment
  - Test progress tracking corruption handling
  - Test cooldown tracking failures
  - Test statistics calculation errors with default values
  - Test shutdown errors during cleanup operations
  - Test concurrent access errors with proper locking
  - Test memory allocation failures during large operations
  - Test disk space failures during model serialization
  - Test network timeout errors during distributed operations

- Coverage Target: 85%+

### src/models/training_pipeline.py - Model Training Pipeline
**Classes Found:** TrainingStage, TrainingType, ValidationStrategy, TrainingConfig, TrainingProgress, DataQualityReport, ModelTrainingPipeline
**Methods Analyzed:** __init__, run_initial_training, run_incremental_training, run_retraining_pipeline, train_room_models, _prepare_training_data, _query_room_events, _validate_data_quality, _can_proceed_with_quality_issues, _extract_features_and_targets, _split_training_data, _time_series_split, _expanding_window_split, _rolling_window_split, _holdout_split, _train_models, _validate_models, _evaluate_and_select_best_model, _meets_quality_thresholds, _deploy_trained_models, _generate_model_version, _save_model_artifacts, _cleanup_training_artifacts, _register_trained_models, _notify_tracking_manager_of_completion, _update_training_stats, get_active_pipelines, get_pipeline_history, get_training_statistics, get_model_registry, get_model_versions, get_model_performance, load_model_from_artifacts, compare_models, _is_valid_model_type, _create_model_instance, _invoke_callback_if_configured

**Required Tests:**
- Unit Tests:
  - Test TrainingConfig dataclass initialization with all parameters and defaults
  - Test TrainingProgress.update_stage method with stage transitions and progress calculations
  - Test DataQualityReport creation and add_recommendation method
  - Test ModelTrainingPipeline.__init__ with different configuration options
  - Test pipeline artifacts path creation and directory structure setup
  - Test run_initial_training with various room_ids configurations (None, specific list)
  - Test run_initial_training parallel processing with semaphore limits
  - Test run_incremental_training with different model types and new data periods
  - Test run_retraining_pipeline with different trigger reasons and strategies
  - Test train_room_models complete pipeline execution with all stages
  - Test _prepare_training_data with different lookback periods and database scenarios
  - Test _query_room_events mock data generation and error handling
  - Test _validate_data_quality with comprehensive data validation scenarios
  - Test _can_proceed_with_quality_issues decision logic with various quality issues
  - Test _extract_features_and_targets mock feature generation and integration
  - Test all data splitting strategies: time_series_split, expanding_window_split, rolling_window_split, holdout_split
  - Test _train_models with ensemble and specific model type training
  - Test _validate_models with prediction generation and scoring metrics
  - Test _evaluate_and_select_best_model with different selection metrics (mae, rmse, r2)
  - Test _meets_quality_thresholds with various accuracy and error combinations
  - Test _deploy_trained_models with model registry registration and versioning
  - Test _generate_model_version unique identifier generation
  - Test _save_model_artifacts with pickle serialization and metadata storage
  - Test _cleanup_training_artifacts cleanup operations
  - Test model registry and version tracking operations
  - Test training statistics tracking and updates
  - Test callback invocation mechanisms for progress, optimization, and validation
  - Test _is_valid_model_type with ModelType enum validation
  - Test _create_model_instance dynamic model creation for all supported types

- Integration Tests:
  - Test complete training pipeline execution from initialization to deployment
  - Test integration with FeatureEngineeringEngine and FeatureStore
  - Test integration with database_manager for data retrieval
  - Test integration with TrackingManager for model registration and monitoring
  - Test ensemble model integration with base predictors
  - Test model artifact persistence and loading across pipeline runs
  - Test parallel training execution with resource limits
  - Test cross-validation strategy execution with temporal data
  - Test model comparison and A/B testing functionality
  - Test pipeline failure recovery and error propagation
  - Test training progress callbacks and monitoring integration
  - Test model deployment to production registry
  - Test incremental training with existing models
  - Test retraining pipeline triggered by accuracy degradation

- Edge Cases:
  - Test with insufficient training data (below min_samples_per_room threshold)
  - Test with empty or malformed raw data from database
  - Test with extreme data quality issues (100% missing values, no timestamps)
  - Test with data freshness validation failures
  - Test with temporal consistency validation failures (non-monotonic timestamps)
  - Test with feature extraction failures returning empty DataFrames
  - Test with data splitting edge cases (very small datasets, single samples)
  - Test with model training failures for all base models
  - Test with validation failures due to prediction errors
  - Test with model evaluation failures and score calculation errors
  - Test with model deployment failures during artifact saving
  - Test with pickle serialization failures for complex models
  - Test with metadata serialization failures for JSON export
  - Test with model registry conflicts and version collisions
  - Test with cleanup failures during artifact management
  - Test with tracking manager integration failures
  - Test with callback function execution failures
  - Test with extreme training configurations (zero splits, huge lookback)
  - Test with concurrent pipeline execution conflicts
  - Test with model loading failures from corrupted artifacts
  - Test with model comparison on non-existent or failed models

- Error Handling:
  - Test ModelTrainingError raising with proper model_type, room_id, and cause
  - Test InsufficientTrainingDataError with data_points and minimum_required
  - Test OccupancyPredictionError during data splitting with context details
  - Test exception handling in run_initial_training with training task failures
  - Test exception handling in run_incremental_training with proper error propagation
  - Test exception handling in run_retraining_pipeline with trigger context
  - Test exception handling in train_room_models with stage-specific error tracking
  - Test exception handling in _prepare_training_data with database failures
  - Test exception handling in _validate_data_quality with calculation errors
  - Test exception handling in _extract_features_and_targets with engine failures
  - Test exception handling in data splitting methods with validation errors
  - Test exception handling in _train_models with individual model failures
  - Test exception handling in _validate_models with prediction failures
  - Test exception handling in _evaluate_and_select_best_model with empty results
  - Test exception handling in _deploy_trained_models with artifact saving failures
  - Test exception handling in _save_model_artifacts with file system errors
  - Test exception handling in model loading with corrupted or missing files
  - Test exception handling in training statistics updates with invalid data
  - Test exception handling in callback invocation with malformed functions
  - Test exception handling in model registry operations with concurrent access
  - Test exception handling in pipeline cleanup with resource conflicts
  - Test progress tracking during failures with proper stage updates
  - Test error message formatting and logging for all failure scenarios
  - Test graceful degradation with partial model training failures
  - Test timeout handling for long-running training operations
  - Test resource limit enforcement during parallel training

- Coverage Target: 85%+

### src/adaptation/validator.py - Prediction Validation
**Classes Found:** 
- ValidationStatus (Enum)
- AccuracyLevel (Enum) 
- ValidationRecord (dataclass)
- AccuracyMetrics (dataclass)
- PredictionValidator (main class)
- ValidationError (custom exception)

**Methods Analyzed:**
ValidationRecord: validate_against_actual, mark_expired, mark_failed, to_dict
AccuracyMetrics: __post_init__, validation_rate, expiration_rate, bias_direction, confidence_calibration_score, to_dict
PredictionValidator: __init__, start_background_tasks, stop_background_tasks, record_prediction, validate_prediction, get_accuracy_metrics, get_room_accuracy, get_model_accuracy, get_overall_accuracy, get_pending_validations, expire_old_predictions, export_validation_data, get_validation_stats, cleanup_old_records, cleanup_old_predictions, get_performance_stats, get_total_predictions, get_validation_rate, get_accuracy_trend, get_database_accuracy_statistics, plus 20+ private helper methods

**Required Tests:**

- Unit Tests:
  - Test ValidationRecord validation with different error thresholds (5, 10, 15, 30+ minutes)
  - Test ValidationRecord status transitions (pending -> validated/expired/failed)
  - Test ValidationRecord accuracy level classification (excellent/good/acceptable/poor/unacceptable)
  - Test ValidationRecord serialization/deserialization (to_dict, datetime handling)
  - Test AccuracyMetrics property calculations (validation_rate, expiration_rate, bias_direction)
  - Test AccuracyMetrics confidence calibration score computation
  - Test AccuracyMetrics alternative parameter name handling (avg_error_minutes, etc.)
  - Test PredictionValidator initialization with all configuration parameters
  - Test record_prediction with PredictionResult objects vs individual parameters
  - Test record_prediction thread-safety with concurrent access
  - Test validate_prediction with various time windows and transition types
  - Test validate_prediction transition type matching (flexible rules)
  - Test get_accuracy_metrics with different filtering combinations (room, model, time)
  - Test get_accuracy_metrics caching behavior and cache invalidation
  - Test get_pending_validations with room filtering and expiration status
  - Test expire_old_predictions with custom cutoff times
  - Test cleanup_old_records memory management and index updates
  - Test export_validation_data in CSV and JSON formats
  - Test get_validation_stats comprehensive statistics calculation
  - Test get_performance_stats with cache hit rates and processing metrics
  - Test get_accuracy_trend with different interval configurations
  - Test metrics calculation from records (error statistics, bias analysis)
  - Test confidence analysis (calibration, overconfidence, underconfidence rates)
  - Test background task lifecycle (start/stop, cleanup loops)
  - Test batch database operations (inserts, updates, queries)
  - Test _transition_types_match flexible matching rules
  - Test _calculate_metrics_from_records with various record sets
  - Test cache management (TTL, size limits, invalidation)
  - Test deque-based queue operations for batch processing

- Integration Tests:
  - Test database integration for prediction storage and retrieval
  - Test database batch operations with multiple records
  - Test database accuracy statistics with SQL aggregations
  - Test AsyncSession usage and proper transaction handling
  - Test database error handling and graceful degradation
  - Test cache behavior across database and memory operations
  - Test background task coordination with database operations
  - Test concurrent prediction recording and validation
  - Test large dataset handling (10k+ predictions)
  - Test cross-room and cross-model accuracy analysis
  - Test export functionality with real data sets
  - Test cleanup operations impact on database consistency

- Edge Cases:
  - Test with zero predictions recorded
  - Test with all predictions expired or failed
  - Test with identical prediction times (timestamp collisions)
  - Test with extreme error values (negative, very large)
  - Test with missing or invalid datetime values
  - Test with corrupted validation records
  - Test with memory limits exceeded (cleanup triggers)
  - Test with database connection failures during operations
  - Test with timezone-aware datetime handling across UTC
  - Test with very short or very long validation time windows
  - Test with empty or null model types and room IDs
  - Test with malformed prediction intervals or alternatives
  - Test with circular dependencies in validation chains
  - Test with cache size limits and eviction policies
  - Test with background task cancellation scenarios
  - Test with CSV/JSON export of empty or large datasets
  - Test with statistical calculations on single-sample datasets
  - Test with confidence scores outside 0.0-1.0 range
  - Test with duplicate prediction IDs or validation attempts

- Error Handling:
  - Test ValidationError raising with proper error codes and severity
  - Test DatabaseError handling in batch operations
  - Test exception handling in background tasks (graceful failure)
  - Test error handling during database connection failures
  - Test error handling in CSV/JSON export operations
  - Test error handling in metrics calculation with invalid data
  - Test error handling in cache operations (corruption, memory issues)
  - Test error handling in thread synchronization scenarios
  - Test error handling in statistical calculations (division by zero)
  - Test error handling in datetime operations (timezone issues)
  - Test error handling in file I/O operations during export
  - Test error handling in SQLAlchemy operations (session management)
  - Test error handling in numpy operations (NaN, infinity values)
  - Test error handling in JSON serialization of complex objects
  - Test error handling during cleanup operations with locked resources
  - Test error handling in deque operations (empty queue scenarios)
  - Test error handling in async/await operations (cancellation, timeout)
  - Test error handling in model type conversions (enum vs string)
  - Test error handling in prediction ID generation (uniqueness violations)
  - Test error handling in background task shutdown scenarios

- Coverage Target: 85%+

### src/adaptation/optimizer.py - Model Optimization
**Classes Found:** OptimizationStrategy, OptimizationObjective, OptimizationStatus, HyperparameterSpace, OptimizationResult, OptimizationConfig, ModelOptimizer, OptimizationError
**Methods Analyzed:** 
- HyperparameterSpace: __init__, _validate_parameters, get_parameter_names, is_continuous, get_bounds, get_choices, sample, to_dict
- OptimizationResult: to_dict
- OptimizationConfig: __post_init__
- ModelOptimizer: __init__, optimize_model_parameters, get_cached_parameters, get_optimization_stats, _should_optimize, _get_parameter_space, _adapt_parameter_space, _create_objective_function, _create_model_with_params, _bayesian_optimization, _run_bayesian_optimization_async, _grid_search_optimization, _random_search_optimization, _performance_adaptive_optimization, _create_default_result, _update_improvement_average, _initialize_parameter_spaces, _measure_prediction_latency, _get_baseline_performance, _generate_hyperparameter_combinations, _update_performance_history, _measure_memory_usage

**Required Tests:**
- Unit Tests:
  - OptimizationStrategy enum values (BAYESIAN, GRID_SEARCH, RANDOM_SEARCH, GRADIENT_BASED, PERFORMANCE_ADAPTIVE)
  - OptimizationObjective enum values (ACCURACY, CONFIDENCE_CALIBRATION, PREDICTION_TIME, DRIFT_RESISTANCE, COMPOSITE)
  - OptimizationStatus enum values (PENDING, INITIALIZING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT)
  - HyperparameterSpace constructor with valid tuple/list parameters
  - HyperparameterSpace._validate_parameters with valid/invalid parameter definitions
  - HyperparameterSpace.get_parameter_names returns correct list
  - HyperparameterSpace.is_continuous correctly identifies parameter types
  - HyperparameterSpace.get_bounds for continuous parameters
  - HyperparameterSpace.get_choices for discrete parameters
  - HyperparameterSpace.sample generates correct number of samples
  - HyperparameterSpace.to_dict serialization format
  - OptimizationResult.to_dict complete serialization
  - OptimizationConfig.__post_init__ n_calls/n_initial_points validation
  - ModelOptimizer.__init__ with various configurations
  - ModelOptimizer.optimize_model_parameters with different model types
  - ModelOptimizer.get_cached_parameters retrieval and copying
  - ModelOptimizer.get_optimization_stats calculation accuracy
  - ModelOptimizer._should_optimize decision logic with performance context
  - ModelOptimizer._get_parameter_space retrieval for different model types
  - ModelOptimizer._adapt_parameter_space based on performance context
  - ModelOptimizer._create_objective_function with different objectives
  - ModelOptimizer._create_model_with_params with/without set_parameters method
  - ModelOptimizer._bayesian_optimization with/without skopt
  - ModelOptimizer._run_bayesian_optimization_async parameter space conversion
  - ModelOptimizer._grid_search_optimization parameter grid generation
  - ModelOptimizer._random_search_optimization parameter sampling
  - ModelOptimizer._performance_adaptive_optimization with performance trends
  - ModelOptimizer._create_default_result structure and values
  - ModelOptimizer._update_improvement_average calculation
  - ModelOptimizer._initialize_parameter_spaces for all model types (LSTM, XGBoost, HMM, Gaussian Process)
  - ModelOptimizer._measure_prediction_latency timing accuracy
  - ModelOptimizer._get_baseline_performance with different model interfaces
  - ModelOptimizer._generate_hyperparameter_combinations grid vs random strategies
  - ModelOptimizer._update_performance_history sliding window management
  - ModelOptimizer._measure_memory_usage with different measurement methods
  - OptimizationError inheritance and initialization

- Integration Tests:
  - End-to-end optimization with real BasePredictor models
  - Integration with accuracy_tracker and drift_detector components
  - Bayesian optimization with scikit-optimize when available
  - Grid search optimization with sklearn.model_selection.ParameterGrid
  - Parameter caching and retrieval across optimization sessions
  - Performance history tracking and trend analysis
  - Multi-objective optimization with composite scoring
  - Adaptive optimization based on model performance degradation
  - Thread safety with concurrent optimization operations
  - Memory usage monitoring during optimization
  - Integration with training data validation and splitting

- Edge Cases:
  - HyperparameterSpace with empty parameters dictionary
  - HyperparameterSpace with single-element tuple/list parameters
  - Parameter validation with invalid tuple lengths (not 2-tuple)
  - Parameter validation with min >= max continuous bounds
  - Parameter validation with empty discrete choice lists
  - Parameter sampling with n_samples = 0
  - OptimizationConfig with n_calls < n_initial_points
  - ModelOptimizer with disabled optimization (config.enabled = False)
  - Optimization with missing model type in parameter spaces
  - Optimization with empty training/validation data
  - Bayesian optimization fallback when skopt unavailable
  - Grid search with excessive parameter combinations
  - Random search with zero n_calls
  - Performance-adaptive optimization with empty performance history
  - Objective function returning inf/-inf scores
  - Model creation failure during parameter setting
  - Prediction latency measurement with models lacking predict method
  - Memory usage measurement fallbacks when methods fail
  - Optimization timeout handling
  - Parameter space adaptation with malformed performance context
  - Concurrent access to optimization history and parameter cache
  - Performance history sliding window with single entry

- Error Handling:
  - HyperparameterSpace constructor with non-tuple/non-list parameters
  - Parameter name not found in space during bounds/choices retrieval
  - Continuous parameter bounds requested for discrete parameter
  - Discrete parameter choices requested for continuous parameter
  - OptimizationConfig validation failures
  - ModelOptimizer initialization with None config
  - Optimization with corrupted performance context
  - Model training failures during objective function evaluation
  - AsyncIO handling errors in objective function
  - Bayesian optimization failures and fallback to random search
  - Grid search parameter space conversion errors
  - Random search parameter generation failures
  - Model copying/deepcopy failures in _create_model_with_params
  - Thread safety violations during concurrent optimization
  - Memory measurement failures and fallback values
  - Prediction latency measurement errors and defaults
  - Performance history update failures
  - Parameter cache corruption handling
  - OptimizationError proper error codes and severity levels
  - Exception propagation through async optimization methods
  - Timeout handling during long-running optimizations
  - Resource cleanup after failed optimization attempts

- Coverage Target: 85%+

### src/adaptation/monitoring_enhanced_tracking.py - Enhanced Monitoring
**Classes Found:** MonitoringEnhancedTrackingManager
**Methods Analyzed:** __init__, _wrap_tracking_methods, _monitored_record_prediction, _monitored_validate_prediction, _monitored_start_tracking, _monitored_stop_tracking, record_concept_drift, record_feature_computation, record_database_operation, record_mqtt_publish, update_connection_status, get_monitoring_status, track_model_training, __getattr__, create_monitoring_enhanced_tracking_manager, get_enhanced_tracking_manager

**Required Tests:**
- Unit Tests:
  - MonitoringEnhancedTrackingManager.__init__ with valid TrackingManager instance
  - _wrap_tracking_methods correctly stores original methods and replaces with monitored versions
  - _monitored_record_prediction calls original method and records monitoring data
  - _monitored_record_prediction with PredictionResult containing confidence values
  - _monitored_record_prediction with PredictionResult without confidence values
  - _monitored_record_prediction with ModelType enum vs string model types
  - _monitored_validate_prediction calls original method and records accuracy metrics
  - _monitored_validate_prediction with valid result dictionary containing accuracy data
  - _monitored_validate_prediction with malformed result dictionary
  - _monitored_start_tracking starts monitoring and original tracking in correct order
  - _monitored_start_tracking success alert triggering
  - _monitored_stop_tracking stops original tracking first, then monitoring
  - _monitored_stop_tracking with exception handling and monitoring cleanup
  - record_concept_drift calls monitoring integration and optionally tracking manager
  - record_concept_drift when tracking_manager lacks record_concept_drift method
  - record_feature_computation metric recording with valid parameters
  - record_database_operation metric recording with default and custom status
  - record_mqtt_publish metric recording with topic and room parameters
  - update_connection_status with various connection types and states
  - get_monitoring_status returns comprehensive status with monitoring and tracking data
  - get_monitoring_status with tracking manager status retrieval failure
  - track_model_training context manager functionality
  - __getattr__ delegation to original tracking manager
  - create_monitoring_enhanced_tracking_manager factory function
  - get_enhanced_tracking_manager convenience function

- Integration Tests:
  - Full integration with real TrackingManager and MonitoringIntegration instances
  - End-to-end prediction recording and validation workflow with monitoring
  - Start/stop tracking lifecycle with monitoring system integration
  - Concept drift recording propagation to both monitoring and tracking systems
  - Monitoring status aggregation from multiple components
  - Method wrapping preserves original functionality while adding monitoring
  - Alert triggering integration during system startup and validation errors
  - Context manager integration for model training operations

- Edge Cases:
  - MonitoringEnhancedTrackingManager with None tracking_manager
  - Method wrapping when original methods don't exist on tracking_manager
  - _monitored_record_prediction with None or malformed PredictionResult
  - _monitored_validate_prediction with None actual_time parameter
  - _monitored_start_tracking when monitoring integration start_monitoring fails
  - _monitored_stop_tracking when both original and monitoring stop methods fail
  - record_concept_drift with invalid parameters (None room_id, negative severity)
  - get_monitoring_status when monitoring_integration.get_monitoring_status fails
  - __getattr__ with non-existent attributes on tracking_manager
  - track_model_training context manager with exception inside context
  - Factory functions with invalid TrackingConfig parameters
  - Concurrent access to wrapped methods during monitoring operations
  - Monitoring integration unavailable during initialization
  - Method delegation when tracking_manager is None after initialization

- Error Handling:
  - Exception propagation in _monitored_record_prediction during original method call
  - Exception handling in _monitored_validate_prediction with alert triggering
  - Exception propagation in _monitored_start_tracking with startup failure alert
  - Exception handling in _monitored_stop_tracking with cleanup attempt
  - Alert manager failures during error alert triggering
  - MonitoringIntegration method call failures (start_monitoring, stop_monitoring)
  - TrackingManager original method call failures during monitoring operations
  - PredictionResult attribute access failures (confidence, prediction_type)
  - ModelType enum value extraction failures
  - DateTime operations failures during validation monitoring
  - Async operation failures within context managers
  - get_monitoring_integration() function failures during initialization
  - Method wrapping failures when tracking_manager methods are not callable
  - __getattr__ failures when neither class has the requested attribute
  - Resource cleanup failures during exception handling
  - Monitoring data recording failures with fallback behaviors
  - Alert triggering timeout or connection failures
  - Concurrent modification issues during method wrapping
  - Memory leaks prevention during long-running monitoring operations

- Coverage Target: 85%+

### src/integration/websocket_api.py - WebSocket API
**Classes Found:** WebSocketEndpoint (enum), MessageType (enum), WebSocketMessage (dataclass), ClientAuthRequest (BaseModel), ClientSubscription (BaseModel), ClientConnection (dataclass), WebSocketStats (dataclass), WebSocketConnectionManager, WebSocketAPIServer
**Methods Analyzed:** WebSocketMessage.to_json(), WebSocketMessage.from_json(), ClientConnection.__post_init__(), ClientConnection.update_activity(), ClientConnection.update_heartbeat(), ClientConnection.is_rate_limited(), ClientConnection.increment_message_count(), ClientConnection.apply_rate_limit(), ClientConnection.can_access_room(), ClientConnection.has_capability(), WebSocketConnectionManager.__init__(), WebSocketConnectionManager.connect(), WebSocketConnectionManager.disconnect(), WebSocketConnectionManager.authenticate_client(), WebSocketConnectionManager.subscribe_client(), WebSocketConnectionManager.unsubscribe_client(), WebSocketConnectionManager.send_message(), WebSocketConnectionManager.broadcast_to_endpoint(), WebSocketConnectionManager.process_acknowledgment(), WebSocketConnectionManager.send_heartbeat(), WebSocketConnectionManager.get_connection_stats(), WebSocketAPIServer.__init__(), WebSocketAPIServer.initialize(), WebSocketAPIServer.start(), WebSocketAPIServer.stop(), WebSocketAPIServer.publish_prediction_update(), WebSocketAPIServer.publish_system_status_update(), WebSocketAPIServer.publish_alert_notification(), WebSocketAPIServer.get_server_stats(), websocket_endpoint(), health_endpoint(), create_websocket_app(), create_websocket_api_server(), websocket_api_context()

**Required Tests:**
- Unit Tests:
  - WebSocketEndpoint enum values (PREDICTIONS, SYSTEM_STATUS, ALERTS, ROOM_SPECIFIC)
  - MessageType enum values (CONNECTION, AUTHENTICATION, HEARTBEAT, SUBSCRIBE, UNSUBSCRIBE, SUBSCRIPTION_STATUS, PREDICTION_UPDATE, SYSTEM_STATUS_UPDATE, ALERT_NOTIFICATION, DRIFT_NOTIFICATION, ACKNOWLEDGE, ERROR, RATE_LIMIT_WARNING)
  - WebSocketMessage.to_json() serialization with all fields
  - WebSocketMessage.from_json() deserialization with valid JSON
  - WebSocketMessage round-trip serialization/deserialization integrity
  - ClientAuthRequest validation with valid/invalid API keys
  - ClientAuthRequest.validate_api_key() minimum length requirement
  - ClientSubscription endpoint pattern validation
  - ClientSubscription with valid/invalid room_id patterns
  - ClientConnection.__post_init__() default value initialization
  - ClientConnection.update_activity() timestamp updates
  - ClientConnection.update_heartbeat() combined timestamp updates
  - ClientConnection.is_rate_limited() with various message counts and time windows
  - ClientConnection.increment_message_count() counter updates
  - ClientConnection.apply_rate_limit() duration setting
  - ClientConnection.can_access_room() with empty/populated room filters
  - ClientConnection.has_capability() with various capabilities
  - WebSocketStats dataclass initialization and field access
  - WebSocketConnectionManager.__init__() with default/custom config
  - WebSocketConnectionManager.connect() new connection registration
  - WebSocketConnectionManager.connect() max connections enforcement
  - WebSocketConnectionManager.disconnect() connection cleanup
  - WebSocketConnectionManager.authenticate_client() success/failure paths
  - WebSocketConnectionManager.subscribe_client() validation and room access
  - WebSocketConnectionManager.unsubscribe_client() subscription removal
  - WebSocketConnectionManager.send_message() rate limiting checks
  - WebSocketConnectionManager.broadcast_to_endpoint() targeted delivery
  - WebSocketConnectionManager.process_acknowledgment() message removal
  - WebSocketConnectionManager.send_heartbeat() message creation and sending
  - WebSocketConnectionManager.get_connection_stats() calculation accuracy
  - WebSocketConnectionManager._update_endpoint_stats() counter updates
  - WebSocketConnectionManager._send_rate_limit_warning() warning message format
  - WebSocketAPIServer.__init__() initialization with/without tracking_manager
  - WebSocketAPIServer.initialize() background task creation
  - WebSocketAPIServer.start() WebSocket server startup
  - WebSocketAPIServer.stop() graceful shutdown process
  - WebSocketAPIServer.publish_prediction_update() message broadcasting
  - WebSocketAPIServer.publish_system_status_update() status distribution  
  - WebSocketAPIServer.publish_alert_notification() alert delivery with acknowledgment
  - WebSocketAPIServer.get_server_stats() comprehensive statistics gathering
  - WebSocketAPIServer._normalize_endpoint() path standardization
  - WebSocketAPIServer._format_prediction_data() PredictionResult conversion
  - WebSocketAPIServer._format_time_until() human-readable time formatting
  - WebSocketAPIServer._heartbeat_loop() periodic heartbeat sending
  - WebSocketAPIServer._cleanup_loop() stale connection removal
  - WebSocketAPIServer._acknowledgment_timeout_loop() message timeout handling
  - WebSocketAPIServer._handle_websocket_connection() connection lifecycle management
  - WebSocketAPIServer._process_client_message() message type routing
  - WebSocketAPIServer._handle_authentication() auth request processing
  - WebSocketAPIServer._handle_subscription() subscription request processing
  - WebSocketAPIServer._handle_unsubscription() unsubscription request processing
  - WebSocketAPIServer._handle_heartbeat_response() heartbeat acknowledgment
  - WebSocketAPIServer._handle_acknowledgment() message acknowledgment processing
  - WebSocketAPIServer._send_error_message() error response formatting
  - WebSocketAPIServer._close_all_connections() batch connection cleanup
  - WebSocketAPIServer._register_with_tracking_manager() callback registration
  - WebSocketAPIServer._handle_tracking_manager_callback() callback processing
  - websocket_endpoint() Starlette WebSocket handler
  - health_endpoint() health check responses
  - create_websocket_app() Starlette application setup
  - create_websocket_api_server() factory function with default config
  - websocket_api_context() context manager lifecycle

- Integration Tests:
  - End-to-end WebSocket connection flow with authentication
  - Real-time prediction updates through WebSocket to connected clients
  - Multiple concurrent client connections with different subscriptions
  - Room-specific filtering and access control validation
  - Rate limiting enforcement across multiple message types
  - Heartbeat mechanism maintaining connection liveness
  - Message acknowledgment flow for critical alerts
  - Integration with TrackingManager for automatic prediction updates
  - WebSocket server startup/shutdown with background task management
  - CORS middleware functionality with various origins
  - Health endpoints accessibility and response format
  - Starlette application routing to correct WebSocket handlers
  - Connection cleanup under various failure scenarios
  - Statistics collection accuracy across multiple connections
  - Subscription management with dynamic room access changes
  - Message broadcasting performance with many concurrent connections
  - WebSocket upgrade negotiation and protocol compliance
  - Error handling integration between connection manager and server
  - Configuration loading and application to connection limits
  - Background loop coordination during server lifecycle

- Edge Cases:
  - WebSocketMessage.from_json() with malformed JSON data
  - WebSocketMessage with missing required fields
  - ClientAuthRequest with extremely long API key strings
  - ClientConnection with None values for optional datetime fields
  - ClientConnection rate limiting edge cases (exactly at limit, reset timing)
  - WebSocketConnectionManager at exactly max_connections limit
  - WebSocketConnectionManager.authenticate_client() with disabled API key
  - WebSocketConnectionManager.subscribe_client() without authentication
  - WebSocketConnectionManager.send_message() to disconnected client
  - WebSocketConnectionManager.broadcast_to_endpoint() with no subscribers
  - WebSocketAPIServer.initialize() with DISABLE_BACKGROUND_TASKS set
  - WebSocketAPIServer background loops during rapid shutdown
  - WebSocketAPIServer._format_prediction_data() with None prediction_result fields
  - WebSocketAPIServer._format_time_until() with negative seconds
  - WebSocket connection handling with immediate disconnection
  - Message processing with unsupported MessageType values
  - Authentication with missing or corrupted auth data
  - Subscription to non-existent endpoints
  - Heartbeat responses from already-disconnected clients
  - Acknowledgment for non-existent message IDs
  - Connection cleanup during active message broadcasting
  - Statistics calculation with rapidly changing connection counts
  - Rate limit warning sending to rate-limited clients
  - TrackingManager callback with None or invalid data
  - WebSocket server binding to already-occupied port
  - Context manager exception handling during startup/shutdown
  - Concurrent access to connection dictionaries during updates

- Error Handling:
  - WebSocketMessage.from_json() JSON decode errors
  - WebSocketMessage.to_json() serialization failures
  - ClientAuthRequest validation failures for invalid API keys
  - ClientSubscription pattern matching failures
  - ClientConnection initialization with corrupted data
  - WebSocketConnectionManager connection registration failures
  - WebSocketConnectionManager.connect() with invalid WebSocket protocol
  - WebSocketConnectionManager.authenticate_client() with WebSocketAuthenticationError
  - WebSocketConnectionManager.subscribe_client() with WebSocketValidationError
  - WebSocketConnectionManager.send_message() WebSocket send failures
  - WebSocketConnectionManager rate limiting violations and warnings
  - WebSocketAPIServer.initialize() WebSocketConnectionError handling
  - WebSocketAPIServer.start() server binding failures
  - WebSocketAPIServer.stop() shutdown error propagation
  - WebSocketAPIServer background task exception handling
  - WebSocketAPIServer._handle_websocket_connection() protocol errors
  - WebSocketAPIServer._process_client_message() message parsing errors
  - WebSocketAPIServer._handle_authentication() processing errors
  - WebSocketAPIServer._handle_subscription() subscription processing errors
  - WebSocketAPIServer._send_error_message() error message delivery failures
  - WebSocket connection timeout and network failures
  - Message queue overflow handling
  - Memory exhaustion during large message broadcasts
  - Concurrent modification exceptions in connection management
  - Background loop exception recovery and logging
  - TrackingManager integration failure handling
  - Configuration validation and default fallback errors
  - WebSocket protocol version mismatches
  - SSL/TLS handshake failures (if applicable)
  - Resource cleanup failures during error conditions

- Coverage Target: 85%+

### src/adaptation/tracking_manager.py - Tracking Management
**Classes Found:** TrackingConfig, TrackingManager, TrackingManagerError
**Methods Analyzed:** 49 public methods across main classes including initialization, monitoring, validation, drift detection, MQTT integration, API server management, dashboard integration, and WebSocket API

**Required Tests:**
- Unit Tests:
  - TrackingConfig initialization with default and custom parameters
  - TrackingConfig.__post_init__ alert threshold defaults setting
  - TrackingManager.__init__ with various configuration combinations
  - TrackingManager.initialize() with all components (validator, tracker, drift detector, retrainer, MQTT)
  - TrackingManager.start_tracking() and background task creation
  - TrackingManager.stop_tracking() graceful shutdown
  - TrackingManager.record_prediction() with automatic MQTT publishing
  - TrackingManager.handle_room_state_change() triggering validation
  - TrackingManager.get_tracking_status() comprehensive status reporting
  - TrackingManager.get_real_time_metrics() for rooms and model types
  - TrackingManager.get_active_alerts() with filtering
  - TrackingManager.acknowledge_alert() alert acknowledgment
  - TrackingManager.check_drift() manual drift detection
  - TrackingManager.get_drift_status() drift system status
  - TrackingManager.request_manual_retraining() manual retraining requests
  - TrackingManager.get_retraining_status() and cancel_retraining()
  - TrackingManager.register_model() and unregister_model() model registry
  - TrackingManager.start_api_server() and stop_api_server() API integration
  - TrackingManager.get_api_server_status() API server state
  - TrackingManager.get_enhanced_mqtt_status() MQTT integration status
  - TrackingManager.get_realtime_publishing_status() real-time publishing
  - TrackingManager.get_websocket_api_status() WebSocket API status
  - TrackingManager.get_dashboard_status() dashboard status
  - TrackingManager.get_room_prediction() room-specific predictions
  - TrackingManager.get_accuracy_metrics() accuracy reporting
  - TrackingManager.trigger_manual_retrain() manual training trigger
  - TrackingManager.get_system_stats() comprehensive system statistics
  - TrackingManager.publish_system_status_update() WebSocket broadcasting
  - TrackingManager.publish_alert_notification() alert broadcasting
  - TrackingManager._calculate_uptime_seconds() timezone-aware uptime calculation
  - TrackingManager._validation_monitoring_loop() background validation
  - TrackingManager._check_for_room_state_changes() database state monitoring
  - TrackingManager._cleanup_loop() periodic cache cleanup
  - TrackingManager._drift_detection_loop() automatic drift detection
  - TrackingManager._perform_drift_detection() room drift checking
  - TrackingManager._get_rooms_with_recent_activity() room filtering
  - TrackingManager._handle_drift_detection_results() alert generation
  - TrackingManager._evaluate_accuracy_based_retraining() accuracy-triggered retraining
  - TrackingManager._evaluate_drift_based_retraining() drift-triggered retraining
  - TrackingManager._initialize_realtime_publishing() real-time system setup
  - TrackingManager._shutdown_realtime_publishing() graceful real-time shutdown
  - TrackingManager._initialize_websocket_api() WebSocket API setup
  - TrackingManager._shutdown_websocket_api() WebSocket API shutdown
  - TrackingManager._initialize_dashboard() dashboard setup and configuration
  - TrackingManager._shutdown_dashboard() dashboard graceful shutdown
  - TrackingManagerError exception with proper inheritance

- Integration Tests:
  - Full tracking manager lifecycle (initialize → start → record → validate → stop)
  - TrackingManager with real database manager and room state changes
  - TrackingManager with actual MQTT integration and Home Assistant publishing
  - Automatic validation triggered by database state changes
  - Drift detection integration with accuracy tracking and retraining
  - Enhanced MQTT manager integration with real-time broadcasting
  - API server integration with tracking manager coordination
  - WebSocket API server integration with real-time updates
  - Performance dashboard integration with tracking manager data
  - Prediction recording → validation → accuracy tracking → alerting flow
  - Manual retraining requests → adaptive retrainer → model registry
  - Background task coordination and graceful shutdown
  - Notification callback integration across all components
  - Multi-component failure recovery and graceful degradation
  - Real-time publishing across multiple channels (MQTT, WebSocket, SSE)

- Edge Cases:
  - TrackingConfig with None alert_thresholds triggering __post_init__
  - TrackingManager initialization with disabled components (None parameters)
  - TrackingManager.initialize() with missing dependencies (validator, tracker, etc.)
  - TrackingManager with tracking disabled (config.enabled = False)
  - Background task startup with DISABLE_BACKGROUND_TASKS environment variable
  - Prediction recording with invalid prediction_result metadata
  - Room state change handling with None/missing room_id
  - Database query failures in _check_for_room_state_changes()
  - Drift detection with insufficient data or failed feature engineering
  - MQTT publishing failures with graceful error handling
  - WebSocket API publishing with no connected clients
  - Dashboard initialization with unavailable dashboard components (DASHBOARD_AVAILABLE = False)
  - Model registration with duplicate model keys
  - Retraining requests with invalid model types or strategies
  - API server startup failures and fallback behavior
  - Real-time publishing with no enabled channels
  - Prediction cache cleanup with timezone-naive timestamps
  - Uptime calculation with timezone-naive start time
  - Component shutdown with partially initialized systems
  - Graceful fallback when enhanced features unavailable
  - Thread safety in multi-threaded prediction recording
  - Memory management in long-running monitoring loops

- Error Handling:
  - TrackingManagerError proper exception inheritance and error codes
  - TrackingManager.initialize() failure with component initialization errors
  - TrackingManager.start_tracking() with already active tracking
  - TrackingManager.record_prediction() with disabled tracking
  - Database connection failures in validation monitoring
  - MQTT connection failures during prediction publishing
  - Enhanced MQTT manager initialization failures with graceful fallback
  - WebSocket API server startup failures
  - Dashboard component import failures (ImportError handling)
  - Drift detector failures with error recovery
  - Adaptive retrainer communication failures
  - Model registry corruption or invalid model instances
  - API server binding failures (port in use, permission errors)
  - Background task cancellation during shutdown
  - Prediction cache corruption and recovery
  - Real-time publishing channel failures with fallback
  - Notification callback failures with error isolation
  - Concurrent access violations in thread-safe operations
  - Resource cleanup failures during shutdown
  - Exception propagation through async methods
  - Timeout handling in background loops
  - Memory pressure during cache cleanup
  - Database query timeouts and retry logic
  - WebSocket connection drops and reconnection
  - Alert generation failures with fallback logging

- Coverage Target: 85%+

### src/adaptation/tracker.py - Performance Tracking
**Classes Found:** AlertSeverity, TrendDirection, RealTimeMetrics, AccuracyAlert, AccuracyTracker, AccuracyTrackingError
**Methods Analyzed:** 45+ methods including properties, async methods, data processing, alert management, trend analysis, and monitoring loops

**Required Tests:**
- Unit Tests:
  - AlertSeverity enum values (INFO, WARNING, CRITICAL, EMERGENCY)
  - TrendDirection enum values (IMPROVING, STABLE, DEGRADING, UNKNOWN)
  - RealTimeMetrics.__init__ with various parameter combinations
  - RealTimeMetrics.overall_health_score calculation with different metric combinations
  - RealTimeMetrics.overall_health_score edge cases (zero predictions, extreme values)
  - RealTimeMetrics.is_healthy property with different health scores and trends
  - RealTimeMetrics.to_dict serialization with all optional fields (None/populated)
  - AccuracyAlert.__init__ with required and optional parameters
  - AccuracyAlert.age_minutes calculation with different time zones
  - AccuracyAlert.requires_escalation logic for each severity level and age thresholds
  - AccuracyAlert.acknowledge method updates (acknowledged_by, timestamp)
  - AccuracyAlert.resolve method updates and logging
  - AccuracyAlert.escalate method incrementing level and timestamp updates
  - AccuracyAlert.escalate edge cases (max escalations reached, already acknowledged/resolved)
  - AccuracyAlert.to_dict complete serialization with all fields
  - AccuracyTracker.__init__ with default and custom configurations
  - AccuracyTracker.start_monitoring task creation and flag setting
  - AccuracyTracker.start_monitoring already active scenario
  - AccuracyTracker.stop_monitoring graceful shutdown and task cleanup
  - AccuracyTracker.get_real_time_metrics with room_id filter
  - AccuracyTracker.get_real_time_metrics with model_type filter  
  - AccuracyTracker.get_real_time_metrics with both filters
  - AccuracyTracker.get_real_time_metrics global metrics (no filters)
  - AccuracyTracker.get_active_alerts with no filters
  - AccuracyTracker.get_active_alerts with room_id filter
  - AccuracyTracker.get_active_alerts with severity filter
  - AccuracyTracker.get_active_alerts sorting by severity and age
  - AccuracyTracker.acknowledge_alert successful acknowledgment
  - AccuracyTracker.acknowledge_alert non-existent alert
  - AccuracyTracker.get_accuracy_trends with room_id filter
  - AccuracyTracker.get_accuracy_trends global trends
  - AccuracyTracker.export_tracking_data with all options enabled/disabled
  - AccuracyTracker.export_tracking_data file writing and record counting
  - AccuracyTracker.add_notification_callback preventing duplicates
  - AccuracyTracker.remove_notification_callback
  - AccuracyTracker.get_tracker_stats complete statistics collection
  - AccuracyTracker._analyze_trend with minimum/sufficient data points
  - AccuracyTracker._analyze_trend linear regression and R-squared calculation
  - AccuracyTracker._analyze_trend direction determination (improving/stable/degrading)
  - AccuracyTracker._calculate_global_trend from individual trends
  - AccuracyTracker._calculate_validation_lag average calculation
  - AccuracyTracker._model_types_match with enum and string combinations
  - AccuracyTracker.update_from_accuracy_metrics dominant level detection
  - AccuracyTracker.extract_recent_validation_records filtering and sorting
  - AccuracyTracker._calculate_real_time_metrics window calculations
  - AccuracyTracker._calculate_real_time_metrics trend integration
  - AccuracyTracker._analyze_trend_for_entity with existing/missing history
  - AccuracyTracker._check_entity_alerts accuracy threshold checking
  - AccuracyTracker._check_entity_alerts error threshold checking
  - AccuracyTracker._check_entity_alerts trend degradation detection
  - AccuracyTracker._check_entity_alerts validation lag detection
  - AccuracyTracker._check_entity_alerts alert deduplication logic
  - AccuracyTracker._should_auto_resolve_alert for each alert condition type
  - AccuracyTracker._should_auto_resolve_alert improvement threshold logic
  - AccuracyTracker._notify_alert_callbacks with sync/async callbacks
  - AccuracyTrackingError initialization with custom severity

- Integration Tests:
  - AccuracyTracker integration with PredictionValidator
  - Real-time metrics calculation with actual ValidationRecord data
  - Alert creation and escalation workflow end-to-end
  - Background monitoring loops (_monitoring_loop and _alert_management_loop)
  - Thread-safe operations with concurrent metric updates and alert checking
  - Notification callback execution with real alert scenarios
  - Export functionality with realistic tracking data
  - Auto-resolution workflow with improving conditions
  - Trend analysis with time-series accuracy data
  - Memory management with large numbers of alerts and metrics
  - Integration with AccuracyLevel and ValidationRecord from validator module

- Edge Cases:
  - RealTimeMetrics with zero predictions in all windows
  - RealTimeMetrics.overall_health_score with extreme confidence values (0, 1)
  - RealTimeMetrics health calculation with missing trend data
  - AccuracyAlert age calculation across timezone boundaries
  - AccuracyAlert escalation after max escalations reached
  - AccuracyAlert requires_escalation with acknowledged/resolved states
  - AccuracyTracker with empty prediction_validator records
  - AccuracyTracker metrics calculation with no historical data
  - AccuracyTracker alert thresholds at boundary values
  - AccuracyTracker trend analysis with single data point
  - AccuracyTracker trend analysis with identical values (zero slope)
  - AccuracyTracker global trend with empty individual trends
  - AccuracyTracker validation lag with no recent records
  - AccuracyTracker model type matching with None values
  - AccuracyTracker alert deduplication with similar conditions
  - AccuracyTracker auto-resolution with marginal improvements
  - AccuracyTracker export with empty tracking data
  - AccuracyTracker notification callbacks with empty callback list
  - AccuracyTracker background tasks with immediate shutdown
  - AccuracyTracker metric updates with corrupted validator data
  - AccuracyTracker trend history with maxlen deque overflow
  - AccuracyTracker alert cleanup with very old alerts
  - AccuracyTracker statistics calculation with partially initialized state

- Error Handling:
  - RealTimeMetrics.to_dict with invalid model_type values
  - AccuracyAlert.to_dict serialization errors
  - AccuracyAlert escalation/acknowledgment/resolution logging failures
  - AccuracyTracker initialization with None prediction_validator
  - AccuracyTracker start_monitoring task creation failures
  - AccuracyTracker stop_monitoring with failed task gathering
  - AccuracyTracker get_real_time_metrics with lock acquisition errors
  - AccuracyTracker get_active_alerts with corrupted alert data
  - AccuracyTracker export_tracking_data file writing permissions/errors
  - AccuracyTracker notification callback execution failures
  - AccuracyTracker._update_real_time_metrics with validator lock errors
  - AccuracyTracker._calculate_real_time_metrics with validator API failures
  - AccuracyTracker._analyze_trend with malformed data points
  - AccuracyTracker._analyze_trend with statistics calculation errors (division by zero)
  - AccuracyTracker._calculate_validation_lag with invalid timestamps
  - AccuracyTracker._check_alert_conditions with corrupted metrics
  - AccuracyTracker._check_entity_alerts with alert creation failures
  - AccuracyTracker._should_auto_resolve_alert with missing metrics
  - AccuracyTracker._notify_alert_callbacks with callback exceptions
  - AccuracyTracker background loop exception handling and recovery
  - AccuracyTracker._monitoring_loop with asyncio cancellation
  - AccuracyTracker._alert_management_loop timeout handling
  - AccuracyTracker thread safety violations during concurrent operations
  - AccuracyTracker memory cleanup during exception scenarios
  - AccuracyTrackingError proper error code and context propagation
  - AccuracyTracker validator integration failures
  - AccuracyTracker trend analysis with infinite/NaN values
  - AccuracyTracker alert threshold validation failures
  - AccuracyTracker statistics collection with partial data corruption

- Coverage Target: 85%+

### src/integration/api_server.py - REST API Server
**Classes Found:** PredictionResponse, SystemHealthResponse, AccuracyMetricsResponse, ManualRetrainRequest, SystemStatsResponse, ErrorResponse, RateLimitTracker, APIServer
**Methods Analyzed:** 40+ endpoints, middleware functions, error handlers, dependency functions, utility methods

**Required Tests:**
- Unit Tests:
  - **Pydantic Model Validation Tests:**
    - PredictionResponse.validate_transition_type() with valid types ["occupied", "vacant", "occupied_to_vacant", "vacant_to_occupied", "state_change"] and invalid types
    - PredictionResponse.validate_confidence() with values 0.0, 0.5, 1.0, -0.1, 1.1
    - AccuracyMetricsResponse.validate_rate() with accuracy_rate and confidence_calibration bounds [0.0-1.0]
    - AccuracyMetricsResponse.validate_average_error() with negative values rejection
    - AccuracyMetricsResponse.validate_counts() with non-negative integer validation
    - AccuracyMetricsResponse.validate_trend_direction() with ["improving", "stable", "degrading"] and invalid values
    - ManualRetrainRequest.validate_room_id() with config.rooms lookup and invalid room_id handling
    - ManualRetrainRequest.validate_strategy() with ["auto", "incremental", "full", "feature_refresh"] patterns
    - ManualRetrainRequest.validate_reason() with empty string and whitespace handling
    - ErrorResponse.dict() datetime serialization with .isoformat()

  - **Rate Limiting Tests:**
    - RateLimitTracker.__init__() empty requests dictionary initialization
    - RateLimitTracker.is_allowed() window calculation with timedelta(minutes=window_minutes)
    - RateLimitTracker.is_allowed() request cleanup with window_start filtering
    - RateLimitTracker.is_allowed() limit checking and False return when exceeded
    - RateLimitTracker.is_allowed() request tracking with append to client_ip list
    - check_rate_limit() config.api.rate_limit_enabled bypass
    - check_rate_limit() APIRateLimitError raising with client IP and limits
    - check_rate_limit() client IP extraction from request.client.host

  - **Authentication Tests:**
    - verify_api_key() config.api.api_key_enabled bypass return True
    - verify_api_key() missing credentials APIAuthenticationError with "Missing authorization header"
    - verify_api_key() invalid credentials APIAuthenticationError with "Key does not match"
    - verify_api_key() valid credentials return True

  - **Dependency Injection Tests:**
    - get_tracking_manager() lazy import and instance creation
    - get_tracking_manager() TrackingConfig default creation when config.tracking is None
    - get_tracking_manager() TrackingManager.initialize() call when _tracking_active is False
    - set_tracking_manager() global instance assignment
    - get_mqtt_manager() MQTTIntegrationManager creation with config.mqtt

  - **Application Factory Tests:**
    - create_app() FastAPI configuration with title, description, version, debug, docs_url, redoc_url, lifespan
    - create_app() middleware stack order: RequestLoggingMiddleware, SecurityHeadersMiddleware, AuthenticationMiddleware, CORSMiddleware, TrustedHostMiddleware
    - create_app() conditional middleware addition (JWT, CORS) based on config flags
    - create_app() exception handler registration for APIError, OccupancyPredictionError, Exception
    - create_app() request middleware with UUID request_id generation and rate limiting

  - **Error Handler Tests:**
    - api_error_handler() HTTP_400_BAD_REQUEST status code and ErrorResponse JSON structure
    - api_error_handler() comprehensive logging with request_id, error_code, context, traceback
    - system_error_handler() status code mapping based on ErrorSeverity (LOW/MEDIUM → 400, others → 500)
    - system_error_handler() structured logging with severity and exception details
    - general_exception_handler() HTTP_500_INTERNAL_SERVER_ERROR and unhandled exception logging
    - general_exception_handler() exception type extraction and request context logging

  - **Background Task Tests:**
    - background_health_check() health_monitor.start_monitoring() and incident_manager.start_incident_response()
    - background_health_check() periodic health status evaluation and critical/degraded logging
    - background_health_check() incident statistics logging every 5+ minutes
    - background_health_check() graceful shutdown with monitoring system cleanup
    - lifespan() health task creation and cancellation with asyncio.CancelledError handling

  - **APIServer Class Tests:**
    - APIServer.__init__() tracking_manager assignment and config.api initialization
    - APIServer.start() config.enabled check and early return
    - APIServer.start() uvicorn.Config creation with host, port, log_level, access_log
    - APIServer.start() asyncio.create_task(server.serve()) execution
    - APIServer.stop() server.should_exit = True and server_task await
    - APIServer.is_running() server_task existence and done() status check

- Integration Tests:
  - **Health Check Endpoint Integration:**
    - GET /health database connection verification with get_database_manager() and health_check()
    - GET /health tracking manager status with get_tracking_status() and component health analysis
    - GET /health comprehensive health determination with database, tracking, mqtt status aggregation
    - GET /health/comprehensive integration with health_monitor system and component health collection
    - GET /health/components/{component_name} individual component health with 24h history
    - GET /health/system system health summary with health_monitor integration
    - GET /health/monitoring monitoring system statistics and registered check enumeration
    - POST /health/monitoring/start and /stop health monitoring lifecycle management

  - **Prediction Endpoint Integration:**
    - GET /predictions/{room_id} room validation with config.rooms lookup
    - GET /predictions/{room_id} tracking_manager.get_room_prediction() integration
    - GET /predictions/{room_id} datetime parsing with .replace("Z", "+00:00") timezone handling
    - GET /predictions all rooms iteration with individual prediction aggregation
    - GET /predictions partial failure handling with warning logging and continuation

  - **Incident Management Integration:**
    - GET /incidents active incident collection with incident_manager.get_active_incidents()
    - GET /incidents/{incident_id} individual incident retrieval with existence validation
    - GET /incidents/history time window validation (1-168 hours) and history retrieval
    - GET /incidents/statistics incident response system statistics aggregation
    - POST /incidents/{incident_id}/acknowledge incident acknowledgment with user tracking
    - POST /incidents/{incident_id}/resolve manual incident resolution with notes
    - POST /incidents/response/start and /stop incident response system lifecycle

  - **Model Management Integration:**
    - GET /accuracy room_id validation and tracking_manager.get_accuracy_metrics() integration
    - GET /accuracy AccuracyMetricsResponse construction with comprehensive metrics mapping
    - POST /model/retrain ManualRetrainRequest validation and tracking_manager.trigger_manual_retrain()
    - POST /model/retrain background task execution and response structure creation

  - **MQTT and System Integration:**
    - POST /mqtt/refresh mqtt_manager.cleanup_discovery() and initialize() sequence
    - GET /stats comprehensive system statistics with database, mqtt, tracking component aggregation
    - GET /stats SystemStatsResponse construction with version, uptime, prediction metrics

- Edge Cases:
  - **Authentication Edge Cases:**
    - Missing HTTP Authorization header handling
    - Malformed Bearer token format
    - API key with special characters or encoding issues
    - Concurrent authentication requests with rate limiting

  - **Rate Limiting Edge Cases:**
    - Simultaneous requests from same IP at rate limit boundary
    - Clock drift affecting window calculation
    - Memory cleanup with expired request timestamps
    - Rate limiter state persistence across server restarts

  - **Health Check Edge Cases:**
    - Database connection timeout during health check
    - Tracking manager initialization failure states
    - MQTT broker disconnection during health evaluation
    - Health monitor system failure fallback responses
    - Component health check timeouts and partial failures

  - **Prediction Edge Cases:**
    - Room configuration changes during prediction requests
    - Tracking manager unavailable or uninitialized state
    - Prediction data with malformed datetime strings
    - Empty or null prediction responses from tracking manager
    - Concurrent prediction requests for same room

  - **Error Handling Edge Cases:**
    - Exception handler recursion prevention
    - Request ID generation failure scenarios
    - Logging system failure during error handling
    - Memory exhaustion during large request processing
    - Network interruption during async operations

  - **Background Task Edge Cases:**
    - Health check task cancellation during startup/shutdown
    - Monitoring system initialization race conditions
    - Incident response system failure recovery
    - Task cleanup during unexpected server termination
    - Resource cleanup order dependency issues

  - **Integration Edge Cases:**
    - Config reload during active API requests
    - Database transaction failures during endpoint execution
    - MQTT reconnection during discovery refresh
    - Tracking manager state changes during API calls
    - Circular dependency resolution in service initialization

- Error Handling:
  - **API Exception Classes:**
    - APIError base class with error_code, message, context parameters
    - APIAuthenticationError with specific authentication failure messages
    - APIRateLimitError with client_ip, limit, window context
    - APIResourceNotFoundError with resource_type and resource_id
    - APIServerError with endpoint, operation, cause exception chaining

  - **Request Processing Errors:**
    - Malformed JSON request body handling
    - Invalid query parameter types and ranges
    - Missing required headers or authentication
    - Request timeout during long operations
    - Request size limits and overflow handling

  - **Service Integration Errors:**
    - TrackingManager initialization failures with fallback responses
    - Database connection pool exhaustion with queue timeout
    - MQTT broker disconnection with retry mechanisms
    - Health monitor system crashes with basic health fallback
    - Incident manager failures with degraded functionality

  - **Async Operation Errors:**
    - asyncio.CancelledError proper handling in background tasks
    - Task cleanup during server shutdown sequence
    - Concurrent request handling with shared resource conflicts
    - Async generator exhaustion in streaming responses
    - Event loop closure during active operations

  - **Validation and Serialization Errors:**
    - Pydantic model validation failures with detailed error messages
    - Datetime serialization edge cases (timezone, format variations)
    - Response model construction with missing or invalid fields
    - JSON serialization failures with custom object types
    - Request/response size limits with truncation handling

- Coverage Target: 85%+

### src/integration/auth/auth_models.py - Authentication Models
**Classes Found:** AuthUser, LoginRequest, LoginResponse, RefreshRequest, RefreshResponse, LogoutRequest, TokenInfo, PasswordChangeRequest, UserCreateRequest, APIKey
**Methods Analyzed:** AuthUser.validate_permissions(), AuthUser.validate_roles(), AuthUser.has_permission(), AuthUser.has_role(), AuthUser.to_token_claims(), LoginRequest.validate_username(), LoginRequest.validate_password(), RefreshRequest.validate_refresh_token(), PasswordChangeRequest.passwords_match(), PasswordChangeRequest.validate_new_password(), UserCreateRequest.validate_email(), UserCreateRequest.validate_username(), APIKey.is_expired(), APIKey.has_permission()

**Required Tests:**
- Unit Tests:
  - AuthUser model creation with valid and invalid field combinations
  - Permission validation with valid permissions (read, write, admin, model_retrain, system_config, prediction_view, accuracy_view, health_check)
  - Permission validation with invalid permissions should raise ValueError
  - Role validation with valid roles (user, admin, operator, viewer) and invalid roles
  - has_permission() method with direct permissions and admin override behavior
  - has_role() method with various role combinations
  - to_token_claims() method with and without optional fields (email, last_login)
  - LoginRequest username validation (alphanumeric, special chars _-., normalization to lowercase)
  - LoginRequest password validation (minimum 8 chars, complexity requirements - 3 of 4: upper, lower, digit, special)
  - LoginResponse model with required fields and schema example validation
  - RefreshRequest token validation (minimum length, JWT format with 3 parts)
  - RefreshResponse model field validation
  - LogoutRequest with optional refresh_token and revoke_all_tokens flag
  - TokenInfo model with datetime fields and boolean flags
  - PasswordChangeRequest password matching validation using Pydantic V2 ValidationInfo
  - PasswordChangeRequest new password complexity validation
  - UserCreateRequest email validation using regex pattern
  - UserCreateRequest username validation and normalization
  - APIKey model with datetime fields and usage tracking
  - APIKey.is_expired() method with timezone-aware datetime comparisons
  - APIKey.has_permission() method for service-to-service auth

- Integration Tests:
  - Full authentication flow with LoginRequest -> LoginResponse validation
  - Token refresh flow with RefreshRequest -> RefreshResponse validation
  - Password change flow with current/new password validation
  - User creation flow with admin permissions validation
  - API key authentication with permission checking
  - Cross-model validation between AuthUser and TokenInfo consistency

- Edge Cases:
  - Empty permission and role lists vs populated lists
  - Timezone handling in datetime fields (UTC vs local time)
  - Maximum field length validation (username 50 chars, password 128 chars)
  - Special characters in usernames at boundary conditions
  - Password complexity edge cases (exactly 3 of 4 requirements)
  - JWT token with malformed parts (less than 3, empty parts)
  - Email validation edge cases (multiple @, missing TLD, unicode chars)
  - API key expiration at exact boundary timestamp
  - Pydantic V2 ValidationInfo handling in password confirmation

- Error Handling:
  - ValueError exceptions for invalid permissions and roles with specific messages
  - ValueError for password complexity failures with descriptive errors
  - ValueError for username format violations
  - ValueError for email format violations  
  - ValueError for JWT token format validation errors
  - ValueError for password mismatch in PasswordChangeRequest
  - Model validation errors for required fields
  - Field validator exceptions with proper error context

- Coverage Target: 85%+
### src/integration/auth/exceptions.py - Auth Exceptions
**Classes Found:** AuthenticationError, AuthorizationError, TokenExpiredError, TokenInvalidError, TokenRevokedError, InsufficientPermissionsError, AccountDisabledError, InvalidCredentialsError, RateLimitExceededError, SecurityViolationError
**Methods Analyzed:** __init__ methods for all exception classes with various parameter combinations

**Required Tests:**
- Unit Tests: 
  - AuthenticationError creation with default message, custom message, reason and context parameters
  - AuthorizationError creation with default message, required_permission and user_permissions
  - TokenExpiredError creation with default and custom expired_at timestamp
  - TokenInvalidError creation with validation_error details
  - TokenRevokedError creation with revoked_at timestamp
  - InsufficientPermissionsError creation with required_permission formatting
  - AccountDisabledError creation with/without username parameter
  - InvalidCredentialsError creation with default/custom messages
  - RateLimitExceededError creation with limit and window_seconds parameters
  - SecurityViolationError creation with violation_type and auto-generated messages
  - Context parameter handling and merging for all exception types
  - Inheritance chain validation (AuthenticationError <- APIError, etc.)
  - Error code assignment verification (AUTHENTICATION_FAILED, AUTHORIZATION_FAILED, etc.)
  - Error severity level verification (HIGH, MEDIUM) for different exception types

- Integration Tests:
  - Exception serialization to JSON for API responses
  - Exception logging integration with structured context data
  - Exception handling in FastAPI error handlers
  - Exception propagation through authentication middleware
  - Exception context preservation through async call stacks
  - Multiple exception chaining scenarios (e.g., TokenExpiredError -> AuthenticationError)

- Edge Cases:
  - Empty string parameters for optional fields
  - None values for all optional parameters
  - Large context dictionaries with nested objects
  - Unicode characters in error messages and context
  - Very long permission names and validation error strings
  - Circular references in context dictionaries
  - Invalid datetime strings for expired_at/revoked_at fields
  - Empty lists for user_permissions parameter
  - Special characters in violation_type parameter

- Error Handling:
  - Invalid parameter types passed to constructors
  - Context dictionary with non-serializable objects
  - Memory constraints with large error context data
  - Exception creation during low-memory conditions
  - Thread safety when creating exceptions concurrently
  - Exception creation with corrupt or malformed input data
  - Inheritance behavior when parent class initialization fails

- Coverage Target: 85%+

### src/integration/auth/endpoints.py - Auth Endpoints
**Classes Found:** No classes (module with functions and router)
**Methods Analyzed:** verify_password, hash_password, get_user_by_username, login, refresh_token, logout, get_current_user_info, change_password, get_token_info, list_users, create_user, delete_user

**Required Tests:**
- Unit Tests: [specific test scenarios for 85%+ coverage]
  - **Password Functions:**
    - `verify_password()` with correct password hash matches
    - `verify_password()` with incorrect password returns False
    - `hash_password()` generates consistent SHA256 hashes
    - `hash_password()` produces different hashes for different passwords
    - `get_user_by_username()` returns correct user data for valid username
    - `get_user_by_username()` returns None for non-existent user
    - `get_user_by_username()` is case-insensitive for usernames

  - **Login Endpoint (/login):**
    - Successful login with valid admin credentials returns LoginResponse
    - Successful login with valid operator credentials returns LoginResponse
    - Successful login with valid viewer credentials returns LoginResponse
    - Failed login with non-existent username returns 401
    - Failed login with incorrect password returns 401
    - Failed login with inactive user account returns 401
    - Login with remember_me=True extends refresh token to 90 days
    - Login with remember_me=False uses default refresh token expiration
    - Login updates user's last_login timestamp in USER_STORE
    - Login logs successful authentication with client IP and user agent
    - Login logs failed authentication attempts with warning level
    - Login handles JWT token generation errors gracefully
    - Login handles general exceptions with 500 error response

  - **Refresh Token Endpoint (/refresh):**
    - Successful token refresh with valid refresh token returns RefreshResponse
    - Failed token refresh with invalid refresh token returns 401
    - Failed token refresh with expired refresh token returns 401
    - Token refresh handles JWT manager exceptions properly
    - Token refresh logs successful operations
    - Token refresh handles general exceptions with 500 error response

  - **Logout Endpoint (/logout):**
    - Successful logout with refresh token revokes token and returns success
    - Logout without refresh token still returns success message
    - Logout logs user logout activity
    - Logout counts revoked tokens correctly in response
    - Logout handles token revocation failures gracefully
    - Logout handles general exceptions with 500 error response

  - **User Info Endpoint (/me):**
    - Returns authenticated user information correctly
    - Requires valid authentication (dependency injection test)

  - **Change Password Endpoint (/change-password):**
    - Successful password change with correct current password
    - Failed password change with incorrect current password returns 400
    - Password change for non-existent user returns 404
    - Password change updates USER_STORE with new hash
    - Password change logs successful operations
    - Password change handles general exceptions with 500 error response

  - **Token Info Endpoint (/token/info):**
    - Returns TokenInfo for valid JWT token
    - Returns 401 for missing authorization credentials
    - Returns 400 for invalid token format
    - Handles JWT manager token info errors properly
    - Maps token info fields correctly to TokenInfo model
    - Handles general exceptions with 500 error response

  - **Admin List Users Endpoint (/users GET):**
    - Returns list of all users for admin user
    - Requires admin role (dependency injection test)
    - Converts USER_STORE data to AuthUser models correctly
    - Returns empty list if USER_STORE is empty

  - **Admin Create User Endpoint (/users POST):**
    - Successfully creates new user with valid data
    - Returns 400 for duplicate username
    - Generates unique user_id using secrets.token_hex
    - Hashes password before storing in USER_STORE
    - Sets default is_active=True for new users
    - Sets creation timestamp correctly
    - Requires admin role (dependency injection test)
    - Logs successful user creation with admin info
    - Handles general exceptions with 500 error response

  - **Admin Delete User Endpoint (/users/{username} DELETE):**
    - Successfully deletes existing user
    - Returns 400 when admin tries to delete themselves
    - Returns 404 for non-existent user
    - Removes user from USER_STORE completely
    - Handles case-insensitive username matching
    - Requires admin role (dependency injection test)
    - Logs successful user deletion with admin info
    - Handles general exceptions with 500 error response

- Integration Tests: [specific integration scenarios]
  - **Full Authentication Flow:**
    - Complete login-use API-logout cycle with token validation
    - Token refresh before expiration maintains session continuity
    - Token refresh after expiration requires re-authentication
    - Multiple concurrent logins for same user handle properly
    - Session invalidation after password change

  - **Role-Based Access Control:**
    - Admin users can access all admin endpoints
    - Operator users cannot access admin endpoints
    - Viewer users have read-only access only
    - Permission verification across different endpoint types

  - **JWT Manager Integration:**
    - Token generation with correct claims and expiration
    - Token validation with proper error handling
    - Token blacklisting and revocation functionality
    - Remember me functionality with extended expiration

  - **Dependency Injection:**
    - get_current_user dependency provides correct user context
    - require_admin dependency blocks non-admin users
    - get_jwt_manager dependency provides configured JWT manager
    - security_scheme dependency extracts bearer tokens correctly

  - **USER_STORE Operations:**
    - Thread-safe access to in-memory user store
    - Password hash storage and verification consistency
    - User data structure integrity across operations
    - Default user accounts (admin, operator, viewer) availability

- Edge Cases: [specific edge cases to test]
  - **Password Security Edge Cases:**
    - Empty password string handling
    - Very long password strings (>1000 characters)
    - Password with special characters and Unicode
    - Password hash collision scenarios (theoretical)

  - **User Management Edge Cases:**
    - Username with special characters and spaces
    - Email validation and format edge cases
    - User creation with empty or duplicate permissions
    - User deletion during active session
    - Case sensitivity in username lookups

  - **Token Handling Edge Cases:**
    - Malformed JWT tokens in authorization header
    - JWT tokens with missing required claims
    - JWT tokens with future issued_at timestamps
    - Extremely short or long token expiration times
    - Token refresh with revoked refresh token

  - **Authentication Flow Edge Cases:**
    - Login attempts during system maintenance mode
    - Concurrent login attempts for same user
    - Remember me with system clock changes
    - Authentication with missing or corrupted USER_STORE
    - Token introspection with partially invalid tokens

  - **Request/Response Edge Cases:**
    - Request bodies with missing required fields
    - Request bodies with invalid JSON format
    - Very large request payloads exceeding limits
    - Response serialization with datetime timezone issues
    - HTTP header injection in user agent strings

- Error Handling: [specific error conditions]
  - **Authentication Errors:**
    - APIAuthenticationError from JWT manager properly converted to HTTP 401
    - Invalid credentials produce consistent error messages
    - Account lockout scenarios (if implemented)
    - Token expiration handling with proper error codes

  - **Authorization Errors:**
    - Non-admin users accessing admin endpoints return 403
    - Missing or invalid bearer tokens return 401
    - Insufficient permissions return appropriate error codes
    - Role validation failures with descriptive messages

  - **Data Validation Errors:**
    - Pydantic validation errors for request models
    - Invalid email format in user creation
    - Password complexity validation (if implemented)
    - Username format validation and constraints

  - **System Errors:**
    - USER_STORE corruption or unavailability
    - JWT manager initialization failures
    - Logging system failures during authentication
    - Memory exhaustion with large user stores
    - Secrets module failures during user ID generation

  - **Network and Client Errors:**
    - Missing or malformed authorization headers
    - Client IP extraction failures
    - User agent parsing edge cases
    - Request timeout during authentication operations
    - Concurrent request handling with shared resources

- Coverage Target: 85%+

### src/integration/auth/dependencies.py - Auth Dependencies
**Classes Found:** HTTPBearer (imported), AuthUser (imported), JWTManager (imported)
**Methods Analyzed:** get_jwt_manager(), get_current_user(), get_optional_user(), require_permission(), require_role(), require_admin(), require_permissions(), validate_api_key(), get_request_context()

**Required Tests:**

- **Unit Tests:**
  - `get_jwt_manager()` function:
    - Test singleton pattern with global _jwt_manager instance
    - Test JWT manager creation when config.api.jwt.enabled is True
    - Test HTTPException when JWT is disabled (status 500)
    - Test config loading and JWTManager instantiation
    - Test multiple calls return same instance

  - `get_current_user()` function:
    - Test successful authentication with valid Bearer token
    - Test user extraction from request.state when already authenticated by middleware
    - Test AuthUser creation with all token payload fields (sub, username, email, permissions, roles, is_admin)
    - Test authentication with missing credentials (401 status)
    - Test authentication with invalid token (APIAuthenticationError → 401)
    - Test general exception handling (500 status)
    - Test WWW-Authenticate header inclusion in error responses
    - Test logging of warning and error messages

  - `get_optional_user()` function:
    - Test successful user retrieval when authenticated
    - Test None return when authentication fails (HTTPException caught)
    - Test passthrough of request and credentials parameters

  - `require_permission()` dependency factory:
    - Test permission checker creation with specific permission string
    - Test successful access with user having required permission
    - Test access denial with 403 status when permission missing
    - Test admin bypass logic (if implemented)
    - Test logging of permission denial warnings
    - Test proper dependency chain with get_current_user

  - `require_role()` dependency factory:
    - Test role checker creation with specific role string
    - Test successful access with user having required role
    - Test access denial with 403 status when role missing
    - Test logging of role access denial warnings
    - Test dependency chain with get_current_user

  - `require_admin()` dependency:
    - Test successful admin access when user.is_admin is True
    - Test access denial with 403 status when user.is_admin is False
    - Test logging of admin access denial warnings
    - Test dependency chain with get_current_user

  - `require_permissions()` dependency factory:
    - Test with require_all=True (user must have ALL permissions)
    - Test with require_all=False (user needs ANY permission)
    - Test admin bypass when user.is_admin is True
    - Test missing permissions calculation and error messages
    - Test permission intersection logic
    - Test empty permissions list edge case
    - Test logging of insufficient permissions warnings

  - `validate_api_key()` function:
    - Test with API key validation disabled (returns True)
    - Test with API key validation enabled and valid key
    - Test with invalid API key (returns False)
    - Test API key extraction from X-API-Key header
    - Test API key extraction from query parameter in debug mode
    - Test no API key provided (returns False)
    - Test debug mode vs production mode behavior

  - `get_request_context()` function:
    - Test context extraction with all available request attributes
    - Test with missing request.state attributes (graceful handling)
    - Test user context addition when request.state.user exists
    - Test unknown client IP handling
    - Test missing User-Agent header handling
    - Test request_id and timestamp extraction

- **Integration Tests:**
  - FastAPI dependency injection integration:
    - Test complete authentication flow in FastAPI endpoints
    - Test dependency chaining (get_current_user → require_permission)
    - Test HTTP Bearer security scheme integration
    - Test middleware interaction with get_current_user

  - JWT Manager integration:
    - Test token validation through JWTManager
    - Test configuration loading and JWT manager setup
    - Test error propagation from JWT validation

  - AuthUser model integration:
    - Test AuthUser creation from token payload
    - Test has_permission() and has_role() method calls
    - Test user attribute population and defaults

- **Edge Cases:**
  - Authentication edge cases:
    - Empty token string in Bearer credentials
    - Malformed Authorization header format
    - Token with missing 'sub' field (required for user_id)
    - Token with unexpected payload structure
    - Concurrent authentication requests with singleton JWT manager

  - Permission/Role edge cases:
    - Empty permission/role strings
    - Case sensitivity in permission/role matching
    - Special characters in permission/role names
    - Very long permission/role lists
    - Circular dependency scenarios in nested dependency factories

  - API Key validation edge cases:
    - API key with special characters or encoding issues
    - Extremely long API key strings
    - API key in both header and query parameter (precedence)
    - Missing request object in validate_api_key

  - Request context edge cases:
    - Request with no client information (client is None)
    - Request with malformed URL or headers
    - Request state with unexpected attribute types
    - Very large User-Agent strings or headers

- **Error Handling:**
  - JWT Manager errors:
    - JWTManager instantiation failures
    - Token validation raising unexpected exceptions
    - Configuration loading errors in get_jwt_manager

  - FastAPI HTTPException scenarios:
    - Proper status code assignment (401, 403, 500)
    - Correct error message formatting
    - WWW-Authenticate header presence in auth failures
    - Exception details not leaking sensitive information

  - Dependency injection errors:
    - Missing dependencies in dependency chain
    - Circular dependency resolution
    - Async function dependency resolution failures

  - Logging and monitoring:
    - Log message formatting and content validation
    - Warning vs error log level usage
    - Security event logging for failed authentications
    - Performance monitoring for authentication overhead

  - State management errors:
    - Global JWT manager state corruption
    - Thread safety in singleton pattern
    - Memory leaks in global state management
    - Race conditions in concurrent requests

- **Coverage Target: 85%+**


### src/integration/realtime_api_endpoints.py - Real-time API Endpoints
**Classes Found:** WebSocketSubscription, RealtimeStatsResponse, WebSocketConnectionHandler
**Methods Analyzed:** set_integration_manager, get_integration_manager, websocket_predictions_endpoint, websocket_room_predictions_endpoint, sse_predictions_endpoint, sse_room_predictions_endpoint, get_realtime_stats, get_realtime_connections, cleanup_stale_connections, test_realtime_broadcast, realtime_health_check, _handle_client_websocket_message, get_available_channels

**Required Tests:**
- Unit Tests:
  - **Global Manager Functions:**
    - `set_integration_manager()` - Set global integration manager state
    - `get_integration_manager()` - Return manager or raise HTTPException(503)
    - Manager availability error handling with proper HTTP status codes
  
  - **WebSocketSubscription Model:**
    - Valid action values ('subscribe', 'unsubscribe')
    - Required room_id field validation
    - Optional client_metadata field handling
    - Pydantic validation error scenarios
  
  - **RealtimeStatsResponse Model:**
    - All integer fields (websocket_connections, sse_connections, etc.)
    - List field (channels_active) serialization
    - Float field (uptime_seconds) precision handling
    - Model construction from integration manager stats
  
  - **WebSocketConnectionHandler Class:**
    - `connect()` method with and without connection_id
    - `disconnect()` method with valid/invalid connection_ids
    - `send_message()` success and failure scenarios
    - `broadcast_message()` with mixed success/failure clients
    - `get_connection_count()` accuracy with dynamic connections
    - `get_client_session_info()` with valid/invalid client_ids
    - `get_all_sessions()` returns copy of session data
    - `cleanup_stale_connections()` with various idle times
    - Session metadata tracking (connected_at, last_activity, message_count)
    - Connection state management during failures

- Integration Tests:
  - **WebSocket Endpoints:**
    - `/realtime/predictions` - General WebSocket connection flow
    - `/realtime/predictions/{room_id}` - Room-specific connection with auto-subscription
    - WebSocket message handling for subscribe/unsubscribe actions
    - WebSocket ping/pong health checks
    - WebSocket client disconnection cleanup
    - Integration manager WebSocket handler delegation
    - Multiple client connection handling with room subscriptions
  
  - **SSE Endpoints:**
    - `/realtime/events` - Server-Sent Events stream creation
    - `/realtime/events/{room_id}` - Room-specific SSE streams
    - StreamingResponse wrapper handling for SSE streams
    - SSE handler availability and error responses
  
  - **Stats and Monitoring Endpoints:**
    - `/realtime/stats` - Statistics collection from integration manager
    - `/realtime/connections` - Connection detail extraction
    - `/realtime/connections/cleanup` - Stale connection cleanup execution
    - `/health` - Real-time system health status determination
    - `/channels` - Available channel configuration reporting
  
  - **Test and Broadcast Endpoints:**
    - `/broadcast/test` - Test message broadcast via MQTT manager
    - RealtimePredictionEvent creation and broadcasting
    - Test event data structure and publishing

- Edge Cases:
  - **WebSocket Connection Edge Cases:**
    - Client disconnection during subscription confirmation
    - WebSocket state inconsistencies (CONNECTED vs actual state)
    - Malformed JSON message handling from WebSocket clients
    - WebSocket handler unavailable from integration manager
    - Client connection timeout during room subscription setup
    - Race conditions in connection registration and cleanup
  
  - **SSE Stream Edge Cases:**
    - SSE handler returning non-StreamingResponse objects
    - SSE stream creation failure scenarios
    - Request cancellation during SSE stream setup
    - Room-specific SSE handler unavailability
  
  - **Message Processing Edge Cases:**
    - Unknown WebSocket message types from clients
    - Missing room_id in subscription/unsubscription requests
    - WebSocket send failures during response transmission
    - Client metadata corruption in WebSocket manager
    - Concurrent subscription changes for same client
  
  - **Stats Collection Edge Cases:**
    - Integration manager stats unavailability
    - Partial stats data (missing nested dictionaries)
    - API WebSocket handler session corruption
    - Stats calculation during system shutdown
    - Connection count inconsistencies during rapid connect/disconnect
  
  - **Broadcast and Health Edge Cases:**
    - Enhanced MQTT manager unavailability during broadcast
    - Test event creation failure scenarios
    - Health check during integration manager state changes
    - Channel configuration retrieval during system reconfiguration

- Error Handling:
  - **WebSocket Exception Handling:**
    - WebSocketDisconnect exception proper logging and cleanup
    - WebSocket connection state validation before operations
    - JSON encoding/decoding errors in WebSocket message handling
    - WebSocket client_state validation before close operations
    - Exception handling in WebSocket message loops
  
  - **HTTP Exception Scenarios:**
    - HTTPException(503) for unavailable services (integration manager, SSE handler)
    - HTTPException(500) for internal server errors in all endpoints
    - HTTPException(401) for API authentication failures in stats endpoint
    - Proper error message formatting and logging context
  
  - **Integration Manager Error Handling:**
    - TrackingIntegrationManager unavailability scenarios
    - WebSocket/SSE handler retrieval failures
    - Stats collection failures with partial data recovery
    - Enhanced MQTT manager integration errors
  
  - **Connection Management Error Handling:**
    - WebSocket connection failure during send operations
    - Stale connection cleanup during high concurrent load
    - Session data corruption recovery mechanisms
    - Connection ID generation collision handling
    - Memory cleanup for abandoned connections
  
  - **API Key Validation Error Handling:**
    - APIAuthenticationError for invalid API key formats
    - API key length validation (<10 characters)
    - Missing API key handling in stats endpoint
    - APIError propagation to HTTP exceptions

- Coverage Target: 85%+

### src/integration/enhanced_integration_manager.py - Enhanced Integration Manager
**Classes Found:** EnhancedIntegrationManager, EnhancedIntegrationStats, CommandRequest, CommandResponse, EnhancedIntegrationError
**Methods Analyzed:** __init__, initialize, shutdown, update_entity_state, process_command, handle_prediction_update, handle_system_status_update, get_integration_stats, _define_and_publish_entities, _define_and_publish_services, _setup_command_handlers, _start_background_tasks, _command_processing_loop, _entity_monitoring_loop, _check_entity_availability, _cleanup_old_responses, and 13 command handler methods

**Required Tests:**
- Unit Tests:
  - **EnhancedIntegrationManager Initialization:**
    - Constructor with all optional parameters (mqtt_integration_manager, tracking_manager, notification_callbacks)
    - Constructor with minimal parameters (defaults and None values)
    - Configuration loading and validation (config, mqtt_config, rooms, tracking_config)
    - Initial state setup (command_handlers, entity_states, background_tasks, stats)
    - Logger initialization and component initialization logging

  - **System Lifecycle Management:**
    - `initialize()` with valid MQTT integration manager and discovery publisher
    - `initialize()` with uninitialized MQTT integration manager (auto-initialization)
    - `initialize()` without MQTT integration manager (warning scenario)
    - `initialize()` with entity/service definition and publishing failures
    - `shutdown()` with active background tasks cancellation and cleanup
    - `shutdown()` with already completed tasks (no-op scenario)
    - `shutdown()` with exception handling during task cleanup

  - **Entity State Management:**
    - `update_entity_state()` with valid entity_id, state, and attributes
    - `update_entity_state()` with inactive integration system (returns False)
    - `update_entity_state()` with missing entity definition in HA definitions
    - `update_entity_state()` with MQTT publishing success and failure scenarios
    - `update_entity_state()` with state topic validation and payload creation
    - Entity state tracking in local cache (entity_states, last_state_update)

  - **Command Processing System:**
    - `process_command()` with valid command and known handler (sync and async handlers)
    - `process_command()` with unknown command (error response)
    - `process_command()` with command handler exceptions (error handling)
    - `process_command()` with correlation_id tracking and response storage
    - Command statistics tracking (commands_processed, last_command_processed)
    - CommandRequest and CommandResponse dataclass validation

  - **Prediction and System Updates:**
    - `handle_prediction_update()` with complete PredictionResult and room_id
    - `handle_prediction_update()` with inactive integration system (early return)
    - `handle_prediction_update()` with entity state update failures
    - `handle_system_status_update()` with comprehensive system status dictionary
    - `handle_system_status_update()` with partial status mappings and missing keys
    - Status entity mapping validation (status_mappings dictionary)

  - **Statistics and Monitoring:**
    - `get_integration_stats()` with complete statistics compilation
    - `get_integration_stats()` with missing MQTT integration manager
    - `get_integration_stats()` with missing HA entity definitions
    - EnhancedIntegrationStats dataclass field validation and updates
    - Statistics aggregation from multiple sources (enhanced, mqtt, entity definitions)

  - **Private Method Operations:**
    - `_define_and_publish_entities()` with successful entity definition and publishing
    - `_define_and_publish_entities()` with missing HA entity definitions
    - `_define_and_publish_services()` with service definition and button entity creation
    - `_setup_command_handlers()` with all 13 command handlers registration
    - `_start_background_tasks()` with command processing and monitoring task creation

  - **Background Task Management:**
    - `_command_processing_loop()` with command queue processing and timeout handling
    - `_command_processing_loop()` with response publishing and correlation ID handling
    - `_command_processing_loop()` with shutdown event and graceful termination
    - `_entity_monitoring_loop()` with entity availability checking and response cleanup
    - `_entity_monitoring_loop()` with 30-second interval timing and error handling
    - `_check_entity_availability()` with device availability publishing
    - `_cleanup_old_responses()` with 1-hour expiration and cleanup logic

  - **Command Handler Methods (All 13):**
    - `_handle_retrain_model()` with tracking manager delegation and room_id/force parameters
    - `_handle_validate_model()` with model performance validation and days parameter
    - `_handle_restart_system()` with system restart acknowledgment
    - `_handle_refresh_discovery()` with discovery publisher refresh and result counting
    - `_handle_reset_statistics()` with confirmation requirement and stats reset
    - `_handle_generate_diagnostic()` with include_logs/include_metrics parameters
    - `_handle_check_database()` with database health check simulation
    - `_handle_force_prediction()` with room_id requirement and tracking manager delegation
    - `_handle_prediction_enable()` with enabled/disabled state management
    - `_handle_mqtt_enable()` with MQTT publishing control
    - `_handle_set_interval()` with prediction interval configuration
    - `_handle_set_log_level()` with logging level modification

  - **Dataclass Validation:**
    - EnhancedIntegrationStats with all 8 fields and default values
    - CommandRequest with required and optional fields (timestamp, correlation_id)
    - CommandResponse with success/error scenarios and timestamp defaults
    - EnhancedIntegrationError with custom error code and severity handling

- Integration Tests:
  - **MQTT Integration Manager Integration:**
    - Full initialization with real MQTT integration manager and discovery publisher
    - Entity definition and publishing through MQTT discovery system
    - Service definition and button entity creation via MQTT
    - State updates published through MQTT with QoS and retain settings
    - Device availability updates through MQTT discovery publisher

  - **HA Entity Definitions Integration:**
    - HAEntityDefinitions integration with discovery publisher and configuration
    - Entity definition retrieval and state topic validation
    - Entity statistics collection and compilation
    - All entity types creation and publishing validation

  - **Tracking Manager Integration:**
    - Command delegation to tracking manager for model operations
    - Prediction update handling with real PredictionResult objects
    - Model retraining and validation through tracking manager interface
    - Force prediction execution through tracking manager

  - **Background Task Coordination:**
    - Command processing loop with real command queue operations
    - Entity monitoring loop with availability checking and cleanup
    - Task cancellation and cleanup during shutdown
    - Concurrent task execution and resource management

- Edge Cases:
  - **Initialization Edge Cases:**
    - MQTT integration manager without stats or uninitialized state
    - Missing discovery publisher in MQTT integration manager
    - HA entity definitions creation failure scenarios
    - Command handler setup with reflection/method binding issues

  - **Entity State Edge Cases:**
    - Entity state updates with missing state topics
    - MQTT publishing failures during state updates
    - Entity definition not found for given entity_id
    - State payload creation with invalid attributes

  - **Command Processing Edge Cases:**
    - Command handlers returning None or invalid response types
    - Async handler execution with timeout or cancellation
    - Command correlation ID collisions or invalid formats
    - Response topic publishing failures in command processing loop

  - **Background Task Edge Cases:**
    - Command queue empty timeout scenarios (1.0 second timeout)
    - Entity monitoring with 30-second intervals and system shutdown timing
    - Response cleanup with timestamp comparison edge cases (1-hour threshold)
    - Task exception handling and logging without system failure

  - **System Status Edge Cases:**
    - System status updates with missing status_mappings keys
    - Partial system status dictionaries with missing fields
    - Entity state updates for system status with invalid values
    - Status mapping iteration with dynamic key availability

  - **Statistics Collection Edge Cases:**
    - MQTT integration manager stats unavailable or corrupted
    - HA entity definitions stats collection failures
    - Entity states count with concurrent modifications
    - Command handlers count with dynamic handler registration

- Error Handling:
  - **Initialization Error Handling:**
    - Exception handling during entity and service definition/publishing
    - HA entity definitions creation failures with proper error logging
    - Background task startup failures with cleanup and rollback
    - Configuration loading errors and validation failures

  - **Command Processing Error Handling:**
    - Command handler exceptions with error response generation
    - Unknown command handling with warning logging
    - Response publishing failures in command processing loop
    - Correlation ID tracking with invalid or duplicate IDs

  - **Entity Management Error Handling:**
    - Entity state update failures with graceful degradation
    - MQTT publishing errors during state updates
    - Entity availability checking failures with error logging
    - Device availability publishing errors

  - **Background Task Error Handling:**
    - Command processing loop exceptions with sleep/retry logic
    - Entity monitoring loop errors with 5-second error backoff
    - Response cleanup failures with error logging
    - Task cancellation exceptions during shutdown

  - **Integration Error Handling:**
    - MQTT integration manager unavailable scenarios
    - Tracking manager method unavailable or missing
    - Discovery publisher operations failures
    - System component unavailable error responses

  - **Command Handler Error Handling:**
    - Each command handler's exception handling and error response format
    - Tracking manager delegation failures with appropriate error messages
    - Parameter validation and requirement checking (room_id, confirmation)
    - Method existence validation using hasattr checks

- Coverage Target: 85%+

### src/integration/discovery_publisher.py - Service Discovery
**Classes Found:** [DiscoveryPublisher, EnhancedDiscoveryError, EntityState (Enum), EntityCategory (Enum), DeviceClass (Enum), EntityAvailability (dataclass), ServiceConfig (dataclass), EntityMetadata (dataclass), DeviceInfo (dataclass), SensorConfig (dataclass)]
**Methods Analyzed:** [__init__, publish_all_discovery, publish_room_discovery, publish_system_discovery, remove_discovery, refresh_discovery, get_discovery_stats, publish_device_availability, publish_service_discovery, update_entity_state, cleanup_entities, _validate_published_entities, _publish_service_button, _create_prediction_sensor, _create_next_transition_sensor, _create_confidence_sensor, _create_time_until_sensor, _create_reliability_sensor, _create_system_status_sensor, _create_uptime_sensor, _create_predictions_count_sensor, _create_accuracy_sensor, _create_alerts_sensor, _create_database_status_sensor, _create_tracking_status_sensor, _publish_sensor_discovery]

**Required Tests:**
- Unit Tests:
  - **DiscoveryPublisher Initialization:**
    - Constructor with all required parameters (mqtt_publisher, config, rooms)
    - Optional callback parameter handling (availability_check_callback, state_change_callback)
    - Device info creation with proper identifiers and capabilities
    - Discovery state initialization (discovery_published, published_entities, entity_metadata)
    - Service integration setup (available_services, command_handlers)
    - Statistics initialization with default values
    - Availability topic configuration and device class setup
    - Diagnostic info creation with timestamp and version information
    
  - **Core Discovery Publishing:**
    - publish_all_discovery() with discovery enabled/disabled scenarios
    - Sequential publishing steps: device availability → room discovery → system discovery → services
    - Discovery state tracking and statistics updates after successful publishing
    - Results aggregation from multiple publish operations
    - Entity validation after successful discovery publishing
    - Error handling during multi-step discovery process
    
  - **Room Discovery Methods:**
    - publish_room_discovery() for individual room configurations
    - Multiple sensor creation per room (prediction, transition, confidence, time_until, reliability)
    - Room-specific sensor configuration generation with proper unique IDs
    - Error handling during room sensor discovery publishing
    
  - **System Discovery Methods:**
    - publish_system_discovery() for all system sensors
    - System sensor creation (status, uptime, predictions_count, accuracy, alerts, database, tracking)
    - System-wide sensor configuration with proper device association
    - Statistics tracking for system discovery operations
    
  - **Entity Management:**
    - remove_discovery() with existing and non-existing entities
    - Empty payload publishing for entity removal
    - Published entities cleanup from internal tracking
    - Entity metadata removal during cleanup
    - refresh_discovery() clearing state and republishing all entities
    - cleanup_entities() with selective and full entity cleanup
    
  - **State and Availability Management:**
    - publish_device_availability() with online/offline states
    - Availability payload creation with enhanced metadata
    - Device availability topic validation and error handling
    - update_entity_state() for entity metadata updates
    - State change callback invocation (sync/async callback handling)
    - Entity metadata state updates with timestamps
    
  - **Service Discovery:**
    - publish_service_discovery() for Home Assistant service buttons
    - Service configuration creation (manual_retrain, refresh_discovery, reset_statistics, force_prediction)
    - Service button discovery payload generation
    - Command topic and template configuration
    - Service integration with available_services tracking
    
  - **Statistics and Monitoring:**
    - get_discovery_stats() comprehensive statistics collection
    - Discovery statistics updates during operations
    - Entity metadata statistics (count, status tracking)
    - Device information statistics (capabilities, last_seen)
    - Service statistics (available services, command handlers)
    
  - **Private Sensor Creation Methods:**
    - All _create_*_sensor() methods with proper SensorConfig creation
    - Sensor-specific configurations (value_template, device_class, icons)
    - Room-specific sensor configurations with proper topic structure
    - System sensor configurations with appropriate entity categories
    - Enhanced sensor attributes (expire_after, availability, state_class)
    
  - **Enhanced Discovery Publishing:**
    - _publish_sensor_discovery() with enhanced payload creation
    - Device information enrichment (suggested_area, configuration_url, hw_version)
    - Enhanced sensor attributes inclusion in discovery payload
    - Entity metadata creation after successful discovery
    - Availability configuration for sensors
    
  - **Validation and Error Handling:**
    - _validate_published_entities() metadata creation for successful entities
    - Entity category assignment based on sensor type
    - _publish_service_button() with service configuration validation
    - Button entity discovery payload creation with proper device association

- Integration Tests:
  - **Full Discovery Workflow Integration:**
    - Complete discovery publishing workflow from initialization to validation
    - MQTT publisher integration with actual message publishing
    - Room configuration integration with multiple room types
    - Service discovery integration with Home Assistant command topics
    - Entity lifecycle management from creation to cleanup
    
  - **MQTT Integration Testing:**
    - MQTT publisher publish_json() integration for discovery messages
    - Topic structure validation for discovery messages
    - Retention and QoS settings for discovery messages
    - MQTT publish result handling and error propagation
    - Availability topic publishing with proper payload structure
    
  - **Configuration Integration:**
    - MQTTConfig integration with discovery prefix and device identifiers
    - RoomConfig integration for room-specific sensor creation
    - Device identifier consistency across all discovery messages
    - Topic prefix application across all published topics
    
  - **Callback Integration Testing:**
    - Availability check callback invocation during device availability updates
    - State change callback invocation with proper parameters
    - Async vs sync callback handling
    - Callback error handling and logging
    
  - **Service Button Integration:**
    - Service button discovery with command topic configuration
    - Button payload formatting for Home Assistant integration
    - Service configuration storage and retrieval
    - Command handler registration and integration

- Edge Cases:
  - **Discovery Configuration Edge Cases:**
    - Discovery disabled scenario (config.discovery_enabled = False)
    - Missing device identifier in configuration
    - Invalid topic prefix configurations
    - Empty rooms dictionary handling
    - Missing availability topic configuration
    
  - **Entity State Edge Cases:**
    - Entity state updates for non-existent entities
    - Concurrent entity state updates
    - Invalid entity state values
    - Missing entity metadata during state updates
    - Entity state updates during discovery cleanup
    
  - **Discovery Message Edge Cases:**
    - Large discovery payload handling (sensor with many attributes)
    - Discovery payload JSON serialization failures
    - Topic name length limits and validation
    - Duplicate unique ID handling across sensors
    - Missing required sensor configuration attributes
    
  - **Service Discovery Edge Cases:**
    - Service discovery with empty services list
    - Service button creation with missing required fields
    - Command template validation failures
    - Service configuration conflicts (duplicate service names)
    - Service discovery during device unavailability
    
  - **Availability and Device Edge Cases:**
    - Device availability publishing without availability topic
    - Device availability during MQTT connection failures
    - Device capabilities serialization with complex data types
    - Device diagnostic info with missing version information
    - Last seen timestamp handling with timezone issues
    
  - **Cleanup and Lifecycle Edge Cases:**
    - Entity cleanup during active discovery operations
    - Cleanup of partially published entities
    - Metadata cleanup race conditions
    - Discovery refresh during ongoing MQTT operations
    - Entity removal with missing discovery topics

- Error Handling:
  - **MQTT Publishing Error Handling:**
    - MQTTPublishResult error propagation from failed publishes
    - MQTT connection failures during discovery publishing
    - MQTT timeout handling during large discovery operations
    - Partial publishing failure recovery (some entities succeed, others fail)
    - MQTT broker unavailability during discovery operations
    
  - **Configuration Validation Error Handling:**
    - Invalid MQTT configuration handling
    - Missing required configuration fields
    - Room configuration validation errors
    - Device information validation failures
    - Service configuration validation errors
    
  - **Entity Management Error Handling:**
    - Entity metadata corruption recovery
    - Entity state update failures with proper logging
    - Entity cleanup failures with partial success handling
    - Published entities dictionary corruption recovery
    - Discovery topic generation failures
    
  - **JSON and Payload Error Handling:**
    - JSON serialization failures for complex discovery payloads
    - Invalid template syntax in sensor configurations
    - Payload size limit handling for large discovery messages
    - Character encoding issues in discovery payloads
    - Malformed sensor configuration attribute handling
    
  - **Callback Error Handling:**
    - Exception handling in availability check callbacks
    - State change callback failures without affecting main operation
    - Async callback timeout handling
    - Callback parameter validation failures
    - Missing callback method handling
    
  - **Statistics and Monitoring Error Handling:**
    - Statistics collection failures during concurrent operations
    - Discovery statistics corruption recovery
    - Entity metadata statistics calculation errors
    - Device information statistics with missing fields
    - Service statistics tracking failures

- Coverage Target: 85%+

### src/integration/auth/middleware.py - Auth Middleware
**Classes Found:** SecurityHeadersMiddleware, AuthenticationMiddleware, RequestLoggingMiddleware
**Methods Analyzed:** dispatch, _authenticate_request, _is_public_endpoint, _is_admin_endpoint, _get_client_ip, _check_rate_limit

**Required Tests:**
- Unit Tests:
  - **SecurityHeadersMiddleware Tests:**
    - Verify all security headers are added correctly (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Content-Security-Policy, Strict-Transport-Security, Referrer-Policy, Permissions-Policy)
    - Test removal of information disclosure headers (Server, X-Powered-By, X-AspNet-Version, X-AspNetMvc-Version)
    - Test X-Request-ID header addition when request.state.request_id exists
    - Test initialization with debug flag variations
    - Verify header values match security best practices

  - **AuthenticationMiddleware Tests:**
    - Test JWT authentication with valid tokens (access token validation)
    - Test API key authentication fallback when JWT fails
    - Test initialization with JWT enabled/disabled configurations
    - Test user context setting (request.state.user assignment)
    - Test admin endpoint access control (is_admin validation)
    - Test request ID generation and assignment
    - Test public endpoint bypass (exact matches and pattern matching)
    - Test test mode endpoint patterns in test environment
    - Rate limiting validation with proper time window calculations
    - Rate limiting request count tracking per client IP
    - Client IP extraction from various headers (X-Forwarded-For, X-Real-IP, fallback)
    - JWT token parsing from Authorization header (Bearer scheme validation)
    - AuthUser object creation from JWT payload and API key scenarios
    - Error handling for missing JWT manager configuration

  - **RequestLoggingMiddleware Tests:**
    - Test request logging with all metadata fields (request_id, method, path, query, client_ip, user_agent, content_type, content_length)
    - Test response logging with processing time calculation
    - Test X-Processing-Time header addition
    - Test user_id extraction from request.state.user
    - Test log_body flag initialization and usage
    - Test anonymous user logging when no user context exists
    - Test request ID fallback generation when not present

  - **Helper Method Tests:**
    - _is_public_endpoint: exact matches, pattern matching, test mode patterns
    - _is_admin_endpoint: admin pattern matching for various admin paths
    - _get_client_ip: X-Forwarded-For parsing, X-Real-IP extraction, fallback to request.client.host
    - _check_rate_limit: window calculations, request count tracking, rate limit disabled scenarios

- Integration Tests:
  - **End-to-End Middleware Chain Testing:**
    - Test complete middleware stack execution order
    - Test middleware interaction with FastAPI endpoints
    - Test security headers with authentication flow
    - Test request logging with authentication success/failure
    - Test rate limiting with actual HTTP requests
    - Test admin endpoint protection with real JWT tokens

  - **JWT Integration Testing:**
    - Test JWT validation with real JWTManager instance
    - Test token expiration handling
    - Test token refresh scenarios
    - Test JWT payload extraction and user creation
    - Test JWT authentication with different roles and permissions

  - **Database and Config Integration:**
    - Test configuration loading for JWT, API key, and rate limiting settings
    - Test middleware behavior with different config.api settings
    - Test test environment detection and endpoint modifications

- Edge Cases:
  - **Authentication Edge Cases:**
    - Malformed Authorization headers (missing Bearer, invalid format)
    - Empty or whitespace-only tokens
    - JWT token with missing required claims (sub, permissions, roles)
    - API key authentication when JWT is disabled
    - Invalid API key format or value
    - Missing Authorization header entirely
    - Token validation exceptions and fallback handling
    - User creation with partial JWT payload data

  - **Rate Limiting Edge Cases:**
    - Rate limiting window boundary calculations
    - Concurrent request handling for same client IP
    - Rate limit disabled but request counting still occurs
    - Memory cleanup of old request timestamps
    - Client IP extraction failures (no client, unknown IP)
    - Rate limit exceeded exactly at threshold

  - **Security Headers Edge Cases:**
    - Response headers modification after other middleware
    - Header removal when headers don't exist
    - Header overwriting scenarios
    - Request ID missing from request state
    - Debug mode variations in header behavior

  - **Request Logging Edge Cases:**
    - Request logging with missing user agent
    - Response logging with missing content length
    - Processing time calculation edge cases
    - Log body flag behavior with different request types
    - User ID extraction when user object format is unexpected

- Error Handling:
  - **Authentication Error Scenarios:**
    - APIAuthenticationError with proper error codes and messages
    - APISecurityError for security violations
    - JWT validation exceptions and proper JSON responses
    - JWT manager not configured (500 error response)
    - Admin access denied (403 error response)
    - Rate limit exceeded (429 error response)
    - Generic middleware exceptions (500 error with request ID)

  - **Configuration Error Handling:**
    - Missing JWT configuration when JWT is enabled
    - Invalid API key configuration
    - Rate limiting configuration validation
    - Environment variable detection failures

  - **Network and Request Error Handling:**
    - Missing request.client scenarios
    - Invalid request headers parsing
    - Request state attribute access errors
    - Response header manipulation failures

  - **Logging Error Scenarios:**
    - Logger configuration issues
    - Structured logging format validation
    - Log level configuration testing
    - Exception logging with proper context

- Coverage Target: 85%+

### src/integration/auth/jwt_manager.py - JWT Token Management
**Classes Found:** TokenBlacklist, JWTManager
**Methods Analyzed:** TokenBlacklist.__init__, add_token, is_blacklisted, _cleanup_expired; JWTManager.__init__, generate_access_token, generate_refresh_token, validate_token, refresh_access_token, revoke_token, get_token_info, _create_jwt_token, _decode_jwt_token, _create_signature, _base64url_encode, _base64url_decode, _check_rate_limit

**Required Tests:**
- Unit Tests: 
  - **TokenBlacklist Class:**
    - `__init__()` - Initialize blacklisted token sets and cleanup timing
    - `add_token()` - Token and JTI addition with optional JTI parameter
    - `add_token()` - Logging verification for blacklisted tokens
    - `is_blacklisted()` - Token blacklist checking with cleanup trigger
    - `is_blacklisted()` - JTI-based blacklist checking
    - `is_blacklisted()` - Return false for non-blacklisted tokens
    - `_cleanup_expired()` - Cleanup interval timing validation
    - `_cleanup_expired()` - Last cleanup timestamp updates

  - **JWTManager Class:**
    - `__init__()` - Configuration validation with proper JWTConfig
    - `__init__()` - Secret key validation (presence and minimum length)
    - `__init__()` - Blacklist initialization based on config flag
    - `__init__()` - Rate limiting initialization with default values
    - `generate_access_token()` - Token structure with all standard claims (sub, iat, exp, nbf, iss, aud, jti, type)
    - `generate_access_token()` - Permissions embedding in token payload
    - `generate_access_token()` - Additional claims handling without reserved claim overrides
    - `generate_access_token()` - Rate limiting enforcement per user
    - `generate_access_token()` - UUID JTI generation for uniqueness
    - `generate_refresh_token()` - Refresh token structure with minimal claims
    - `generate_refresh_token()` - Longer expiration time for refresh tokens
    - `generate_refresh_token()` - Rate limiting for refresh token generation
    - `validate_token()` - Token signature verification with HMAC-SHA256
    - `validate_token()` - Token type validation (access vs refresh)
    - `validate_token()` - Expiration time validation with current timestamp
    - `validate_token()` - Not-before time validation
    - `validate_token()` - Issuer and audience claim validation
    - `validate_token()` - Required claim presence validation
    - `validate_token()` - Blacklist checking integration
    - `validate_token()` - Exception handling and re-raising
    - `refresh_access_token()` - Refresh token validation before new token generation
    - `refresh_access_token()` - Old refresh token blacklisting
    - `refresh_access_token()` - New access and refresh token pair generation
    - `refresh_access_token()` - User permission retrieval and embedding
    - `revoke_token()` - Token blacklisting with JTI extraction
    - `revoke_token()` - Blacklist disabled scenario handling
    - `revoke_token()` - Token decoding failure handling
    - `get_token_info()` - Token information extraction without expiration validation
    - `get_token_info()` - Expiration status calculation
    - `get_token_info()` - Blacklist status integration
    - `get_token_info()` - Error handling for invalid tokens
    - `_create_jwt_token()` - JWT header creation with algorithm and type
    - `_create_jwt_token()` - Base64URL encoding of header and payload
    - `_create_jwt_token()` - HMAC signature creation and attachment
    - `_decode_jwt_token()` - Token part splitting validation (3 parts)
    - `_decode_jwt_token()` - Signature verification with constant-time comparison
    - `_decode_jwt_token()` - Header and payload JSON decoding
    - `_decode_jwt_token()` - Algorithm validation against configuration
    - `_decode_jwt_token()` - Expiration verification toggle support
    - `_create_signature()` - HMAC-SHA256 signature generation with secret key
    - `_base64url_encode()` - String and bytes input handling
    - `_base64url_encode()` - Padding removal for base64url compliance
    - `_base64url_decode()` - Padding restoration for decoding
    - `_base64url_decode()` - UTF-8 string decoding from bytes
    - `_check_rate_limit()` - Rate limiting window cleanup (60 seconds)
    - `_check_rate_limit()` - Operation count validation against limit (30/minute)
    - `_check_rate_limit()` - User operation tracking and timestamp recording

- Integration Tests:
  - **Full JWT Lifecycle Integration:**
    - Generate access token → validate token → extract user info flow
    - Generate refresh token → use for access token refresh → validate new tokens
    - Token generation → revocation → validation failure sequence
    - Rate limiting across multiple rapid token generation attempts
    - Blacklist integration with token validation and revocation

  - **Security Validation Integration:**
    - Token tampering detection (modified payload, signature, header)
    - Expired token rejection across all validation methods
    - Cross-algorithm attack prevention (JWT algorithm confusion)
    - Token replay attack prevention via blacklisting
    - Signature verification with wrong secret key

  - **Configuration Integration:**
    - Different JWTConfig settings (issuer, audience, expiration times)
    - Blacklist enabled/disabled scenarios across token operations
    - Secret key strength validation during manager initialization
    - Algorithm configuration consistency across token lifecycle

- Edge Cases:
  - **Token Structure Edge Cases:**
    - Malformed JWT tokens (wrong number of parts, invalid base64url encoding)
    - Empty or null token strings in validation
    - Tokens with missing required claims (sub, iat, exp, jti)
    - Tokens with invalid claim types (non-string user_id, non-list permissions)
    - Reserved claim override attempts in additional_claims

  - **Timing and Expiration Edge Cases:**
    - Tokens at exact expiration timestamp boundary
    - Not-before time in the future validation
    - Clock skew scenarios with slightly future timestamps
    - Expired refresh tokens used for token refresh
    - Rate limiting window boundary conditions (59-61 seconds)

  - **Blacklist Management Edge Cases:**
    - Blacklist cleanup during high token volume
    - Memory usage with large numbers of blacklisted tokens
    - Concurrent token blacklisting and validation
    - Blacklist disabled but revoke_token called scenarios
    - Token info retrieval for blacklisted tokens

  - **Rate Limiting Edge Cases:**
    - Exactly 30 operations within 60 seconds boundary testing
    - Rate limiting across different users simultaneously
    - Rate limit reset after window expiration
    - Rate limiting with system clock changes
    - Multiple token types counting toward same user limit

  - **Encoding/Decoding Edge Cases:**
    - Base64URL padding edge cases (different string lengths)
    - Unicode characters in additional claims
    - Very large payloads approaching JWT size limits
    - Special characters in user_id and permissions
    - JSON serialization edge cases with datetime objects

- Error Handling:
  - **APIAuthenticationError Scenarios:**
    - Invalid token format (not 3 parts separated by dots)
    - Invalid signature verification failures
    - Expired token validation attempts
    - Wrong token type validation (access vs refresh)
    - Missing or invalid issuer/audience claims
    - Blacklisted token validation attempts
    - Not-before time validation failures

  - **APISecurityError Scenarios:**
    - Rate limiting exceeded for token generation
    - Multiple rapid token requests within rate limit window
    - Rate limit enforcement error handling

  - **ValueError Scenarios:**
    - Missing JWT secret key during initialization
    - Secret key shorter than 32 characters
    - Invalid JWTConfig parameter validation

  - **Configuration Validation Errors:**
    - Invalid algorithm specification in config
    - Missing issuer or audience configuration
    - Invalid token expiration time configurations
    - Blacklist configuration inconsistencies

  - **JSON and Encoding Errors:**
    - Invalid JSON in token payload or header
    - Base64URL decoding failures for corrupted tokens
    - UTF-8 encoding issues in token components
    - JSON serialization errors for complex additional claims

  - **System and Resource Errors:**
    - Memory constraints during blacklist operations
    - Clock synchronization issues affecting timestamps
    - Cryptographic operation failures in signature creation
    - Token storage and retrieval during high concurrency

- Coverage Target: 85%+

### src/integration/dashboard.py - Dashboard Integration
**Classes Found:** DashboardMode, MetricType, DashboardConfig, SystemOverview, WebSocketManager, PerformanceDashboard, DashboardError
**Methods Analyzed:** __init__, connect, disconnect, send_personal_message, broadcast, get_connection_stats, _create_fastapi_app, _register_routes, start_dashboard, stop_dashboard, _update_loop, _get_system_overview, _get_accuracy_dashboard_data, _get_drift_dashboard_data, _get_retraining_dashboard_data, _get_system_health_data, _get_alerts_dashboard_data, _get_trends_dashboard_data, _get_dashboard_stats, _get_websocket_initial_data, _get_websocket_update_data, _handle_websocket_message, _get_requested_data, _trigger_manual_retraining, _acknowledge_alert, _get_cached_data, _cache_data

**Required Tests:**
- Unit Tests:
  - **DashboardConfig:**
    - Default configuration instantiation with proper values
    - Configuration validation for host/port/timeout values
    - Mode enum validation (DEVELOPMENT, PRODUCTION, READONLY)
    - List field defaults for allowed_origins and feature flags
    - Configuration serialization and field access
  
  - **SystemOverview:**
    - to_dict() method converting all fields to proper dictionary format
    - Default field initialization with correct types (float, int, str, datetime)
    - Health score calculation and status determination logic
    - Timestamp handling and ISO format conversion
    - Metric aggregation from tracking manager status
  
  - **WebSocketManager:**
    - Connection acceptance with max_connections limit enforcement
    - Connection metadata tracking (connected_at, messages_sent, last_message_at)
    - Thread-safe connection management with _lock usage
    - Connection rejection when at capacity (>= max_connections)
    - send_personal_message with JSON serialization and metadata updates
    - broadcast to multiple connections with disconnection cleanup
    - get_connection_stats calculation for active/max/available connections
    - disconnect cleanup removing connections and metadata
    - Connection state validation before send operations
  
  - **PerformanceDashboard:**
    - Initialization with tracking_manager dependency injection
    - FastAPI availability validation and graceful fallback
    - Cache management with TTL expiration and size limits
    - Dashboard state management (_running flag, task lifecycle)
    - FastAPI app creation with middleware and route registration
    - CORS configuration based on config.enable_cors
    - Start/stop lifecycle with background task management
    - Cache key generation and data expiration handling
    - Error handling for missing tracking manager components
  
  - **Data Retrieval Methods:**
    - _get_system_overview cache behavior and tracking manager integration
    - _get_accuracy_dashboard_data filtering by room_id/model_type/hours_back
    - _get_drift_dashboard_data room filtering and status formatting
    - _get_retraining_dashboard_data queue summary and task processing
    - _get_system_health_data component status aggregation
    - _get_alerts_dashboard_data severity/room filtering and grouping
    - _get_trends_dashboard_data time period calculation and data formatting
    - _get_dashboard_stats uptime calculation and configuration reporting
  
  - **WebSocket Data Handling:**
    - _get_websocket_initial_data system overview and feature flags
    - _get_websocket_update_data real-time metrics and alert summary
    - _handle_websocket_message JSON parsing and message type routing
    - _get_requested_data type-specific data retrieval (overview/accuracy/drift/retraining/alerts)
    - WebSocket ping/pong handling and keepalive functionality
  
  - **Manual Actions:**
    - _trigger_manual_retraining request validation and tracking manager integration
    - _acknowledge_alert alert_id validation and user tracking
    - Action availability based on mode (READONLY restrictions)
    - Success/error response formatting with detailed information
  
  - **Cache System:**
    - _get_cached_data TTL validation and expired entry cleanup
    - _cache_data thread-safe storage with timestamp tracking
    - Cache size limiting (100 entries) with oldest entry removal
    - Cache key generation for different data types and filters
    - Cache hit rate calculation and performance tracking

- Integration Tests:
  - **FastAPI Integration:**
    - All dashboard API endpoints (/api/dashboard/overview, /accuracy, /drift, /retraining, /health, /alerts, /trends, /stats)
    - WebSocket endpoint (/ws/dashboard) connection and message handling
    - CORS middleware functionality with configured origins
    - Authentication integration for API key validation
    - HTTP exception handling (500 for server errors, 503 for unavailable services)
  
  - **TrackingManager Integration:**
    - get_tracking_status() data retrieval and processing
    - get_real_time_metrics() filtering and dashboard formatting
    - get_drift_status() room filtering and status aggregation
    - get_retraining_status() queue processing and task tracking
    - get_active_alerts() severity/room filtering and alert processing
    - request_manual_retraining() action integration
    - acknowledge_alert() alert management integration
  
  - **WebSocket Real-time Updates:**
    - Client connection management and capacity limits
    - Real-time update broadcasting via _update_loop
    - Message handling for ping/pong, subscribe, request_data types
    - Connection cleanup on disconnect and error scenarios
    - Initial data transmission to new connections
    - Update interval timing and background task lifecycle
  
  - **Background Task Management:**
    - Dashboard start/stop with uvicorn server lifecycle
    - Update loop task cancellation and cleanup
    - DISABLE_BACKGROUND_TASKS environment variable handling
    - Concurrent task management (server_task, update_task)
    - Server configuration with debug/logging settings
  
  - **Cache Integration:**
    - Multi-threaded cache access with _cache_lock
    - TTL expiration during concurrent operations
    - Cache performance with tracking manager data retrieval
    - Memory usage patterns with cache size limiting
    - Cache effectiveness across different data types

- Edge Cases:
  - **FastAPI Unavailability:**
    - Dashboard initialization failure when FastAPI not installed
    - Graceful error handling with OccupancyPredictionError
    - Module import fallback behavior (FASTAPI_AVAILABLE = False)
    - Component availability validation before feature usage
  
  - **WebSocket Connection Edge Cases:**
    - Connection acceptance failure at max capacity
    - Client disconnection during message transmission
    - WebSocketDisconnect exception handling and cleanup
    - Malformed JSON message handling from clients
    - Connection metadata corruption and recovery
    - Race conditions in connection add/remove operations
    - AsyncIO timeout during message receive (30s timeout)
  
  - **TrackingManager Integration Edge Cases:**
    - Tracking manager component unavailability (validator, drift_detector, retrainer)
    - Partial status data with missing nested dictionaries
    - API method failures returning None or empty data
    - Component health status inconsistencies
    - Background task failure during status retrieval
  
  - **Cache System Edge Cases:**
    - Cache corruption during thread access
    - TTL edge cases with rapid cache access
    - Cache size overflow with high data volume
    - Memory pressure during cache cleanup
    - Cache key collision handling
    - Concurrent cache operations with lock contention
  
  - **Data Processing Edge Cases:**
    - Non-serializable data types in dashboard responses
    - Large data payloads exceeding memory limits
    - Time zone handling in timestamp operations
    - Datetime serialization failures in JSON responses
    - Partial data retrieval with incomplete tracking status
  
  - **Background Task Edge Cases:**
    - Update loop failure with automatic recovery (5s retry delay)
    - Server task cancellation during shutdown
    - Environment variable changes during runtime
    - Port conflicts during dashboard startup
    - Network interface binding failures
    - Concurrent start/stop operations

- Error Handling:
  - **Dashboard Operation Errors:**
    - DashboardError custom exception with severity levels
    - FastAPI HTTP exception mapping (500, 503 status codes)
    - OccupancyPredictionError propagation from core system
    - Error context preservation and logging integration
  
  - **WebSocket Error Handling:**
    - WebSocketDisconnect during send operations
    - JSON encoding/decoding errors in message handling
    - Connection state validation before operations
    - Broadcast failure with partial success tracking
    - Message queue overflow and backpressure handling
  
  - **Data Retrieval Error Handling:**
    - Tracking manager method failures with fallback responses
    - Cache operation failures with direct data retrieval
    - Timeout handling for long-running data operations
    - Partial data recovery with error reporting
    - Response formatting failures with error messages
  
  - **Background Task Error Handling:**
    - Update loop exception handling with retry logic
    - Server startup failures with cleanup procedures
    - Task cancellation during dashboard shutdown
    - AsyncIO task exception propagation
    - Resource cleanup on failure scenarios
  
  - **Integration Error Handling:**
    - Missing tracking manager dependencies
    - Configuration validation failures
    - Network binding errors with informative messages
    - CORS configuration errors
    - Authentication integration failures

- Coverage Target: 85%+

### src/integration/mqtt_integration_manager.py - MQTT Integration Management
**Classes Found:** MQTTIntegrationStats, MQTTIntegrationManager, MQTTIntegrationError
**Methods Analyzed:** __init__, initialize, start_integration, stop_integration, publish_prediction, publish_system_status, refresh_discovery, get_integration_stats, is_connected, add_notification_callback, remove_notification_callback, _system_status_publishing_loop, _on_mqtt_connect, _on_mqtt_disconnect, update_system_stats, update_device_availability, handle_service_command, cleanup_discovery, _check_system_availability, _handle_entity_state_change

**Required Tests:**

- **Unit Tests:**
  - **Initialization Tests:**
    - Test MQTTIntegrationManager initialization with default config
    - Test initialization with custom mqtt_config and rooms
    - Test initialization with notification callbacks
    - Test proper configuration loading from global config
    - Test stats initialization with correct default values
    - Test background task initialization and shutdown event setup
  
  - **Component Initialization Tests:**
    - Test successful MQTT publisher initialization with connection callbacks
    - Test prediction publisher initialization with correct parameters
    - Test discovery publisher initialization when discovery_enabled is True
    - Test discovery publisher skipped when discovery_enabled is False
    - Test background task startup after component initialization
    - Test stats updates after successful initialization
    
  - **Publishing Tests:**
    - Test publish_prediction with valid PredictionResult and room_id
    - Test publish_prediction returns False when integration inactive
    - Test publish_prediction with missing current_state parameter
    - Test successful prediction publishing updates stats correctly
    - Test failed prediction publishing increments error counters
    - Test publish_system_status with all optional parameters
    - Test publish_system_status with minimal parameters
    - Test system status publishing updates stats correctly
  
  - **Discovery Management Tests:**
    - Test refresh_discovery with successful discovery publisher
    - Test refresh_discovery returns False when discovery publisher unavailable
    - Test discovery stats tracking after successful refresh
    - Test cleanup_discovery with specific entity_ids list
    - Test cleanup_discovery with no entity_ids (cleanup all)
    - Test cleanup success/failure counting and reporting
  
  - **Stats and Status Tests:**
    - Test get_integration_stats returns comprehensive stats dictionary
    - Test stats include MQTT publisher stats when available
    - Test stats include prediction publisher stats when available
    - Test stats include discovery publisher stats when available
    - Test system health calculation logic
    - Test uptime calculation accuracy
    - Test is_connected returns correct status based on components
  
  - **Callback Management Tests:**
    - Test add_notification_callback adds callback to list
    - Test add_notification_callback prevents duplicates
    - Test remove_notification_callback removes existing callback
    - Test remove_notification_callback handles non-existent callback gracefully
    - Test notification callbacks called during MQTT connect events
    - Test notification callbacks called during MQTT disconnect events
    - Test notification callbacks handle both sync and async functions
  
  - **Service Command Handling Tests:**
    - Test handle_service_command with "manual_retrain" service
    - Test handle_service_command with "refresh_discovery" service  
    - Test handle_service_command with "reset_statistics" service
    - Test handle_service_command with "force_prediction" service
    - Test handle_service_command with unknown service name
    - Test service commands extract correct parameters from command_data
    - Test service commands return appropriate success/failure status
  
  - **Background Task Management Tests:**
    - Test start_integration creates and manages background tasks
    - Test start_integration handles already active integration gracefully
    - Test start_integration skips when publishing disabled
    - Test stop_integration gracefully shuts down all components
    - Test stop_integration waits for background tasks completion
    - Test stop_integration handles exceptions during shutdown
  
  - **Private Method Tests:**
    - Test _system_status_publishing_loop cycles correctly
    - Test _system_status_publishing_loop handles shutdown event
    - Test _system_status_publishing_loop error handling and retry logic
    - Test _check_system_availability returns correct status
    - Test _handle_entity_state_change processes state changes correctly
    - Test _handle_entity_state_change calls notification callbacks
    - Test MQTT connect callback updates stats and calls notifications
    - Test MQTT disconnect callback updates stats and calls notifications

- **Integration Tests:**
  - **Full Integration Lifecycle:**
    - Test complete initialization → start → publish → stop workflow
    - Test integration with real MQTTPublisher, PredictionPublisher, and DiscoveryPublisher
    - Test background task coordination between components
    - Test system status publishing integration with tracking manager
    - Test discovery publishing integration with Home Assistant
  
  - **Cross-Component Communication:**
    - Test prediction publishing triggers discovery updates when needed
    - Test MQTT connection state changes affect all publishers
    - Test system status includes stats from all integrated components
    - Test service commands coordinate between discovery and prediction publishers
    - Test notification callbacks receive events from all integrated components
  
  - **Configuration Integration:**
    - Test integration respects mqtt_config.publishing_enabled flag
    - Test integration respects mqtt_config.discovery_enabled flag
    - Test integration uses mqtt_config.status_update_interval_seconds
    - Test integration loads room configurations correctly
    - Test configuration changes affect component initialization

- **Edge Cases:**
  - **Component Unavailability:**
    - Test behavior when MQTT publisher initialization fails
    - Test behavior when prediction publisher is None
    - Test behavior when discovery publisher is None
    - Test graceful degradation when some components unavailable
    - Test stats reporting when components partially initialized
  
  - **Timing and Race Conditions:**
    - Test rapid start/stop cycles
    - Test concurrent publish_prediction calls
    - Test background task shutdown during active publishing
    - Test notification callback exceptions don't break event handling
    - Test stats updates during concurrent operations
  
  - **Resource Management:**
    - Test background task cleanup on initialization failure
    - Test MQTT publisher cleanup on integration stop
    - Test notification callback list management during concurrent modifications
    - Test system stats memory usage with large histories
    - Test discovery cleanup with large entity lists
  
  - **State Transition Edge Cases:**
    - Test publishing when integration becomes inactive mid-operation
    - Test start_integration called multiple times
    - Test stop_integration called when not active
    - Test component failures during active integration
    - Test recovery after temporary MQTT disconnection

- **Error Handling:**
  - **Initialization Errors:**
    - Test MQTTIntegrationError raised on component initialization failure
    - Test proper error logging and stats tracking on init failures
    - Test cleanup of partially initialized components on failure
    - Test error propagation with cause information
    - Test graceful handling of missing configuration
  
  - **Runtime Errors:**
    - Test exception handling in publish_prediction method
    - Test exception handling in publish_system_status method
    - Test exception handling in service command processing
    - Test background task exception handling and logging
    - Test notification callback exception isolation
  
  - **Network and Communication Errors:**
    - Test MQTT connection failure handling
    - Test MQTT disconnection during publishing
    - Test timeout errors in discovery operations
    - Test partial failure handling in batch operations
    - Test retry logic for transient failures
  
  - **Data Validation Errors:**
    - Test invalid PredictionResult handling
    - Test invalid room_id parameter validation
    - Test malformed service command data handling
    - Test invalid entity_id processing
    - Test stats calculation with invalid data

- Coverage Target: 85%+

### src/integration/ha_tracking_bridge.py - HA Tracking Bridge
**Classes Found:** HATrackingBridgeStats (dataclass), HATrackingBridge (main bridge class), HATrackingBridgeError (custom exception)
**Methods Analyzed:** __init__, initialize, shutdown, handle_prediction_made, handle_accuracy_alert, handle_drift_detected, handle_retraining_started, handle_retraining_completed, get_bridge_stats, _setup_tracking_event_handlers, _setup_command_delegation, _start_background_tasks, _system_status_sync_loop, _metrics_sync_loop, _update_system_alert_status, _update_system_drift_status, _delegate_retrain_model, _delegate_validate_model, _delegate_force_prediction, _delegate_check_database, _delegate_generate_diagnostic

**Required Tests:**
- Unit Tests:
  - **HATrackingBridgeStats dataclass:**
    - Default field values initialization
    - Field type validation and dataclass behavior
    - Stats update operations with proper timestamps
    - Error counter incrementing and last_error tracking
    - Statistics serialization for reporting
  
  - **HATrackingBridge.__init__ method:**
    - Proper initialization with tracking_manager and enhanced_integration_manager
    - Stats object creation with default values
    - Background task list initialization (empty)
    - Shutdown event creation and initial state
    - Event handler dictionary initialization (empty)
    - Bridge active state initialization (False)
    - Logger initialization and info message logging
  
  - **HATrackingBridge.initialize method:**
    - Successful initialization workflow with all setup steps
    - Tracking event handlers setup (_setup_tracking_event_handlers call)
    - Enhanced integration manager tracking_manager assignment
    - Command delegation setup (_setup_command_delegation call)
    - Background tasks startup (_start_background_tasks call)
    - Bridge active state set to True
    - Stats bridge_initialized flag set to True
    - Exception handling with proper error logging and stats update
    - Error propagation when initialization fails
  
  - **HATrackingBridge.shutdown method:**
    - Shutdown event setting and bridge active state False
    - Background tasks cancellation (only non-done tasks)
    - AsyncIO gather with return_exceptions=True for task completion
    - Proper cleanup even when no background tasks exist
    - Exception handling during shutdown with error logging
    - Graceful handling of already cancelled tasks
  
  - **Event Handler Methods:**
    - **handle_prediction_made:** Bridge active check, enhanced integration manager call, stats updates, exception handling with error stats
    - **handle_accuracy_alert:** Bridge active check, alert data extraction with getattr fallback, system alert status update, error handling
    - **handle_drift_detected:** Bridge active check, drift data extraction, system drift status update, error handling
    - **handle_retraining_started:** Bridge active check, entity state update with training info, error handling
    - **handle_retraining_completed:** Bridge active check, training completion update, room accuracy update when available, error handling
  
  - **Bridge Statistics:**
    - get_bridge_stats() returns proper dictionary structure
    - Stats object serialization (__dict__ usage)
    - Background task count accuracy
    - Event handler count accuracy
    - Bridge active state reporting
  
  - **Private Setup Methods:**
    - **_setup_tracking_event_handlers:** Event handler dictionary population, tracking manager callback registration (if supported), handler count logging
    - **_setup_command_delegation:** Original handlers backup, command handler override with delegation methods, error handling during setup
  
  - **Command Delegation Methods (5 methods):**
    - **_delegate_retrain_model:** Parameter extraction, tracking manager hasattr check, method call with parameters, stats update, result formatting
    - **_delegate_validate_model:** Parameter extraction, validation method call, stats tracking, error response for unsupported methods
    - **_delegate_force_prediction:** Room ID parameter, force prediction delegation, stats update, unsupported method handling
    - **_delegate_check_database:** Database health check delegation, stats update, fallback success response
    - **_delegate_generate_diagnostic:** Parameter extraction, diagnostic report generation, stats update, fallback to bridge stats

- Integration Tests:
  - **Full Bridge Lifecycle Integration:**
    - Initialize bridge with mock tracking_manager and enhanced_integration_manager
    - Verify all setup methods called in correct order
    - Test background task creation and proper asyncio task management
    - Validate shutdown properly cancels all tasks and cleans up
  
  - **Event Flow Integration:**
    - TrackingManager event emission through bridge to EnhancedIntegrationManager
    - Prediction result propagation with proper data transformation
    - Alert and drift event handling end-to-end
    - Retraining event lifecycle (started -> completed) integration
  
  - **Command Delegation Integration:**
    - HA command reception through enhanced integration manager
    - Command delegation to tracking manager methods
    - Response formatting and error handling through full stack
    - Stats tracking across command delegation workflow
  
  - **Background Task Integration:**
    - System status sync loop with tracking manager integration
    - Metrics sync loop with room-specific accuracy updates
    - Shutdown signal handling and graceful task termination
    - Exception recovery in background tasks with proper logging

- Edge Cases:
  - **Bridge State Edge Cases:**
    - Method calls when bridge is not active (should return early)
    - Initialize called multiple times (should handle gracefully)
    - Shutdown called multiple times or before initialize
    - Event handlers called during shutdown process
    - Command delegation when bridge is shutting down
  
  - **TrackingManager Availability Edge Cases:**
    - TrackingManager missing expected methods (hasattr checks)
    - TrackingManager method calls returning None or raising exceptions
    - TrackingManager callback registration not supported
    - Background task loops when tracking manager methods fail
    - Command delegation when tracking manager is unavailable
  
  - **Enhanced Integration Manager Edge Cases:**
    - Enhanced integration manager method failures during event handling
    - Entity state update failures with proper error logging
    - System status update methods not available
    - Command handler dictionary manipulation failures
  
  - **Background Task Edge Cases:**
    - AsyncIO task creation failures during _start_background_tasks
    - Background task exceptions during shutdown (gather with return_exceptions)
    - Shutdown event race conditions with background task loops
    - Background task loop exceptions with proper error recovery
    - Long-running background tasks during quick shutdown
  
  - **Data Handling Edge Cases:**
    - Event objects missing expected attributes (getattr with defaults)
    - Prediction result objects with missing or invalid data
    - Alert objects with None or invalid severity levels
    - Drift metrics with missing required fields
    - Retraining info/result with incomplete data structures
  
  - **Async/Await Edge Cases:**
    - Concurrent event handler calls with shared state modification
    - Background task cancellation during asyncio.sleep operations
    - Exception propagation from async event handlers
    - Async context manager behavior during shutdown
    - AsyncIO timeout scenarios in background sync loops

- Error Handling:
  - **Initialization Errors:**
    - HATrackingBridgeError with proper error code and severity
    - Exception propagation from setup methods
    - Stats error tracking (bridge_errors increment, last_error storage)
    - Logging integration with proper error context
  
  - **Event Handler Errors:**
    - Exception catching in all handle_* methods
    - Error stats updates (bridge_errors, last_error)
    - Error logging with proper context and room information
    - Graceful degradation when event handling fails
  
  - **Command Delegation Errors:**
    - Exception handling in all _delegate_* methods
    - Error response formatting {"status": "error", "message": str(e)}
    - Stats tracking even when commands fail
    - Fallback responses for unsupported operations
  
  - **Background Task Errors:**
    - Exception handling in sync loop methods with sleep recovery
    - Proper error logging without breaking loop execution
    - Async exception handling in background task management
    - Resource cleanup on background task failures
  
  - **Custom Exception Handling:**
    - HATrackingBridgeError inheritance from OccupancyPredictionError
    - Proper error code assignment ("HA_TRACKING_BRIDGE_ERROR")
    - Severity level configuration (default MEDIUM)
    - Kwargs propagation to parent exception class

- Coverage Target: 85%+

### src/integration/enhanced_mqtt_manager.py - Enhanced MQTT Features
**Classes Found:** EnhancedIntegrationStats, EnhancedMQTTIntegrationManager, EnhancedMQTTIntegrationError
**Methods Analyzed:** __init__, initialize, shutdown, publish_prediction, publish_system_status, handle_websocket_connection, create_sse_stream, add_realtime_callback, remove_realtime_callback, get_integration_stats, get_connection_info, start_discovery_publishing, stop_discovery_publishing, publish_room_batch, _start_enhanced_monitoring, _record_publish_performance, _update_enhanced_stats, _determine_system_status, _performance_monitoring_loop, _stats_update_loop

**Required Tests:**
- Unit Tests:
  - **EnhancedIntegrationStats Dataclass Tests:**
    - Proper initialization with mqtt_stats and realtime_stats parameters
    - Default value assignments for combined metrics fields
    - Serialization and deserialization with asdict() functionality
    - Field type validation for datetime and float types
    - Performance metrics calculation accuracy
  
  - **EnhancedMQTTIntegrationManager Initialization Tests:**
    - Constructor with all parameters (mqtt_config, rooms, notification_callbacks, enabled_channels)
    - Constructor with None parameters using global config loading
    - Base MQTT manager and realtime publisher initialization
    - Channel selection with default [MQTT, WEBSOCKET, SSE] configuration
    - Statistics object creation and initial state validation
    - Performance tracking list initialization (_publish_times, _publish_latencies)
    - Background task list and shutdown event setup
  
  - **Core Publishing Method Tests:**
    - publish_prediction() with valid PredictionResult and room_id
    - publish_prediction() performance metric recording and latency calculation
    - publish_prediction() error handling with exception scenarios
    - publish_prediction() return format validation for all channels
    - publish_system_status() with complete status data integration
    - publish_system_status() combining MQTT and realtime results
    - publish_system_status() system status determination logic
    - publish_room_batch() multiple room prediction publishing
  
  - **Statistics and Monitoring Tests:**
    - get_integration_stats() comprehensive data aggregation
    - get_integration_stats() error handling with fallback values
    - get_connection_info() WebSocket and SSE connection counting
    - _update_enhanced_stats() predictions per minute calculation
    - _update_enhanced_stats() average latency computation
    - _record_publish_performance() metrics tracking and cleanup
    - _record_publish_performance() success rate calculation with EMA
  
  - **Real-time Integration Tests:**
    - handle_websocket_connection() delegation to realtime publisher
    - create_sse_stream() stream creation with optional room filtering
    - add_realtime_callback() and remove_realtime_callback() management
    - Callback function registration and deregistration validation
  
  - **Background Task Management Tests:**
    - _start_enhanced_monitoring() task creation and management
    - _performance_monitoring_loop() 30-second update cycle
    - _performance_monitoring_loop() shutdown event handling
    - _stats_update_loop() 5-minute cleanup cycle with data retention
    - _stats_update_loop() 1-hour data cutoff enforcement
    - Background task cancellation during shutdown process
  
  - **System Status Determination Tests:**
    - _determine_system_status() with various connection states
    - _determine_system_status() threshold-based alert handling (>5 alerts)
    - _determine_system_status() offline/degraded/online status logic
    - _determine_system_status() error handling with "unknown" fallback
  
  - **Delegation Method Tests:**
    - start_discovery_publishing() and stop_discovery_publishing() passthrough
    - Base MQTT manager method availability checking with hasattr()
    - Backward compatibility preservation for existing MQTT functionality

- Integration Tests:
  - **Multi-Channel Publishing Integration:**
    - End-to-end prediction publishing across MQTT, WebSocket, and SSE channels
    - System status broadcasting with real-time and MQTT coordination
    - Performance metric aggregation from multiple publishing channels
    - Channel-specific error handling and partial success scenarios
  
  - **Real-time System Integration:**
    - Integration with RealtimePublishingSystem initialization and configuration
    - WebSocket connection handling through enhanced manager
    - SSE stream creation and management integration
    - Callback system integration with broadcast events
  
  - **Base MQTT Manager Integration:**
    - Enhanced manager wrapping base MQTTIntegrationManager functionality
    - Statistics synchronization between base and enhanced systems
    - Discovery publishing delegation and coordination
    - Configuration sharing between enhanced and base systems
  
  - **Performance Monitoring Integration:**
    - Background task coordination with main application lifecycle
    - Statistics update coordination with real-time system metrics
    - Performance data cleanup integration with memory management
    - Shutdown coordination between enhanced and base systems
  
  - **Configuration Integration:**
    - Global configuration loading with get_config() integration
    - MQTT and room configuration sharing across components
    - Channel configuration propagation to realtime publisher
    - Notification callback coordination across systems

- Edge Cases:
  - **Initialization Edge Cases:**
    - Global configuration loading failures with None parameters
    - Base MQTT manager initialization failures during enhanced setup
    - Realtime publisher initialization failures with error propagation
    - Background task startup failures during enhanced monitoring
    - Missing prediction_publisher attribute handling in base manager
  
  - **Publishing Edge Cases:**
    - publish_prediction() with invalid PredictionResult data structures
    - publish_prediction() with non-existent room_id values
    - Multiple channel publishing with partial failures
    - Performance metric recording with extreme latency values (>10 seconds)
    - Success rate calculation with zero total channels
    - System status publishing with missing or None statistics data
  
  - **Statistics Edge Cases:**
    - Statistics calculation with empty performance data lists
    - Statistics update with missing realtime publisher metrics
    - Connection counting with disconnected or invalid connections
    - Performance metric calculation with concurrent access scenarios
    - Stats update during system shutdown with cancelled tasks
  
  - **Background Task Edge Cases:**
    - Performance monitoring loop exceptions with 30-second recovery
    - Stats update loop exceptions with 60-second fallback delays
    - Task cancellation during active monitoring operations
    - Shutdown event handling with concurrent loop operations
    - Memory cleanup during high-frequency publishing scenarios
  
  - **Real-time Integration Edge Cases:**
    - WebSocket connection handling failures with delegation errors
    - SSE stream creation failures with realtime publisher unavailability
    - Callback registration with duplicate or invalid callback functions
    - Callback removal with non-existent callback references
    - Realtime publisher state inconsistencies during enhanced operations
  
  - **Configuration Edge Cases:**
    - Global config loading failures with system configuration unavailability
    - MQTT configuration validation failures during initialization
    - Room configuration processing with malformed or missing data
    - Channel configuration with unsupported or invalid channel types
    - Configuration updates during runtime with enhanced manager active
  
  - **System Status Edge Cases:**
    - System status determination with missing base manager attributes
    - Database connection status checking with timeout scenarios
    - MQTT connection status validation with stale connection data
    - Realtime system availability checking with _publishing_active failures
    - Alert count processing with negative or invalid values

- Error Handling:
  - **Enhanced Integration Error Handling:**
    - EnhancedMQTTIntegrationError custom exception with cause chaining
    - Error severity classification with ErrorSeverity enum values
    - Error code standardization ("ENHANCED_MQTT_INTEGRATION_ERROR")
    - Error context preservation from underlying system failures
  
  - **Publishing Error Handling:**
    - publish_prediction() exception handling with error result formatting
    - publish_system_status() error handling with combined result structure
    - publish_room_batch() error handling with per-room error tracking
    - Channel-specific error handling with partial success reporting
    - Performance metric recording errors with graceful degradation
  
  - **Background Task Error Handling:**
    - Performance monitoring loop exception handling with continuous operation
    - Stats update loop error recovery with retry mechanisms
    - Task cancellation handling during shutdown procedures
    - AsyncIO timeout handling in monitoring loops (30s/300s timeouts)
    - Resource cleanup error handling with partial failure scenarios
  
  - **Integration Error Handling:**
    - Base MQTT manager method failures with hasattr() safety checks
    - Realtime publisher method failures with delegation error handling
    - Statistics retrieval failures with error result formatting
    - Configuration access failures with fallback value provision
    - System component unavailability with graceful degradation
  
  - **Statistics Error Handling:**
    - Statistics calculation errors with default value fallbacks
    - Connection info retrieval errors with error result formatting
    - Performance metric calculation errors with zero-division protection
    - Data cleanup errors during background processing
    - Concurrent access errors with thread-safety considerations

- Coverage Target: 85%+

### src/integration/monitoring_api.py - Monitoring API Endpoints
**Classes Found:** SystemStatus, MetricsResponse, HealthCheckResponse, AlertsResponse (Pydantic models)
**Methods Analyzed:** get_health_status, get_system_status, get_prometheus_metrics, get_metrics_summary, get_alerts_status, get_performance_summary, get_performance_trend, trigger_test_alert, get_monitoring_info

**Required Tests:**

- **Unit Tests:**
  - Test SystemStatus model validation with valid/invalid data
  - Test MetricsResponse model field validation and serialization
  - Test HealthCheckResponse model with complex checks dict
  - Test AlertsResponse model with notification_channels list
  - Mock get_monitoring_integration() calls for all endpoints
  - Mock get_metrics_manager() and get_alert_manager() dependencies
  - Test health check response formatting with various status types
  - Test system status health score calculation logic (CPU/memory percentages)
  - Test metrics count calculation from Prometheus text format
  - Test performance summary time window validation (1-168 hours)
  - Test trend analysis parameter validation (metric_name, room_id, hours)
  - Test test alert payload generation with proper context structure

- **Integration Tests:**
  - Test complete health endpoint flow with real monitoring integration
  - Test status endpoint integration with monitoring status API
  - Test metrics endpoint returns valid Prometheus format text
  - Test metrics summary endpoint counts actual metrics correctly
  - Test alerts endpoint integration with alert manager status
  - Test performance endpoint with various time windows (1h, 24h, 168h)
  - Test trend analysis endpoint with different metric types
  - Test alert triggering creates actual alert through alert manager
  - Test monitoring info endpoint returns correct service information
  - Test router prefix and tags are properly configured

- **Edge Cases:**
  - Health checks with partial failures (some healthy, some failed)
  - System status with missing or malformed monitoring data
  - Metrics endpoint when no metrics are available (empty string)
  - Performance summary with invalid hours parameter (0, 169, negative)
  - Trend analysis with non-existent metric names
  - Alert triggering when alert manager is unavailable
  - Health response with empty or None health results
  - Status response with missing health_details in monitoring data
  - Metrics summary with metrics containing only comments (no data lines)
  - Performance endpoint with room_id filtering scenarios

- **Error Handling:**
  - HTTPException 500 when monitoring integration fails
  - HTTPException 500 when health monitor is unavailable
  - HTTPException 500 when metrics manager throws exceptions
  - HTTPException 400 for invalid hours parameter in performance endpoints
  - HTTPException 500 when alert manager fails to trigger alerts
  - Logger error calls for all exception scenarios
  - Exception propagation from dependency injection functions
  - Graceful handling when monitoring status is incomplete/corrupted
  - Error responses maintain proper HTTP status codes
  - Exception details included in HTTPException messages
  - Async exception handling in all endpoint methods
  - Resource cleanup on failed health checks or status requests

- Coverage Target: 85%+

### src/integration/ha_entity_definitions.py - HA Entity Definitions
**Classes Found:** HAEntityType, HADeviceClass, HAEntityCategory, HAStateClass, HAEntityConfig, HASensorEntityConfig, HABinarySensorEntityConfig, HAButtonEntityConfig, HASwitchEntityConfig, HANumberEntityConfig, HASelectEntityConfig, HATextEntityConfig, HAImageEntityConfig, HADateTimeEntityConfig, HAServiceDefinition, HAEntityDefinitions, HAEntityDefinitionsError
**Methods Analyzed:** __init__, define_all_entities, define_all_services, publish_all_entities, publish_all_services, get_entity_definition, get_service_definition, get_entity_stats, _define_room_entities, _define_system_entities, _define_diagnostic_entities, _define_control_entities, _define_model_services, _define_system_services, _define_diagnostic_services, _define_room_services, _create_service_button_config, _publish_entity_discovery, _add_sensor_attributes, _add_binary_sensor_attributes, _add_button_attributes, _add_switch_attributes, _add_number_attributes, _add_select_attributes, _add_text_attributes, _add_image_attributes, _add_datetime_attributes

**Required Tests:**
- Unit Tests:
  - **Enum Classes Testing:**
    - HAEntityType enum value validation and completeness
    - HADeviceClass enum coverage for all sensor/binary_sensor/number/button types
    - HAEntityCategory enum values (config, diagnostic, system)
    - HAStateClass enum values (measurement, total, total_increasing)
    
  - **Entity Configuration Classes:**
    - HAEntityConfig base class initialization with all parameters
    - HASensorEntityConfig with sensor-specific attributes and __post_init__ validation
    - HABinarySensorEntityConfig with binary sensor attributes and type enforcement
    - HAButtonEntityConfig with command topics and QoS settings
    - HASwitchEntityConfig with state/payload configurations
    - HANumberEntityConfig with min/max/step validation and mode options
    - HASelectEntityConfig with options list and template validation
    - HATextEntityConfig with character limits and pattern validation
    - HAImageEntityConfig with URL template and SSL verification
    - HADateTimeEntityConfig with format string validation
    - Field validation and default value assignments
    - Device info integration and unique ID generation
    
  - **Service Definition Testing:**
    - HAServiceDefinition initialization with all service fields
    - Service field validation and selector configurations
    - Command topic and template generation
    - Target selector validation and response support flags
    
  - **HAEntityDefinitions Core Methods:**
    - __init__ method with DiscoveryPublisher, MQTT config, rooms, tracking config
    - Entity registry initialization and statistics setup
    - Component dependency injection and validation
    
  - **Entity Definition Methods:**
    - define_all_entities() complete entity ecosystem creation
    - define_all_services() service definition generation
    - Room-specific entity creation via _define_room_entities()
    - System-wide entity creation via _define_system_entities()
    - Diagnostic entity creation via _define_diagnostic_entities()
    - Control entity creation via _define_control_entities()
    - Statistics tracking and error counting
    
  - **Service Definition Methods:**
    - _define_model_services() with model management services
    - _define_system_services() with system control operations
    - _define_diagnostic_services() with monitoring capabilities
    - _define_room_services() with room-specific actions
    - Service field validation and template generation
    
  - **Publishing Operations:**
    - publish_all_entities() with entity type ordering and result tracking
    - publish_all_services() converting services to button entities
    - _publish_entity_discovery() with MQTT discovery protocol
    - Discovery payload creation and attribute addition
    - Entity-specific attribute handling methods
    
  - **Data Retrieval and Statistics:**
    - get_entity_definition() entity lookup by ID
    - get_service_definition() service lookup by name
    - get_entity_stats() comprehensive statistics generation
    - Entity type counting and categorization
    - Registry state management and availability tracking

- Integration Tests:
  - **MQTT Discovery Integration:**
    - Complete entity publishing workflow with real MQTT broker
    - Discovery topic generation and payload validation
    - Device availability integration with discovery publisher
    - Entity state topic creation and template validation
    - QoS and retention settings for different entity types
    
  - **Room Configuration Integration:**
    - Entity creation for multiple room configurations
    - Room-specific entity naming and unique ID generation
    - Sensor mapping and device class assignment
    - Topic prefix integration with room IDs
    
  - **Service Button Integration:**
    - Service to button entity conversion and publishing
    - Command topic routing and payload generation
    - Service field to entity attribute mapping
    - Response handling and status tracking
    
  - **Statistics and Monitoring Integration:**
    - Real-time statistics updates during entity operations
    - Error tracking and recovery operations
    - Entity availability status management
    - Performance metrics collection during mass publishing
    
  - **Discovery Publisher Integration:**
    - Device info propagation to all entities
    - Availability topic inheritance and template validation
    - MQTT publisher dependency injection and usage
    - Discovery prefix and identifier consistency

- Edge Cases:
  - **Entity Configuration Edge Cases:**
    - Empty or invalid room configurations
    - Missing required fields in entity configurations
    - Duplicate unique IDs across different entity types
    - Invalid device class and entity type combinations
    - Template validation with malformed Jinja2 syntax
    - Unit of measurement validation for different device classes
    - Icon validation and fallback behavior
    
  - **Topic Generation Edge Cases:**
    - Special characters in room IDs affecting topic paths
    - Long unique IDs exceeding MQTT topic length limits
    - Topic prefix conflicts and collision handling
    - Command topic validation for bidirectional entities
    - State topic conflicts between different entity types
    
  - **Service Definition Edge Cases:**
    - Invalid service field configurations
    - Missing required fields in service definitions
    - Circular dependencies in service field selectors
    - Invalid target selectors for service entities
    - Template validation in service command templates
    
  - **Publishing Edge Cases:**
    - MQTT broker unavailability during publishing
    - Partial publishing failures with some entities succeeding
    - Discovery payload size exceeding MQTT message limits
    - Entity type ordering conflicts during batch publishing
    - QoS level conflicts between different entity configurations
    
  - **Entity Registry Edge Cases:**
    - Registry corruption during concurrent access
    - Memory pressure with large entity counts (1000+ entities)
    - Entity definition updates during active publishing
    - Statistics calculation overflow with high entity counts
    - Entity availability tracking with intermittent connections
    
  - **Configuration Validation Edge Cases:**
    - TrackingConfig optional parameter handling
    - Device info validation with missing attributes
    - MQTT config validation with invalid broker settings
    - Room config validation with nested room structures
    - Entity attribute validation with type mismatches

- Error Handling:
  - **Entity Definition Errors:**
    - HAEntityDefinitionsError custom exception handling
    - Error severity classification and propagation
    - Entity definition failure recovery and partial success
    - Statistics error tracking and reporting
    - Registry corruption detection and recovery
    
  - **Publishing Operation Errors:**
    - MQTTPublishResult failure handling and retry logic
    - Discovery payload creation failures with fallback
    - Entity-specific attribute addition errors
    - Batch publishing failure with individual result tracking
    - Topic generation failures with validation errors
    
  - **Configuration Validation Errors:**
    - Invalid entity configuration detection and reporting
    - Service definition validation failures
    - Field validation errors with detailed messages
    - Template compilation errors with context information
    - Device class compatibility validation failures
    
  - **Integration Error Handling:**
    - Discovery publisher dependency failures
    - MQTT publisher integration errors
    - Room configuration validation failures
    - Tracking config optional dependency handling
    - Component availability validation errors
    
  - **Runtime Operation Errors:**
    - Entity lookup failures with informative responses
    - Statistics calculation errors with fallback values
    - Registry access failures with concurrent operations
    - Memory management errors with large entity counts
    - Resource cleanup failures during shutdown operations

- Coverage Target: 85%+

### src/integration/enhanced_mqtt_manager.py - Enhanced MQTT Features
**Classes Found:** EnhancedIntegrationStats, EnhancedMQTTIntegrationManager, EnhancedMQTTIntegrationError
**Methods Analyzed:** __init__, initialize, shutdown, publish_prediction, publish_system_status, handle_websocket_connection, create_sse_stream, add_realtime_callback, remove_realtime_callback, get_integration_stats, get_connection_info, start_discovery_publishing, stop_discovery_publishing, publish_room_batch, _start_enhanced_monitoring, _record_publish_performance, _update_enhanced_stats, _determine_system_status, _performance_monitoring_loop, _stats_update_loop

**Required Tests:**
- Unit Tests:
  - **EnhancedIntegrationStats dataclass validation:**
    - Nested dataclass initialization with MQTTIntegrationStats and PublishingMetrics
    - Default value assignments for optional fields
    - Type validation for all fields including datetime and float types
    - Dataclass serialization with asdict() for JSON publishing
  
  - **EnhancedMQTTIntegrationManager initialization:**
    - Constructor with all parameter combinations (None configs, custom configs)
    - Automatic config loading from global config when None provided
    - Base MQTT manager initialization with correct parameters
    - Real-time publisher setup with enabled channels configuration
    - Statistics initialization and default values setup
    - Background task list and shutdown event initialization
  
  - **Integration lifecycle management:**
    - Initialize() method with successful MQTT and real-time system setup
    - Initialize() with prediction publisher attribute validation and assignment
    - Enhanced monitoring startup with background task creation
    - Shutdown() with proper cleanup sequence (event set, task cancellation, system shutdown)
    - Background task cancellation with exception handling
  
  - **Prediction publishing functionality:**
    - publish_prediction() with comprehensive channel broadcasting
    - Performance tracking during prediction publishing (latency calculation)
    - MQTT stats update integration with base manager
    - Error handling for failed publication attempts
    - Results dictionary structure validation for all channels
  
  - **System status publishing:**
    - publish_system_status() with status data preparation
    - System status determination logic (online/degraded/offline)
    - MQTT and real-time channel result combination
    - Comprehensive status payload creation with all required fields
    - Error handling for failed status publishing
  
  - **WebSocket and SSE delegation:**
    - handle_websocket_connection() delegation to realtime publisher
    - create_sse_stream() delegation with optional room filtering
    - Proper parameter passing to underlying real-time publisher methods
  
  - **Callback management:**
    - add_realtime_callback() with proper delegation to publisher
    - remove_realtime_callback() with callback validation
    - Callback list management and validation
  
  - **Statistics collection and reporting:**
    - get_integration_stats() comprehensive data aggregation
    - Enhanced stats update with performance metrics calculation
    - Connection info gathering from multiple sources
    - Statistics error handling and fallback values
  
  - **Backward compatibility delegation:**
    - start_discovery_publishing() delegation with hasattr validation
    - stop_discovery_publishing() delegation with proper error handling
    - publish_room_batch() with multiple room processing
    - Attribute existence checking for optional base manager methods
  
  - **Performance monitoring:**
    - _record_publish_performance() with latency tracking
    - Publish time list management with hour-based cleanup
    - Success rate calculation using exponential moving average
    - Performance data structures maintenance
  
  - **Statistics updates:**
    - _update_enhanced_stats() with real-time metrics aggregation
    - Predictions per minute calculation from recent publish times
    - Channel and connection count aggregation from publisher
    - MQTT stats integration from base manager
  
  - **System status determination:**
    - _determine_system_status() with multiple condition checking
    - Database, MQTT, and real-time system status evaluation
    - Alert count threshold evaluation for degraded status
    - Status string return validation ("online", "degraded", "offline", "unknown")

- Integration Tests:
  - **Full system integration:**
    - End-to-end initialization with real MQTT config and room configurations
    - Real-time publisher integration with all channel types enabled
    - Base MQTT manager integration with prediction publisher sharing
    - Complete shutdown sequence with proper resource cleanup
  
  - **Multi-channel publishing integration:**
    - Prediction publishing across MQTT, WebSocket, and SSE channels
    - System status publishing with comprehensive data propagation
    - Batch room publishing with multiple prediction results
    - Performance tracking across all publishing operations
  
  - **Real-time connection handling:**
    - WebSocket connection establishment and management
    - SSE stream creation and client management
    - Callback integration with real-time events
    - Connection statistics aggregation and reporting
  
  - **Background monitoring integration:**
    - Performance monitoring loop with statistics updates
    - Stats cleanup loop with time-based data retention
    - Monitoring task lifecycle with shutdown event handling
    - Error recovery in monitoring loops with continued operation
  
  - **Configuration loading integration:**
    - Global config integration with automatic MQTT and room config loading
    - Custom config override with proper precedence handling
    - Notification callback integration across both MQTT and real-time systems
    - Channel configuration with enabled/disabled channel handling

- Edge Cases:
  - **Initialization edge cases:**
    - Initialization with corrupted global configuration
    - Base MQTT manager initialization failure with error propagation
    - Real-time publisher initialization failure with proper cleanup
    - Prediction publisher attribute missing in base manager
    - Enhanced monitoring startup failure with partial initialization
  
  - **Publishing edge cases:**
    - Prediction publishing with empty or invalid prediction results
    - System status publishing with None values for optional parameters
    - Multi-room batch publishing with some rooms failing
    - Publishing during shutdown with background task cancellation
    - Channel-specific publishing failures with partial success tracking
  
  - **Performance tracking edge cases:**
    - Performance recording with zero latency values
    - Success rate calculation with no previous rate data
    - Statistics update with empty publish history
    - Memory management with large publish history accumulation
    - Time-based cleanup with timezone handling across daylight savings
  
  - **Connection management edge cases:**
    - WebSocket delegation with connection failures
    - SSE stream creation with invalid room IDs
    - Callback management with duplicate or None callbacks
    - Connection statistics with disconnected publishers
  
  - **Monitoring loop edge cases:**
    - Performance monitoring loop with continuous exceptions
    - Stats cleanup loop with corrupted data structures
    - Background task cancellation during active monitoring
    - Shutdown event handling with concurrent access
    - Memory pressure with accumulated performance data
  
  - **Statistics aggregation edge cases:**
    - Statistics calculation with missing publisher components
    - Integration stats with corrupted nested dataclasses
    - Connection info gathering with unavailable real-time systems
    - Performance metrics with division by zero scenarios
    - Timestamp handling with timezone-naive datetime objects

- Error Handling:
  - **EnhancedMQTTIntegrationError custom exception:**
    - Error instantiation with proper error code and severity
    - Exception inheritance from OccupancyPredictionError
    - Error message formatting and cause handling
    - Severity classification with ErrorSeverity enum
  
  - **Initialization error handling:**
    - Base MQTT manager initialization failure with exception propagation
    - Real-time publisher setup errors with proper error wrapping
    - Enhanced monitoring startup errors with graceful degradation
    - Configuration loading errors with informative messages
  
  - **Publishing operation errors:**
    - Prediction publishing failure with error result dictionary
    - System status publishing errors with combined result handling
    - Batch publishing errors with individual room failure tracking
    - Channel-specific publishing errors with isolation
  
  - **Background task error handling:**
    - Performance monitoring loop exceptions with recovery attempts
    - Stats update loop failures with continued operation
    - Task cancellation errors during shutdown with proper logging
    - Asyncio timeout handling in monitoring loops
  
  - **Delegation error handling:**
    - Base manager method delegation with AttributeError handling
    - Real-time publisher method errors with proper exception propagation
    - Statistics gathering errors with fallback values
    - Connection info retrieval errors with partial data return
  
  - **Resource management errors:**
    - Memory management during performance data accumulation
    - Background task resource cleanup with exception handling
    - Publisher shutdown errors with continued cleanup attempts
    - Statistics update errors with data structure integrity maintenance
  
  - **Integration error recovery:**
    - Partial system initialization with degraded functionality
    - Publishing channel failure isolation with remaining channel operation
    - Statistics calculation errors with fallback metric values
    - Connection tracking errors with graceful degradation

- Coverage Target: 85%+

### src/integration/mqtt_integration_manager.py - MQTT Integration Management
**Classes Found:** MQTTIntegrationStats (dataclass), MQTTIntegrationManager, MQTTIntegrationError
**Methods Analyzed:** __init__, initialize, start_integration, stop_integration, publish_prediction, publish_system_status, refresh_discovery, get_integration_stats, is_connected, add_notification_callback, remove_notification_callback, update_system_stats, update_device_availability, handle_service_command, cleanup_discovery, _system_status_publishing_loop, _on_mqtt_connect, _on_mqtt_disconnect, _check_system_availability, _handle_entity_state_change

**Required Tests:**
- Unit Tests:
  - **MQTTIntegrationStats Tests:**
    - Default dataclass field initialization and validation
    - Statistics update operations and field tracking
    - Datetime field serialization and timezone handling
    - Error counter incrementing and message storage
    
  - **MQTTIntegrationManager Initialization Tests:**
    - Constructor with default config loading from get_config()
    - Constructor with explicit mqtt_config and rooms parameters
    - Constructor with notification_callbacks initialization
    - Component initialization with None validation
    - System start time recording and timezone handling
    - Background task list and shutdown event initialization
    
  - **Core Integration Lifecycle Tests:**
    - initialize() with publishing disabled config handling
    - initialize() with successful MQTT publisher creation
    - initialize() with prediction publisher integration
    - initialize() with discovery publisher optional setup
    - initialize() error handling and MQTTIntegrationError raising
    - start_integration() with already active integration detection
    - start_integration() with disabled publishing configuration
    - start_integration() background task creation and management
    - stop_integration() graceful shutdown with event signaling
    - stop_integration() task cleanup and publisher stopping
    
  - **Publishing Operation Tests:**
    - publish_prediction() with inactive integration early return
    - publish_prediction() successful publishing with stats update
    - publish_prediction() failure handling with error tracking
    - publish_system_status() with optional parameter handling
    - publish_system_status() statistics update and success tracking
    - refresh_discovery() with missing discovery publisher handling
    - refresh_discovery() partial success result aggregation
    
  - **Statistics and Status Tests:**
    - get_integration_stats() comprehensive stats dictionary creation
    - get_integration_stats() nested publisher stats integration
    - get_integration_stats() system health summary calculation
    - get_integration_stats() datetime field ISO format serialization
    - is_connected() boolean logic with multiple condition checking
    - update_system_stats() cached stats storage for publishing
    
  - **Callback Management Tests:**
    - add_notification_callback() with duplicate prevention
    - remove_notification_callback() with existence checking
    - Callback execution in _on_mqtt_connect with error handling
    - Callback execution in _on_mqtt_disconnect with error handling
    - Async vs sync callback detection and appropriate execution
    
  - **Enhanced Integration Features Tests:**
    - update_device_availability() with discovery publisher dependency
    - handle_service_command() routing for different service types
    - handle_service_command() manual_retrain with parameter validation
    - handle_service_command() refresh_discovery with result aggregation
    - handle_service_command() reset_statistics with stats object reset
    - handle_service_command() force_prediction with room_id validation
    - handle_service_command() unknown service graceful handling
    - cleanup_discovery() with optional entity_ids parameter
    
  - **Private Method Tests:**
    - _system_status_publishing_loop() background task execution
    - _system_status_publishing_loop() timeout handling and retry logic
    - _system_status_publishing_loop() cancellation and graceful shutdown
    - _check_system_availability() connection and integration status
    - _handle_entity_state_change() callback notification execution
    - _handle_entity_state_change() logging and error handling

- Integration Tests:
  - **Full Integration Lifecycle:**
    - Complete initialization → start_integration → publishing → stop_integration cycle
    - Real MQTT publisher integration with connection callbacks
    - Discovery publisher initialization with entity publishing
    - Background task coordination with system status publishing
    - Graceful shutdown with all components stopping properly
    
  - **Publishing Pipeline Integration:**
    - PredictionResult publishing through prediction_publisher
    - System status publishing with real statistics aggregation
    - Discovery message publishing and refresh operations
    - Device availability updates with Home Assistant integration
    
  - **Configuration Integration:**
    - Global config loading integration with get_config()
    - MQTT configuration validation and publisher setup
    - Room configuration integration with discovery publishing
    - Publishing enabled/disabled configuration handling
    
  - **Error Propagation Testing:**
    - Component initialization failures and error handling
    - Publishing failures with statistics tracking
    - Background task failures and recovery mechanisms
    - Callback execution errors with graceful degradation
    
  - **Service Command Integration:**
    - Home Assistant service command handling end-to-end
    - Manual retrain integration with tracking system
    - Discovery refresh coordination with publisher components
    - Statistics reset with full system state clearing

- Edge Cases:
  - **Concurrent Operation Handling:**
    - Multiple simultaneous publish_prediction() calls
    - Background task shutdown during active publishing
    - Callback execution during component initialization
    - Statistics updates during get_integration_stats() execution
    
  - **Resource Management Edge Cases:**
    - Large background task lists with cleanup operations
    - Memory usage with extensive statistics tracking
    - Long-running status publishing loops with interruption
    - Publisher component cleanup during active operations
    
  - **Configuration Edge Cases:**
    - Empty rooms dictionary handling
    - Invalid MQTT configuration with initialization failures
    - Missing discovery publisher with method calls
    - Disabled publishing with active integration attempts
    
  - **State Transition Edge Cases:**
    - Initialization failure with partial component setup
    - Integration start with already running background tasks
    - Stop integration without prior initialization
    - Publishing calls before integration is active
    
  - **Callback Execution Edge Cases:**
    - Callback list modification during iteration
    - Exception in callback not affecting other callbacks
    - Mixed async/sync callbacks in notification list
    - Callback removal during execution loop

- Error Handling:
  - **Initialization Error Handling:**
    - MQTTIntegrationError raising with proper error context
    - Component initialization failures with cleanup
    - Configuration loading errors with detailed messages
    - Discovery publishing failures with partial success handling
    
  - **Publishing Error Handling:**
    - Prediction publishing failures with statistics tracking
    - System status publishing errors with retry mechanisms
    - Discovery refresh failures with individual result tracking
    - Device availability update failures with informative logging
    
  - **Background Task Error Handling:**
    - System status publishing loop exception handling
    - Task cancellation during graceful shutdown
    - Background task creation failures with recovery
    - Timeout handling in status publishing intervals
    
  - **Callback Error Handling:**
    - MQTT connection callback exceptions with isolation
    - Notification callback failures with continued execution
    - Entity state change callback errors with logging
    - Service command callback exceptions with graceful degradation
    
  - **Service Command Error Handling:**
    - Unknown service command handling with warning logs
    - Invalid command data validation with error responses
    - Missing required parameters with informative errors
    - Service execution failures with boolean result indication
    
  - **Component Dependency Error Handling:**
    - Missing MQTT publisher with graceful method returns
    - Unavailable prediction publisher with early exit
    - Discovery publisher dependency failures with fallback
    - Publisher stats collection failures with empty results
    
  - **Resource Management Error Handling:**
    - Background task cleanup failures with continued shutdown
    - Statistics calculation errors with default values
    - Memory allocation failures with graceful degradation
    - Component state inconsistencies with recovery attempts

- Coverage Target: 85%+

### src/integration/ha_tracking_bridge.py - HA Tracking Bridge
**Classes Found:** HATrackingBridgeStats, HATrackingBridge, HATrackingBridgeError
**Methods Analyzed:** __init__, initialize, shutdown, handle_prediction_made, handle_accuracy_alert, handle_drift_detected, handle_retraining_started, handle_retraining_completed, get_bridge_stats, _setup_tracking_event_handlers, _setup_command_delegation, _start_background_tasks, _system_status_sync_loop, _metrics_sync_loop, _update_system_alert_status, _update_system_drift_status, _delegate_retrain_model, _delegate_validate_model, _delegate_force_prediction, _delegate_check_database, _delegate_generate_diagnostic

**Required Tests:**
- Unit Tests:
  - **HATrackingBridgeStats dataclass validation:**
    - Default field initialization with proper types
    - DateTime field assignments and None defaults
    - Counter field increments (entity_updates_sent, commands_delegated)
    - Error tracking fields (bridge_errors, last_error)
    - Bridge state tracking (bridge_initialized, bridge_active)
    
  - **HATrackingBridge initialization and setup:**
    - Constructor with required dependencies (tracking_manager, enhanced_integration_manager)
    - Stats initialization and bridge state setup
    - Background tasks list initialization
    - Event handlers dictionary initialization
    - Shutdown event creation
    
  - **Bridge lifecycle management:**
    - Initialize method with event handler setup
    - Command delegation configuration
    - Background task startup sequence
    - Bridge activation state transitions
    - Shutdown method with proper cleanup
    - Task cancellation and graceful termination
    
  - **Event handling methods:**
    - handle_prediction_made with PredictionResult processing
    - handle_accuracy_alert with alert data extraction
    - handle_drift_detected with drift metrics processing
    - handle_retraining_started with training status updates
    - handle_retraining_completed with result processing
    - Bridge active state checks in all handlers
    
  - **Statistics and monitoring:**
    - get_bridge_stats method returning complete stats
    - Stats counter increments in event handlers
    - Error tracking and last error recording
    - Timestamp recording for last operations
    - Background task count reporting
    
  - **Command delegation methods:**
    - _delegate_retrain_model with room_id and force parameters
    - _delegate_validate_model with days parameter validation
    - _delegate_force_prediction with room_id handling
    - _delegate_check_database with health check delegation
    - _delegate_generate_diagnostic with parameter processing
    - hasattr() checks for TrackingManager method availability
    
  - **Background task management:**
    - _start_background_tasks creation and registration
    - _system_status_sync_loop with periodic updates
    - _metrics_sync_loop with accuracy synchronization
    - Shutdown event handling in background loops
    - Task error handling and recovery mechanisms
    
  - **HA entity update methods:**
    - _update_system_alert_status with alert data
    - _update_system_drift_status with drift information
    - Entity state updates through enhanced integration manager
    - Error handling in update operations

- Integration Tests:
  - **Bridge-TrackingManager Integration:**
    - Event handler registration with actual TrackingManager
    - Command delegation to real TrackingManager methods
    - Callback registration and event firing
    - Method availability checks (hasattr) with real objects
    - System status retrieval and processing
    
  - **Bridge-EnhancedIntegrationManager Integration:**
    - Entity state updates through integration manager
    - Command handler override and delegation
    - Prediction update handling with real prediction results
    - System status update propagation
    - Discovery and MQTT integration through manager
    
  - **Background Task Integration:**
    - System status sync with actual tracking manager
    - Metrics sync with real accuracy data
    - Task coordination during shutdown
    - Error propagation between tasks
    - Resource cleanup verification
    
  - **Event Flow Integration:**
    - End-to-end prediction event handling
    - Alert propagation from tracking to HA
    - Drift detection notification flow
    - Retraining event lifecycle (start/complete)
    - Statistics accumulation across event types
    
  - **Command Processing Integration:**
    - HA command reception and delegation
    - Parameter validation and transformation
    - Result formatting and response handling
    - Error propagation from tracking manager
    - Fallback behavior for unsupported commands

- Edge Cases:
  - **Initialization Edge Cases:**
    - TrackingManager without callback support (missing register_callback)
    - EnhancedIntegrationManager with missing command handlers
    - Background task creation failures
    - Event handler setup failures with partial registration
    - Bridge initialization with already active state
    
  - **Event Handling Edge Cases:**
    - Events received when bridge is inactive
    - Malformed event objects with missing attributes
    - Concurrent event processing during shutdown
    - Event handler exceptions during processing
    - Large volume event processing stress
    
  - **Command Delegation Edge Cases:**
    - Commands with missing or invalid parameters
    - TrackingManager methods not available (hasattr false)
    - Command execution timeout scenarios
    - Concurrent command execution
    - Command parameters with edge case values (None, empty, negative)
    
  - **Background Task Edge Cases:**
    - Sync loop exceptions with recovery
    - TrackingManager unavailable during sync
    - Network failures during HA entity updates
    - Shutdown during active sync operations
    - Memory pressure during long-running tasks
    
  - **Statistics and State Edge Cases:**
    - Counter overflow with very high values
    - Concurrent stats updates from multiple threads
    - Stats access during bridge reset
    - Error state persistence across operations
    - Timestamp handling across timezone changes

- Error Handling:
  - **HATrackingBridgeError Exception:**
    - Custom error with OccupancyPredictionError inheritance
    - Error code and severity parameter handling
    - Error propagation with context information
    - Error formatting with proper message structure
    
  - **Initialization Error Handling:**
    - TrackingManager dependency failures
    - EnhancedIntegrationManager setup failures
    - Event handler registration errors
    - Command delegation setup failures
    - Background task startup errors with rollback
    
  - **Event Processing Error Handling:**
    - Exception handling in all event handlers
    - Error statistics tracking and reporting
    - Graceful degradation when handlers fail
    - Error state recovery mechanisms
    - Event queue overflow handling
    
  - **Command Execution Error Handling:**
    - Parameter validation error responses
    - TrackingManager method execution failures
    - Timeout handling for long-running commands
    - Error response formatting consistency
    - Fallback behavior for unsupported operations
    
  - **Background Task Error Handling:**
    - Sync loop exception recovery
    - Task failure isolation (one task failure doesn't affect others)
    - Resource cleanup on task failures
    - Error logging with context information
    - Graceful shutdown on critical errors
    
  - **Integration Error Handling:**
    - HA entity update failures with retry logic
    - MQTT publishing errors through integration manager
    - Discovery service failures
    - Database connectivity issues
    - Network interruption recovery mechanisms
    
  - **Resource Management Error Handling:**
    - Memory leak prevention in long-running tasks
    - Task cancellation timeout handling
    - Resource cleanup verification
    - Deadlock prevention in concurrent operations
    - Graceful degradation under resource pressure

- Coverage Target: 85%+

### src/integration/tracking_integration.py - Tracking Integration
**Classes Found:** IntegrationConfig, TrackingIntegrationManager, TrackingIntegrationError
**Methods Analyzed:** __init__, initialize, shutdown, get_websocket_handler, get_sse_handler, get_integration_stats, add_realtime_callback, remove_realtime_callback, _integrate_with_tracking_manager, _start_integration_tasks, _system_status_broadcast_loop, _connection_monitoring_loop, _handle_alert_broadcast, integrate_tracking_with_realtime_publishing, create_integrated_tracking_manager

**Required Tests:**

- **Unit Tests:**
  - **IntegrationConfig Class:**
    - Test default configuration values initialization
    - Test configuration parameter validation and bounds
    - Test serialization/deserialization for configuration storage
    - Test configuration merging and overriding behaviors
    - Test invalid configuration parameter handling
    
  - **TrackingIntegrationManager Initialization:**
    - Test successful initialization with valid tracking_manager and config
    - Test initialization with None integration_config (uses defaults)
    - Test initialization state validation (integration_active, background_tasks)
    - Test system configuration loading and validation
    - Test logger setup and initial logging messages
    
  - **TrackingIntegrationManager Core Methods:**
    - Test initialize() method with all channels enabled
    - Test initialize() method with selective channel enablement
    - Test initialize() method with real-time publishing disabled
    - Test initialize() method error handling and rollback
    - Test shutdown() method with active background tasks
    - Test shutdown() method with no active components
    - Test shutdown() method error handling during cleanup
    
  - **Handler Methods:**
    - Test get_websocket_handler() returns correct handler when MQTT manager exists
    - Test get_websocket_handler() returns None when MQTT manager doesn't exist
    - Test get_sse_handler() returns correct handler when MQTT manager exists
    - Test get_sse_handler() returns None when MQTT manager doesn't exist
    
  - **Statistics and Management:**
    - Test get_integration_stats() with fully active integration
    - Test get_integration_stats() with partial integration setup
    - Test get_integration_stats() error handling when components fail
    - Test add_realtime_callback() delegation to MQTT manager
    - Test remove_realtime_callback() delegation to MQTT manager
    - Test callback operations when MQTT manager is None
    
  - **Private Method Testing:**
    - Test _integrate_with_tracking_manager() successful integration
    - Test _integrate_with_tracking_manager() with alert broadcasting enabled
    - Test _integrate_with_tracking_manager() error handling
    - Test _start_integration_tasks() task creation and management
    - Test _handle_alert_broadcast() message formatting and publishing
    - Test _handle_alert_broadcast() when broadcasting disabled or no MQTT manager

- **Integration Tests:**
  - **End-to-End Integration Setup:**
    - Test complete integration flow from tracking manager to real-time publishing
    - Test integration with real TrackingManager instance
    - Test integration with real EnhancedMQTTIntegrationManager
    - Test WebSocket and SSE server integration through handlers
    - Test MQTT publishing integration through enhanced manager
    
  - **Background Task Integration:**
    - Test _system_status_broadcast_loop() with real tracking manager
    - Test _connection_monitoring_loop() with real connection monitoring
    - Test background task lifecycle (start, run, shutdown, cleanup)
    - Test multiple background tasks running concurrently
    - Test task coordination during shutdown sequence
    
  - **Real-time Publishing Integration:**
    - Test alert broadcasting through real-time channels
    - Test system status publishing across multiple channels
    - Test callback chain execution from tracking manager to publishers
    - Test message flow from tracking events to WebSocket/SSE clients
    - Test MQTT topic publishing with proper payload formatting
    
  - **Factory Function Integration:**
    - Test integrate_tracking_with_realtime_publishing() complete flow
    - Test create_integrated_tracking_manager() factory function
    - Test integration with different tracking configurations
    - Test integration manager lifecycle through factory functions

- **Edge Cases:**
  - **Configuration Edge Cases:**
    - Test integration with extreme configuration values (timeouts, limits)
    - Test integration with zero/negative interval values
    - Test integration with very high connection limits
    - Test configuration changes during runtime
    - Test partial configuration scenarios
    
  - **State Management Edge Cases:**
    - Test multiple initialize() calls (idempotency)
    - Test shutdown() before initialize()
    - Test shutdown() called multiple times
    - Test operations on shutdown integration manager
    - Test state consistency during concurrent operations
    
  - **Background Task Edge Cases:**
    - Test system status broadcast with slow tracking manager responses
    - Test connection monitoring with rapidly changing connection counts
    - Test task behavior during rapid shutdown/restart cycles
    - Test background task resilience to temporary failures
    - Test task cleanup when individual tasks hang or become unresponsive
    
  - **Handler Edge Cases:**
    - Test handler methods during integration state changes
    - Test handler access patterns during initialization/shutdown
    - Test handler behavior with null/invalid MQTT manager states
    - Test concurrent access to handlers from multiple clients
    
  - **Statistics and Monitoring Edge Cases:**
    - Test statistics collection during integration state transitions
    - Test statistics accuracy with rapidly changing system state
    - Test statistics collection resilience to individual component failures
    - Test statistics method calls during shutdown processes

- **Error Handling:**
  - **Initialization Error Handling:**
    - Test TrackingIntegrationError propagation during initialize()
    - Test recovery from EnhancedMQTTIntegrationManager initialization failures
    - Test handling of invalid tracking_manager references
    - Test system configuration loading failures
    - Test circular dependency resolution in imports
    
  - **Background Task Error Handling:**
    - Test _system_status_broadcast_loop() exception recovery
    - Test _connection_monitoring_loop() exception recovery
    - Test task cancellation timeout handling
    - Test task failure isolation (one task failure doesn't affect others)
    - Test error logging with proper context information
    
  - **Integration Error Handling:**
    - Test tracking manager integration failures
    - Test MQTT manager integration failures during runtime
    - Test callback registration/removal error scenarios
    - Test alert broadcasting failures with graceful degradation
    - Test system status publishing failures
    
  - **Shutdown Error Handling:**
    - Test shutdown with unresponsive background tasks
    - Test shutdown with failed MQTT manager cleanup
    - Test shutdown error logging and final state consistency
    - Test resource cleanup verification after shutdown errors
    - Test graceful degradation during partial shutdown failures
    
  - **Factory Function Error Handling:**
    - Test integrate_tracking_with_realtime_publishing() error propagation
    - Test create_integrated_tracking_manager() error handling
    - Test factory function error recovery and cleanup
    - Test error scenarios during TrackingManager creation
    - Test integration failures with proper resource cleanup
    
  - **Connection and Resource Error Handling:**
    - Test connection limit enforcement and overflow handling
    - Test connection monitoring resilience to connection failures
    - Test resource cleanup during unexpected disconnections
    - Test memory management during high connection scenarios
    - Test graceful degradation under resource pressure

- Coverage Target: 85%+

### src/integration/prediction_publisher.py - Prediction Publisher
**Classes Found:** PredictionPayload, SystemStatusPayload, PredictionPublisher
**Methods Analyzed:** __init__, publish_prediction, publish_system_status, publish_room_batch, get_publisher_stats, _publish_legacy_topics, _format_time_until, _calculate_reliability, _determine_system_status

**Required Tests:**
- Unit Tests:
  - **PredictionPayload dataclass tests:**
    - Validate all required fields are present
    - Test field type conversions (datetime to ISO string)
    - Test dataclass serialization via asdict()
    - Verify default values (system_version, last_updated)
    - Test with None alternatives list
  - **SystemStatusPayload dataclass tests:**
    - Validate all required fields are present
    - Test Optional field handling (last_prediction_time, last_error)
    - Test dataclass serialization via asdict()
    - Verify numeric field types (int, float)
  - **PredictionPublisher.__init__() tests:**
    - Test initialization with valid MQTT publisher, config, and rooms
    - Verify statistics initialization (predictions_published=0, etc.)
    - Test room name cache creation from room configs
    - Verify system_start_time is set to current UTC time
    - Test logger initialization message
  - **publish_prediction() tests:**
    - Test successful prediction publishing with all required fields
    - Test room name lookup from cache and fallback formatting
    - Test time calculation and formatting (time_until_seconds)
    - Test human readable time formatting (_format_time_until integration)
    - Test reliability calculation based on confidence score
    - Test base_predictions and model_weights extraction from metadata
    - Test alternatives formatting (limit to 3, proper time calculations)
    - Test topic generation with config.topic_prefix
    - Test QoS and retain settings from config
    - Test statistics updates (predictions_published increment)
    - Test legacy topics publishing
    - Test with None current_state parameter
    - Test with missing room in cache (fallback name generation)
  - **publish_system_status() tests:**
    - Test successful status publishing with all fields
    - Test uptime calculation from system_start_time
    - Test with None tracking_stats and model_stats parameters
    - Test tracking statistics extraction (total_predictions, etc.)
    - Test model statistics extraction (active_models, trained, failed)
    - Test system status determination logic
    - Test optional parameters (database_connected, active_alerts, last_error)
    - Test topic generation for system status
    - Test QoS and retain settings for system messages
    - Test statistics updates (status_updates_published increment)
  - **publish_room_batch() tests:**
    - Test batch publishing with multiple rooms
    - Test with empty predictions dictionary
    - Test with None current_states parameter
    - Test mixed success/failure results handling
    - Test success counting and logging
    - Verify individual publish_prediction calls for each room
  - **get_publisher_stats() tests:**
    - Test statistics dictionary structure and content
    - Test timestamp formatting (ISO format)
    - Test with None last_prediction_time
    - Verify mqtt_publisher.get_publisher_stats() integration
    - Test rooms count from configuration
  - **_publish_legacy_topics() tests:**
    - Test all legacy topic publications (next_transition_time, transition_type, confidence, time_until)
    - Test topic path construction with base_topic
    - Test payload formatting for each legacy topic
    - Test QoS and retain settings inheritance
    - Test partial failure handling (some topics succeed, others fail)
  - **_format_time_until() tests:**
    - Test seconds formatting (< 60 seconds)
    - Test minutes formatting (< 3600 seconds, singular/plural)
    - Test hours and minutes formatting (< 86400 seconds)
    - Test days and hours formatting (>= 86400 seconds)
    - Test edge cases (0 seconds, exactly 60 seconds, exactly 3600 seconds)
    - Test large values (multiple days)
  - **_calculate_reliability() tests:**
    - Test "high" reliability (confidence >= 0.8)
    - Test "medium" reliability (0.6 <= confidence < 0.8)
    - Test "low" reliability (confidence < 0.6)
    - Test edge cases (exactly 0.8, exactly 0.6)
    - Test extreme values (0.0, 1.0)
  - **_determine_system_status() tests:**
    - Test "offline" status (mqtt_connected=False or database_connected=False)
    - Test "degraded" status (active_alerts > 5 or models_failed > 0)
    - Test "online" status (all systems healthy)
    - Test edge cases (exactly 5 active_alerts, exactly 0 models_failed)

- Integration Tests:
  - **MQTT Publisher Integration:**
    - Test actual MQTT message publishing with real MQTTPublisher instance
    - Verify message payload structure and JSON serialization
    - Test topic hierarchy creation and message routing
    - Test QoS and retain flag handling in published messages
    - Test connection status integration with system status
  - **Configuration Integration:**
    - Test with real MQTTConfig and RoomConfig objects
    - Verify topic prefix usage across all publishing methods
    - Test QoS settings inheritance from configuration
    - Test retain settings for different message types
  - **PredictionResult Integration:**
    - Test with real PredictionResult objects from model predictions
    - Verify metadata extraction (base_model_predictions, model_weights)
    - Test alternatives list processing and formatting
    - Test features_used count integration
  - **Room Configuration Integration:**
    - Test room name resolution from actual room configurations
    - Test with multiple rooms having different configurations
    - Verify room cache updates when configurations change
  - **DateTime Handling Integration:**
    - Test timezone-aware datetime handling (UTC consistency)
    - Test prediction time calculations with real datetime objects
    - Verify ISO format timestamp generation across all payloads
    - Test time_until calculations with past/future predictions

- Edge Cases:
  - **Prediction Time Edge Cases:**
    - Test with prediction_time in the past (negative time_until)
    - Test with very distant future predictions (years ahead)
    - Test with prediction_time exactly equal to current time
    - Test timezone handling edge cases (DST transitions)
  - **Payload Field Edge Cases:**
    - Test with empty strings in required fields
    - Test with very long room names or model types
    - Test with extreme confidence values (0.0, 1.0, negative, > 1.0)
    - Test with None values in optional fields
    - Test with empty lists/dictionaries (alternatives, base_predictions)
  - **Statistics Edge Cases:**
    - Test counter overflow scenarios (very large prediction counts)
    - Test with None last_prediction_time throughout lifecycle
    - Test system_start_time edge cases (future time, very old time)
  - **Room Configuration Edge Cases:**
    - Test with empty rooms dictionary
    - Test with room_id not in configuration
    - Test with room names containing special characters
    - Test room cache updates during runtime
  - **Batch Processing Edge Cases:**
    - Test batch processing with single room
    - Test with very large batch sizes (performance considerations)
    - Test with duplicate room_ids in batch
    - Test batch with all failures vs all successes

- Error Handling:
  - **MQTT Publishing Errors:**
    - Test MQTTPublisher.publish_json() failure handling
    - Test network disconnection during publishing
    - Test MQTT broker unavailability
    - Test malformed JSON serialization errors
    - Test topic validation failures
  - **Data Processing Errors:**
    - Test invalid PredictionResult object handling
    - Test missing required fields in prediction metadata
    - Test datetime parsing/formatting exceptions
    - Test JSON serialization errors with complex objects
    - Test asdict() failures with custom dataclasses
  - **Configuration Errors:**
    - Test with invalid MQTT configuration
    - Test with missing topic_prefix
    - Test with invalid QoS values
    - Test with malformed room configurations
  - **Memory and Resource Errors:**
    - Test large payload handling (memory constraints)
    - Test concurrent publishing error handling
    - Test resource cleanup on failures
    - Test graceful degradation with partial system failures
  - **Exception Propagation:**
    - Test exception handling in all async methods
    - Verify proper error logging with contextual information
    - Test MQTTPublishResult error message population
    - Test exception masking prevention (no silent failures)
    - Test timeout handling in async operations

- Coverage Target: 85%+

### src/utils/metrics.py - Performance Metrics
**Classes Found:** 
- MLMetricsCollector (main metrics collection class)
- MetricsManager (centralized metrics management)
- MultiProcessMetricsManager (multi-process metrics support)

**Methods Analyzed:**
- MLMetricsCollector: __init__, _setup_metrics, update_system_info, record_prediction, record_model_training, record_concept_drift, record_event_processing, record_feature_computation, record_database_operation, record_mqtt_publish, record_ha_api_request, update_ha_connection_status, record_error, update_system_resources, update_active_models_count, update_prediction_queue_size, update_system_health_score, update_uptime, set_gauge, time_operation
- MetricsManager: __init__, start_background_collection, stop_background_collection, get_metrics, get_collector
- MultiProcessMetricsManager: __init__, is_multiprocess_enabled, get_multiprocess_registry, aggregate_multiprocess_metrics, generate_multiprocess_metrics, cleanup_dead_processes
- Module functions: get_metrics_manager, get_metrics_collector, metrics_endpoint_handler, time_prediction, get_multiprocess_metrics_manager, setup_multiprocess_metrics, get_aggregated_metrics, export_multiprocess_metrics

**Required Tests:**

- **Unit Tests:**
  
  - **MLMetricsCollector Tests:**
    - Test __init__ with custom registry vs default REGISTRY
    - Test _setup_metrics creates all expected Prometheus metrics (32+ metrics)
    - Test update_system_info sets system information exactly once
    - Test record_prediction with all parameter combinations (room_id, prediction_type, model_type, duration, accuracy_minutes, confidence, status)
    - Test record_prediction with optional parameters (accuracy_minutes=None, confidence=None)
    - Test record_model_training with duration tracking and accuracy metrics
    - Test record_model_training with missing accuracy_metrics parameter
    - Test record_concept_drift with severity thresholds (high >0.7, medium >0.3, low <=0.3)
    - Test record_event_processing with different sensor types and statuses
    - Test record_feature_computation timing measurements
    - Test record_database_operation with various operation types and tables
    - Test record_mqtt_publish with different topic types and statuses
    - Test record_ha_api_request with different endpoints, methods, and statuses
    - Test update_ha_connection_status boolean conversion (True=1, False=0)
    - Test record_error with timestamp recording using time.time()
    - Test update_system_resources with psutil integration (CPU, memory, disk)
    - Test update_system_resources exception handling (silent failures)
    - Test update_active_models_count gauge updates
    - Test update_prediction_queue_size for different queue types
    - Test update_system_health_score with bounds checking (0.0-1.0)
    - Test update_uptime with datetime calculations
    - Test set_gauge with gauge_map lookup and label handling
    - Test set_gauge with unknown gauge names (silent failure)
    - Test set_gauge exception handling (silent failures)
    - Test time_operation context manager yield and cleanup

  - **MetricsManager Tests:**
    - Test __init__ with custom registry vs default
    - Test start_background_collection thread creation and daemon mode
    - Test start_background_collection prevents multiple threads (_running flag)
    - Test stop_background_collection thread joining with timeout
    - Test background update loop with update_interval timing
    - Test background thread exception handling (silent continues)
    - Test get_metrics Prometheus format output with/without prometheus_client
    - Test get_collector returns MLMetricsCollector instance
    - Test start_time initialization and uptime tracking

  - **MultiProcessMetricsManager Tests:**
    - Test __init__ with prometheus_client availability detection
    - Test __init__ without prometheus_client (multiprocess_enabled=False)
    - Test is_multiprocess_enabled boolean return
    - Test get_multiprocess_registry returns CollectorRegistry or None
    - Test aggregate_multiprocess_metrics with metric families collection
    - Test aggregate_multiprocess_metrics error handling
    - Test generate_multiprocess_metrics Prometheus format output
    - Test generate_multiprocess_metrics error handling
    - Test cleanup_dead_processes with multiprocess.mark_process_dead
    - Test cleanup_dead_processes exception handling

  - **Module Function Tests:**
    - Test get_metrics_manager singleton pattern (same instance returned)
    - Test get_metrics_collector convenience function
    - Test metrics_endpoint_handler returns metrics string
    - Test time_prediction decorator with successful function calls
    - Test time_prediction decorator with exception handling
    - Test time_prediction decorator confidence extraction from result dict
    - Test get_multiprocess_metrics_manager singleton pattern
    - Test setup_multiprocess_metrics initialization sequence
    - Test setup_multiprocess_metrics registry assignment
    - Test get_aggregated_metrics multiprocess vs single process modes
    - Test export_multiprocess_metrics format consistency

- **Integration Tests:**
  
  - **Prometheus Integration Tests:**
    - Test metrics collection with real prometheus_client library
    - Test metrics export in proper Prometheus format
    - Test counter increment operations across multiple calls
    - Test gauge set/inc/dec operations with label combinations
    - Test histogram observe operations with bucket distributions
    - Test metrics registry isolation between test runs
    - Test multiprocess metrics aggregation across mock processes
    
  - **System Resource Integration Tests:**
    - Test psutil integration for CPU/memory/disk metrics
    - Test system resource updates in background thread
    - Test resource metric accuracy against actual system values
    - Test resource collection performance impact
    
  - **Background Thread Integration Tests:**
    - Test background metrics collection lifecycle
    - Test thread cleanup on manager destruction
    - Test concurrent access to metrics from multiple threads
    - Test update_interval timing accuracy
    - Test graceful shutdown behavior

- **Edge Cases:**
  
  - **Missing Dependencies:**
    - Test behavior when prometheus_client import fails
    - Test mock class functionality when prometheus unavailable
    - Test multiprocess features when multiprocess module missing
    - Test psutil availability for system resource collection
    
  - **Invalid Parameters:**
    - Test record_prediction with negative duration values
    - Test record_prediction with invalid confidence values (outside 0-1)
    - Test update_system_health_score with out-of-bounds values
    - Test set_gauge with None values and empty labels
    - Test record operations with None/empty room_id parameters
    
  - **Resource Constraints:**
    - Test behavior under high memory pressure
    - Test background thread behavior with limited CPU
    - Test metrics collection with disk space constraints
    - Test concurrent access patterns under load
    
  - **Threading Edge Cases:**
    - Test background thread start/stop race conditions
    - Test multiple start_background_collection calls
    - Test stop_background_collection timeout scenarios
    - Test daemon thread behavior on process exit
    
  - **Large Scale Scenarios:**
    - Test metrics collection with many rooms (100+ rooms)
    - Test high-frequency metric updates (1000+ ops/second)
    - Test memory usage with long-running metric collection
    - Test performance impact of extensive label combinations

- **Error Handling:**
  
  - **Prometheus Client Errors:**
    - Test metric creation failures (duplicate names, invalid labels)
    - Test metric update failures (registry issues)
    - Test export generation failures
    - Test multiprocess collector failures
    
  - **System Resource Errors:**
    - Test psutil process access failures
    - Test system resource unavailability
    - Test permission errors accessing system metrics
    - Test invalid process ID scenarios
    
  - **Threading Errors:**
    - Test background thread creation failures
    - Test thread interruption handling
    - Test resource cleanup on thread failures
    - Test concurrent modification exceptions
    
  - **Registry Errors:**
    - Test custom registry initialization failures
    - Test metric registration conflicts
    - Test registry cleanup on errors
    - Test multiprocess registry setup failures
    
  - **Mock Implementation Errors:**
    - Test mock class method calls with invalid parameters
    - Test mock context manager behavior
    - Test mock class initialization edge cases
    - Test fallback behavior consistency

- Coverage Target: 85%+

### src/integration/realtime_publisher.py - Real-time Publisher
**Classes Found:** PublishingChannel (Enum), ClientConnection, PublishingMetrics, RealtimePredictionEvent, WebSocketConnectionManager, SSEConnectionManager, RealtimePublishingSystem, RealtimePublishingError
**Methods Analyzed:** 85+ methods across multiple classes including connection management, broadcasting, metrics tracking, and error handling

**Required Tests:**
- Unit Tests:
  - RealtimePredictionEvent.to_websocket_message() - test JSON format and data serialization
  - RealtimePredictionEvent.to_sse_message() - test SSE format with proper headers
  - ClientConnection.update_activity() - test timestamp updating
  - PublishingMetrics.__post_init__() - test default channel_errors initialization
  - WebSocketConnectionManager.connect() - test client ID generation and connection storage
  - WebSocketConnectionManager.disconnect() - test connection cleanup and metadata removal  
  - WebSocketConnectionManager.subscribe_to_room() - test room subscription management
  - WebSocketConnectionManager.unsubscribe_from_room() - test room unsubscription
  - WebSocketConnectionManager.broadcast_to_room() - test targeted room broadcasting
  - WebSocketConnectionManager.broadcast_to_all() - test global broadcasting
  - WebSocketConnectionManager.get_connection_stats() - test statistics calculation
  - SSEConnectionManager.connect() - test queue creation and client registration
  - SSEConnectionManager.disconnect() - test queue cleanup
  - SSEConnectionManager.subscribe_to_room() - test room subscription for SSE
  - SSEConnectionManager.broadcast_to_room() - test SSE room broadcasting
  - SSEConnectionManager.broadcast_to_all() - test global SSE broadcasting
  - SSEConnectionManager.get_connection_stats() - test SSE statistics
  - RealtimePublishingSystem.__init__() - test configuration loading and defaults
  - RealtimePublishingSystem.initialize() - test background task creation and system startup
  - RealtimePublishingSystem.shutdown() - test clean shutdown and task cancellation
  - RealtimePublishingSystem.publish_prediction() - test multi-channel prediction publishing
  - RealtimePublishingSystem.publish_system_status() - test status broadcasting
  - RealtimePublishingSystem._format_prediction_data() - test prediction data formatting
  - RealtimePublishingSystem._format_time_until() - test human-readable time formatting
  - RealtimePublishingSystem._handle_websocket_message() - test message routing logic
  - RealtimePublishingSystem.add_broadcast_callback() - test callback registration
  - RealtimePublishingSystem.remove_broadcast_callback() - test callback removal
  - RealtimePublishingSystem.get_publishing_stats() - test comprehensive statistics
  - RealtimePublishingSystem.handle_websocket_connection() - test WebSocket connection handling
  - RealtimePublishingSystem.create_sse_stream() - test SSE stream creation
- Integration Tests:
  - Multi-channel publishing workflow - test MQTT, WebSocket, and SSE coordination
  - Client subscription management across connection managers
  - Background task coordination during system lifecycle
  - Real WebSocket connection simulation with message exchange
  - SSE stream functionality with real event queuing
  - Connection cleanup during high client turnover
  - Metric accuracy during concurrent operations
  - System shutdown with active connections
- Edge Cases:
  - Connection failures during broadcast operations
  - Invalid JSON messages in WebSocket handlers
  - Client disconnection during active streaming
  - Memory management with weak reference callbacks
  - Queue overflow in SSE connections
  - Concurrent subscription/unsubscription operations
  - System shutdown with pending background tasks
  - Large number of simultaneous connections
  - Network interruption during real-time publishing
  - Malformed prediction data handling
- Error Handling:
  - MQTT publishing failures and fallback behavior
  - WebSocket connection drops with proper cleanup
  - SSE stream interruption and client notification
  - Background task exception handling without system crash
  - Invalid room ID subscription attempts
  - Callback exception isolation
  - Resource exhaustion during high load
  - Configuration errors during initialization
  - Database connectivity issues during publishing
  - Timeout handling in SSE keepalive mechanism
- Coverage Target: 85%+

### src/utils/logger.py - Structured Logging
**Classes Found:** StructuredFormatter, PerformanceLogger, ErrorTracker, MLOperationsLogger, LoggerManager
**Methods Analyzed:** format, log_operation_time, log_prediction_accuracy, log_model_metrics, log_resource_usage, track_error, track_prediction_error, track_data_error, track_integration_error, log_training_event, log_drift_detection, log_model_deployment, log_feature_importance, _setup_logging, get_logger, get_performance_logger, get_error_tracker, get_ml_ops_logger, log_operation

**Required Tests:**
- Unit Tests:
  - **StructuredFormatter Tests:**
    - JSON formatting with all standard fields (timestamp, level, logger, message, etc.)
    - Exception information inclusion in log entries (type, message, traceback)
    - Extra fields handling with include_extra=True/False
    - Standard field filtering (excluding built-in logging fields)
    - UTC timestamp formatting validation
    - JSON serialization with default=str for non-serializable objects
    - Thread and process information inclusion
    
  - **PerformanceLogger Tests:**
    - Operation timing logging with structured metadata
    - Prediction accuracy logging with room_id and confidence
    - Model metrics logging with various metric types
    - Resource usage logging (CPU, memory, disk)
    - Optional parameter handling (room_id, prediction_type, kwargs)
    - Extra keyword arguments propagation
    
  - **ErrorTracker Tests:**
    - Generic error tracking with context and severity levels
    - Prediction-specific error tracking with room and model context
    - Data ingestion error tracking with data source information
    - Integration error tracking (MQTT, HA API endpoints)
    - Critical vs standard error logging level selection
    - Alert flag handling for error notifications
    - Exception info inclusion (exc_info=True)
    
  - **MLOperationsLogger Tests:**
    - Training event logging with metrics and event types
    - Concept drift detection logging with severity and actions
    - Model deployment logging with version and performance metrics
    - Feature importance analysis logging
    - ML lifecycle event categorization
    - Component identification (training, adaptation, deployment, analysis)
    
  - **LoggerManager Tests:**
    - YAML configuration loading from config/logging.yaml
    - Fallback configuration when YAML file missing
    - JSON formatter injection into file handlers
    - Logs directory creation
    - Logger instance creation with consistent naming
    - Global singleton instance management
    - Context manager operation timing with success/failure scenarios
    - Operation start/complete event logging
    
  - **Module-Level Functions Tests:**
    - get_logger_manager singleton behavior
    - Convenience function delegation (get_logger, get_performance_logger, etc.)
    - Global instance initialization and reuse

- Integration Tests:
  - **Logging Configuration Integration:**
    - End-to-end YAML config loading with real logging.yaml file
    - JSON formatter integration with file and console handlers
    - Log rotation configuration with TimedRotatingFileHandler
    - Multiple logger instances with consistent configuration
    - Cross-component logger coordination (performance, error, ml_ops)
    
  - **Performance Monitoring Integration:**
    - Operation timing context manager with real operations
    - Performance metrics correlation across components
    - Resource usage monitoring integration with system metrics
    - Multi-threaded logging performance validation
    
  - **Error Tracking Integration:**
    - Error propagation across system components
    - Alert generation integration with monitoring systems
    - Error context preservation across call stacks
    - Exception handling integration with prediction pipeline
    
  - **ML Operations Integration:**
    - Training lifecycle event logging integration
    - Model deployment workflow logging
    - Drift detection to retraining event correlation
    - Feature importance logging with model analysis

- Edge Cases:
  - **StructuredFormatter Edge Cases:**
    - Log records with None exception info
    - Empty extra fields dictionaries
    - Non-serializable objects in extra fields
    - Very large log messages (>1MB)
    - Unicode characters in log messages
    - Circular references in extra field objects
    
  - **Configuration Edge Cases:**
    - Missing config/logging.yaml file handling
    - Malformed YAML configuration files
    - Invalid handler configurations
    - Missing logs directory with permission issues
    - Configuration loading failures with partial setup
    - Multiple LoggerManager instances (testing singleton behavior)
    
  - **Performance Logger Edge Cases:**
    - Negative duration values
    - None values for optional parameters
    - Empty metrics dictionaries
    - Very large metric values (overflow scenarios)
    - Concurrent logging from multiple threads
    
  - **Error Tracking Edge Cases:**
    - None exception objects
    - Exceptions with non-string messages
    - Empty context dictionaries
    - Invalid severity levels
    - Recursive error tracking (errors within error handlers)
    - Memory pressure during error logging
    
  - **Context Manager Edge Cases:**
    - Exceptions raised before yield in log_operation
    - Exceptions raised after yield in log_operation
    - Nested context manager usage
    - Context manager cleanup during system shutdown
    - Timer overflow with very long operations
    - Concurrent context manager usage with same operation names

- Error Handling:
  - **Configuration Error Handling:**
    - YAML parsing errors with fallback to basic logging
    - File permission errors accessing config files
    - Invalid formatter class references in configuration
    - Handler initialization failures
    - Directory creation failures for log files
    
  - **Logging Error Handling:**
    - JSON serialization failures with fallback
    - Disk space exhaustion during file logging
    - Network failures for remote logging handlers
    - Log file rotation failures
    - Handler-level exceptions with graceful degradation
    
  - **Performance Monitoring Error Handling:**
    - Timer failures in context manager
    - Metric collection failures with system monitoring
    - Resource usage query failures
    - Performance data serialization errors
    
  - **Error Tracker Error Handling:**
    - Recursive error scenarios (ErrorTracker tracking its own errors)
    - Alert system failures during error notification
    - Context serialization failures
    - Error severity validation and fallback
    - Exception info extraction failures
    
  - **ML Operations Error Handling:**
    - Model metric serialization failures
    - Training event logging during system shutdown
    - Feature importance data validation errors
    - Component identification failures
    - Lifecycle event correlation errors
    
  - **Thread Safety Error Handling:**
    - Concurrent logger initialization
    - Race conditions in singleton creation
    - Thread-local storage corruption
    - Logging during interpreter shutdown
    - Resource contention under high logging volume

- Coverage Target: 85%+


### src/utils/health_monitor.py - Health Monitoring
**Classes Found:** HealthStatus(Enum), ComponentType(Enum), HealthThresholds, ComponentHealth, SystemHealth, HealthMonitor
**Methods Analyzed:** __init__, start_monitoring, stop_monitoring, _monitoring_loop, _run_health_checks, _run_single_health_check, _check_system_resources, _check_database_connection, _check_mqtt_broker, _check_api_endpoints, _check_memory_usage, _check_disk_space, _check_network_connectivity, _check_application_metrics, _calculate_system_health, _process_incidents, _trigger_incident_response, get_system_health, get_component_health, get_health_history, is_monitoring_active, get_monitoring_stats

**Required Tests:**
- Unit Tests:
  - **HealthStatus Enum:** Test all enum values (HEALTHY, WARNING, DEGRADED, CRITICAL, UNKNOWN)
  - **ComponentType Enum:** Test all component types (DATABASE, MQTT, API, SYSTEM, NETWORK, APPLICATION, EXTERNAL)
  - **HealthThresholds:** Test default threshold values and custom thresholds initialization
  - **ComponentHealth:** Test initialization, is_healthy(), needs_attention(), to_dict() serialization
  - **SystemHealth:** Test health_score() calculation with various component ratios, to_dict() serialization, alert penalty calculations
  - **HealthMonitor.__init__:** Test initialization with default and custom parameters, dependency injection setup
  - **register_health_check:** Test custom health check registration and validation
  - **add_incident_callback:** Test callback registration and storage
  - **start_monitoring:** Test monitoring activation, task creation, duplicate start prevention, metrics recording
  - **stop_monitoring:** Test monitoring deactivation, task cancellation, cleanup
  - **_run_health_checks:** Test parallel health check execution, result processing, exception handling
  - **_run_single_health_check:** Test individual check execution, timeout handling, consecutive failure tracking, metrics updates
  - **_check_system_resources:** Test CPU/memory/disk usage calculation, threshold evaluation, status determination
  - **_check_database_connection:** Test database connectivity, response time evaluation, health status mapping
  - **_check_mqtt_broker:** Test MQTT broker connection, authentication, timeout handling
  - **_check_api_endpoints:** Test HTTP endpoint health checks, response time aggregation, success rate calculation
  - **_check_memory_usage:** Test process memory monitoring, system memory percentage calculation
  - **_check_disk_space:** Test disk usage monitoring, threshold evaluation
  - **_check_network_connectivity:** Test external connectivity, DNS resolution, HTTP requests
  - **_check_application_metrics:** Test application performance metrics evaluation, issue detection
  - **_calculate_system_health:** Test overall health calculation, component status aggregation, performance scoring
  - **_process_incidents:** Test incident detection, alert threshold evaluation
  - **_trigger_incident_response:** Test alert creation, callback execution, incident logging
  - **get_system_health/get_component_health:** Test health status retrieval, filtering
  - **get_health_history:** Test historical data retrieval with time filtering
  - **is_monitoring_active/get_monitoring_stats:** Test status reporting, statistics calculation

- Integration Tests:
  - **Full Monitoring Cycle:** Test complete health monitoring cycle with real dependencies
  - **Database Integration:** Test real database health checks with TimescaleDB
  - **MQTT Integration:** Test real MQTT broker connectivity and authentication
  - **API Integration:** Test real HTTP endpoint monitoring with various response codes
  - **System Resource Integration:** Test real system metrics collection under various loads
  - **Alert Manager Integration:** Test alert triggering with real alert manager
  - **Metrics Collector Integration:** Test metrics recording with real metrics collector
  - **Multi-Component Health:** Test health monitoring with all component types active
  - **Performance Under Load:** Test monitoring performance with high-frequency checks
  - **Recovery Scenarios:** Test monitoring recovery after component failures

- Edge Cases:
  - **Threshold Boundary Testing:** Test health status changes at exact threshold values
  - **Zero Component Scenario:** Test system health calculation with no registered components
  - **All Critical Scenario:** Test system behavior when all components are critical
  - **Rapid Status Changes:** Test frequent component status transitions
  - **Long-Running Monitoring:** Test monitoring stability over extended periods
  - **Resource Exhaustion:** Test behavior under system resource constraints
  - **Concurrent Health Checks:** Test parallel execution of multiple health checks
  - **Health Check Timeout Edge Cases:** Test various timeout scenarios (just under/over limits)
  - **Memory Leak Prevention:** Test long-running monitoring for memory leaks
  - **Historical Data Overflow:** Test health history deque maxlen behavior

- Error Handling:
  - **Monitoring Loop Exceptions:** Test monitoring loop resilience to various exceptions
  - **Health Check Failures:** Test individual health check failure handling and recovery
  - **Timeout Handling:** Test asyncio timeout scenarios for all health checks
  - **Database Connection Errors:** Test various database connectivity failure modes
  - **MQTT Connection Errors:** Test MQTT broker connectivity failures, authentication errors
  - **HTTP Request Errors:** Test API endpoint failures, network timeouts, DNS resolution failures
  - **System Metrics Errors:** Test psutil exceptions, permission errors, unavailable metrics
  - **Alert Manager Failures:** Test alert triggering failures and fallback behavior
  - **Metrics Collection Errors:** Test metrics recording failures and error isolation
  - **Callback Execution Errors:** Test incident callback failures and error propagation
  - **Concurrent Access Errors:** Test thread safety and concurrent modification scenarios
  - **Resource Cleanup Errors:** Test cleanup failures during monitoring shutdown
  - **Configuration Errors:** Test invalid threshold values, missing dependencies
  - **Network Connectivity Errors:** Test various network failure scenarios
  - **Performance Degradation:** Test monitoring behavior under high system load

- Coverage Target: 85%+
### src/utils/monitoring_integration.py - Monitoring Integration
**Classes Found:** MonitoringIntegration
**Methods Analyzed:** __init__, _setup_integrations, start_monitoring, stop_monitoring, _handle_performance_alert, track_prediction_operation, track_training_operation, record_prediction_accuracy, record_concept_drift, record_feature_computation, record_database_operation, record_mqtt_publish, record_ha_api_request, update_connection_status, get_monitoring_status, get_monitoring_integration

**Required Tests:**
- Unit Tests:
  - **MonitoringIntegration.__init__():**
    - Verify all logger instances are properly initialized
    - Confirm all manager instances are created
    - Test _setup_integrations is called during initialization
    - Validate proper component integration setup
    
  - **MonitoringIntegration._setup_integrations():**
    - Test performance monitor alert callback registration
    - Verify _handle_performance_alert is properly bound
    - Test logging of successful initialization
    - Test with missing performance monitor (error handling)
    
  - **MonitoringIntegration.start_monitoring():**
    - Test successful startup sequence (metrics + monitoring)
    - Verify metrics_manager.start_background_collection() is called
    - Verify monitoring_manager.start_monitoring() is called
    - Test exception handling with proper alert triggering
    - Test logging of successful startup
    
  - **MonitoringIntegration.stop_monitoring():**
    - Test successful shutdown sequence
    - Verify monitoring_manager.stop_monitoring() is called
    - Verify metrics_manager.stop_background_collection() is called
    - Test exception handling during shutdown
    - Test logging of successful shutdown
    
  - **MonitoringIntegration._handle_performance_alert():**
    - Test alert conversion from monitoring format to alert manager format
    - Verify alert manager trigger_alert is called with correct parameters
    - Test alert with room_id attribute
    - Test alert without room_id attribute (None fallback)
    - Test exception handling during alert processing
    
  - **MonitoringIntegration.track_prediction_operation():**
    - Test successful prediction tracking context manager
    - Verify start logging with proper extra parameters
    - Verify duration calculation and performance logging
    - Verify metrics collection with success status
    - Test monitoring manager metric recording
    - Test completion logging with duration
    - Test exception handling with error metrics recording
    - Test alert manager error handling call
    - Test error logging with proper extra parameters and exc_info
    
  - **MonitoringIntegration.track_training_operation():**
    - Test successful training tracking context manager
    - Verify ML ops logger training start event
    - Verify duration calculation and metrics recording
    - Test monitoring manager training time recording
    - Verify ML ops logger completion event with metrics
    - Test exception handling with error alert and logging
    - Test ML ops logger error event recording
    
  - **MonitoringIntegration.record_prediction_accuracy():**
    - Test performance logger accuracy recording
    - Verify metrics collector prediction recording with accuracy
    - Test monitoring manager accuracy metric recording
    - Test with all required parameters
    
  - **MonitoringIntegration.record_concept_drift():**
    - Test ML ops logger drift detection logging
    - Verify metrics collector drift recording
    - Test alert triggering for high severity (>0.5)
    - Test no alert for low severity (<=0.5)
    - Test async task creation for alert triggering
    
  - **MonitoringIntegration.record_feature_computation():**
    - Test metrics collector feature computation recording
    - Verify monitoring manager performance metric recording
    - Test with various feature types
    
  - **MonitoringIntegration.record_database_operation():**
    - Test metrics collector database operation recording
    - Verify monitoring manager database performance recording
    - Test with success status
    - Test with error status
    - Test with different operation types and tables
    
  - **MonitoringIntegration.record_mqtt_publish():**
    - Test metrics collector MQTT publish recording
    - Test with success status
    - Test with error status
    - Test with different topic types and room IDs
    
  - **MonitoringIntegration.record_ha_api_request():**
    - Test metrics collector API request recording
    - Test with different endpoints, methods, and statuses
    
  - **MonitoringIntegration.update_connection_status():**
    - Test metrics collector connection status update
    - Test alert triggering when connection lost (connected=False)
    - Test no alert when connection established (connected=True)
    - Test async task creation for alert
    
  - **MonitoringIntegration.get_monitoring_status():**
    - Test successful status retrieval from all systems
    - Verify monitoring_manager status call
    - Verify alert_manager status call
    - Test metrics availability check
    - Test complete status dictionary structure
    - Test exception handling with error dictionary return
    - Test timestamp inclusion in both success and error cases
    
  - **get_monitoring_integration():**
    - Test singleton pattern implementation
    - Verify single instance creation
    - Test multiple calls return same instance
    - Test global variable management

- Integration Tests:
  - **Full Monitoring System Integration:**
    - Test complete start/stop cycle with real components
    - Verify integration with actual alert, metrics, and monitoring managers
    - Test cross-component communication and data flow
    
  - **Alert System Integration:**
    - Test performance alert handling end-to-end
    - Verify alert conversion and propagation
    - Test concept drift alert triggering
    - Test connection loss alert handling
    
  - **Prediction Workflow Integration:**
    - Test complete prediction tracking workflow
    - Verify integration with performance logging and metrics
    - Test error handling through complete alert pipeline
    
  - **Training Workflow Integration:**
    - Test complete training operation tracking
    - Verify ML ops logging integration
    - Test error handling and alert generation
    
  - **Metrics Collection Integration:**
    - Test all metric recording methods with actual collectors
    - Verify performance monitoring integration
    - Test database and MQTT metric recording
    
  - **Status Monitoring Integration:**
    - Test comprehensive status retrieval
    - Verify integration with all monitoring subsystems
    - Test status during various system states

- Edge Cases:
  - **Context Manager Edge Cases:**
    - Exception raised before yield in prediction tracking
    - Exception raised after yield in prediction tracking
    - Exception raised before yield in training tracking
    - Exception raised after yield in training tracking
    - Nested context manager usage
    - Context manager cleanup during system shutdown
    
  - **Timing Edge Cases:**
    - Very short duration operations (microseconds)
    - Very long duration operations (hours)
    - Concurrent prediction tracking operations
    - System clock changes during operation tracking
    
  - **Alert Threshold Edge Cases:**
    - Concept drift severity exactly at 0.5 threshold
    - Concept drift severity at boundary values (0.0, 1.0)
    - Performance alerts with missing attributes
    - Alert objects with None or empty additional_info
    
  - **Connection Status Edge Cases:**
    - Rapid connection state changes
    - Multiple connection types changing simultaneously
    - Connection status updates during system shutdown
    
  - **Metric Recording Edge Cases:**
    - Recording metrics with None values
    - Recording metrics with extremely large numbers
    - Recording metrics with special float values (inf, nan)
    - Concurrent metric recording operations
    
  - **Status Retrieval Edge Cases:**
    - Partial component failures during status check
    - Status retrieval during system startup/shutdown
    - Missing or uninitialized components
    - Circular dependency issues in status checking

- Error Handling:
  - **Initialization Error Handling:**
    - Failed logger initialization with fallback behavior
    - Missing manager dependencies during initialization
    - Integration setup failures with partial rollback
    
  - **Monitoring System Error Handling:**
    - Start monitoring failures with proper cleanup
    - Stop monitoring failures with graceful degradation
    - Component startup sequence failures
    - Resource exhaustion during monitoring startup
    
  - **Alert Processing Error Handling:**
    - Performance alert handling failures
    - Alert manager communication failures
    - Alert conversion failures with malformed alerts
    - Network failures during alert transmission
    
  - **Context Manager Error Handling:**
    - Exception propagation in prediction tracking
    - Exception propagation in training tracking
    - Cleanup failures in context managers
    - Resource leaks in error scenarios
    
  - **Metric Recording Error Handling:**
    - Metrics collector communication failures
    - Performance monitor recording failures
    - Invalid metric values handling
    - Metric serialization failures
    
  - **Async Operation Error Handling:**
    - Task creation failures for alerts
    - Concurrent operation conflicts
    - Event loop shutdown during async operations
    - Resource cleanup in async error scenarios
    
  - **Component Communication Error Handling:**
    - Manager interface failures
    - Service unavailability handling
    - Timeout handling for component operations
    - Partial service degradation scenarios
    
  - **Status Monitoring Error Handling:**
    - Component status retrieval failures
    - Incomplete status information handling
    - Status inconsistency detection
    - Error status propagation and aggregation

- Coverage Target: 85%+
### src/utils/monitoring.py - System Monitoring
**Classes Found:** PerformanceThreshold, HealthCheckResult, AlertEvent, PerformanceMonitor, SystemHealthMonitor, MonitoringManager
**Methods Analyzed:** PerformanceMonitor (17 methods), SystemHealthMonitor (9 methods), MonitoringManager (8 methods), plus dataclass constructors and global get_monitoring_manager()

**Required Tests:**
- Unit Tests: 
  - **PerformanceThreshold dataclass:**
    - Constructor with all required fields (name, warning_threshold, critical_threshold, unit, description)
    - Field validation and type checking
    - Default value behavior and immutability

  - **HealthCheckResult dataclass:**
    - Constructor with required fields (component, status, response_time, message)
    - Optional details field default behavior
    - Status validation ('healthy', 'warning', 'critical')

  - **AlertEvent dataclass:**
    - Constructor with all required fields (timestamp, alert_type, component, message, metric_name, current_value, threshold)
    - Optional additional_info field handling
    - Alert type validation ('warning', 'critical')

  - **PerformanceMonitor class:**
    - __init__() - Logger initialization, performance_history defaultdict with deque(maxlen=1000), alert_callbacks list, thresholds dictionary setup
    - add_alert_callback() - Callback function registration and list management
    - record_performance_metric() - Metric recording with timestamp, history storage, performance logging, threshold checking
    - _check_threshold() - Warning/critical threshold comparison, alert creation, alert triggering
    - _trigger_alert() - Logger warning, metrics recording, callback notification with error handling
    - get_performance_summary() - Time window filtering, statistics calculation (mean, median, std, min, max, p95, p99)
    - _percentile() - Percentile calculation algorithm, edge cases (empty list, single value)
    - get_trend_analysis() - Linear regression calculation, trend determination, data sufficiency checks

  - **SystemHealthMonitor class:**
    - __init__() - Logger setup, health_checks dict, last_health_check dict, default health check registration
    - register_health_check() - Custom health check function registration
    - _register_default_health_checks() - Default health check mapping setup
    - _check_system_resources() - psutil CPU/memory checks, status determination, response time measurement
    - _check_disk_space() - Disk usage calculation, threshold evaluation, status assignment
    - _check_memory_usage() - Process and system memory analysis, percentage calculation
    - _check_cpu_usage() - CPU percentage monitoring, load average handling
    - run_health_checks() - All health checks execution, error handling, metrics updates
    - get_overall_health_status() - System health assessment, outdated check detection

  - **MonitoringManager class:**
    - __init__() - Component initialization (performance_monitor, health_monitor, metrics_collector)
    - start_monitoring() - Monitoring task creation, state management
    - stop_monitoring() - Task cancellation, cleanup handling
    - _monitoring_loop() - Main monitoring cycle, interval management, error recovery
    - get_performance_monitor() - Performance monitor instance access
    - get_health_monitor() - Health monitor instance access
    - get_monitoring_status() - Comprehensive status collection and formatting

  - **Global functions:**
    - get_monitoring_manager() - Singleton pattern implementation, instance creation and retrieval

- Integration Tests:
  - **PerformanceMonitor Integration:**
    - Integration with logger, performance_logger, error_tracker, metrics_collector
    - End-to-end metric recording flow with threshold checking and alerting
    - Performance history management with real metrics data
    - Alert callback system with multiple registered callbacks
    - Trend analysis with various metric patterns over time

  - **SystemHealthMonitor Integration:**
    - Integration with psutil for real system resource monitoring
    - Health check registration and execution with actual system checks
    - Metrics collector updates during health check runs
    - Overall health status calculation with real system data

  - **MonitoringManager Integration:**
    - Complete monitoring system startup and shutdown
    - Health check and performance summary scheduling
    - Integration between all monitoring components
    - Real-time monitoring loop with actual system metrics

  - **Cross-Component Integration:**
    - PerformanceMonitor alerts triggering SystemHealthMonitor checks
    - MonitoringManager coordinating both performance and health monitoring
    - Shared metrics collector across all monitoring components

- Edge Cases:
  - **PerformanceMonitor Edge Cases:**
    - Empty performance history for summary and trend analysis
    - Performance history at maxlen limit (1000 entries) with wraparound
    - Threshold checking with exactly matching warning/critical values
    - Alert callback exceptions during notification
    - Percentile calculation with single value and empty lists
    - Linear regression with insufficient data points (< 2)
    - Concurrent metric recording with same metric_name and room_id

  - **SystemHealthMonitor Edge Cases:**
    - psutil exceptions during resource monitoring
    - Health check function exceptions during execution
    - Outdated health checks beyond 5-minute threshold
    - Zero CPU usage intervals and measurement timing
    - Memory usage calculations with process termination
    - Load average unavailability on some systems (hasattr check)
    - Health check response time measurement during exceptions

  - **MonitoringManager Edge Cases:**
    - Start monitoring when already running
    - Stop monitoring when not running
    - Monitoring loop exceptions and recovery
    - Task cancellation during health check execution
    - Concurrent start/stop monitoring calls
    - System resource exhaustion during monitoring
    - Long-running health checks affecting monitoring intervals

- Error Handling:
  - **PerformanceMonitor Error Handling:**
    - Logger initialization failures with fallback behavior
    - Performance history storage failures and memory constraints
    - Alert callback execution exceptions with error tracking
    - Threshold configuration errors and validation
    - Statistics calculation errors with insufficient data
    - Performance summary generation with corrupted history data
    - Trend analysis mathematical errors (division by zero, invalid slopes)

  - **SystemHealthMonitor Error Handling:**
    - psutil import/availability issues
    - System resource query failures (CPU, memory, disk)
    - Health check function exceptions and error capture
    - Process monitoring failures with PID changes
    - Disk space check failures on different filesystems
    - CPU usage measurement timeout and interval issues
    - Overall health status assessment with missing data

  - **MonitoringManager Error Handling:**
    - Component initialization failures (performance_monitor, health_monitor)
    - Monitoring task creation and scheduling errors
    - AsyncIO task cancellation and cleanup issues
    - Health check interval management with system clock changes
    - Performance summary generation failures
    - Monitoring loop resilience to component failures
    - Resource cleanup during shutdown with pending operations

  - **Integration Error Handling:**
    - Dependencies on external utilities (logger, metrics_collector)
    - Cross-component communication failures
    - Singleton pattern thread safety issues
    - System resource monitoring under high load
    - Alert notification system failures
    - Performance data persistence and recovery

- Coverage Target: 85%+

### src/utils/alerts.py - Alert System
**Classes Found:** AlertSeverity, AlertChannel, AlertRule, AlertEvent, NotificationConfig, AlertThrottler, EmailNotifier, WebhookNotifier, MQTTNotifier, ErrorRecoveryManager, AlertManager
**Methods Analyzed:** 47 methods across 11 classes including throttling, notification handling, recovery management, and alert lifecycle

**Required Tests:**
- Unit Tests: 
  - **AlertThrottler Testing:**
    - should_send_alert() with first alert (should return True)
    - should_send_alert() within throttle window (should return False)  
    - should_send_alert() after throttle expires (should return True)
    - reset_throttle() clearing alert history
    - Multiple alert IDs with different throttle periods
    - Alert count tracking and incrementation
    
  - **EmailNotifier Testing:**
    - send_alert() with valid configuration and successful email send
    - send_alert() with disabled email configuration (should return False)
    - send_alert() with missing recipients (should return False)
    - _send_email() SMTP connection and authentication
    - _format_email_body() HTML generation for all severity levels
    - _format_context_html() with various context data types
    - Email sending failures and error handling
    - Async execution with run_in_executor
    
  - **WebhookNotifier Testing:**
    - send_alert() with successful webhook POST request
    - send_alert() with disabled webhook configuration
    - send_alert() with HTTP error responses (4xx, 5xx)
    - send_alert() with network timeout scenarios
    - Webhook payload structure validation
    - aiohttp session management and cleanup
    - Connection timeout handling
    
  - **MQTTNotifier Testing:**
    - send_alert() with enabled MQTT configuration
    - send_alert() with disabled MQTT configuration
    - _initialize_mqtt_client() setup and connection
    - MQTT topic construction with severity levels
    - MQTT client initialization failures
    - Alert payload formatting for MQTT
    
  - **ErrorRecoveryManager Testing:**
    - register_recovery_strategy() with various error patterns
    - attempt_recovery() with matching error patterns
    - attempt_recovery() with no matching patterns
    - Recovery strategy execution success/failure scenarios
    - Recovery history tracking and deque maxlen behavior
    - Async recovery execution with run_in_executor
    - Multiple recovery strategies for same error type
    
  - **AlertManager Testing:**
    - __init__() with default and custom NotificationConfig
    - _setup_default_alert_rules() creating 8 default rules
    - add_alert_rule() and remove_alert_rule() operations
    - trigger_alert() with valid rule names
    - trigger_alert() with invalid rule names (should return None)
    - trigger_alert() with throttling active (should be throttled)
    - resolve_alert() for active alerts
    - resolve_alert() for non-existent alerts
    - _generate_alert_id() uniqueness and determinism
    - _send_notifications() across multiple channels
    - handle_prediction_error() with recovery attempts
    - handle_model_training_error() alert creation
    - get_alert_status() metrics and counts
    
  - **Dataclass Validation Testing:**
    - AlertRule with all required and optional fields
    - AlertEvent with proper field types and defaults
    - NotificationConfig with all email/webhook/MQTT settings
    - Enum value validation for AlertSeverity and AlertChannel

- Integration Tests:
  - **Email Integration:**
    - Full email workflow from alert trigger to SMTP delivery
    - Email HTML formatting with various alert severities
    - SMTP server authentication and TLS handling
    - Email delivery failure scenarios and retries
    
  - **Webhook Integration:**
    - End-to-end webhook delivery with real HTTP endpoints
    - Webhook authentication and header handling
    - Webhook retry logic for failed deliveries
    - JSON payload validation at receiving endpoint
    
  - **MQTT Integration:**
    - MQTT broker connection and topic publishing
    - Alert message delivery to Home Assistant MQTT
    - MQTT client lifecycle management
    - Topic hierarchy and message retention
    
  - **Alert Workflow Integration:**
    - Full alert lifecycle: trigger → notify → resolve
    - Multi-channel notification delivery coordination
    - Alert throttling across different notification channels
    - Error recovery integration with alert notifications
    
  - **Cross-Component Integration:**
    - AlertManager with ErrorTracker from logger module
    - AlertManager with MetricsCollector integration
    - Alert rule evaluation with real system metrics
    - Recovery manager integration with actual error scenarios

- Edge Cases:
  - **Throttling Edge Cases:**
    - Alert triggered exactly at throttle expiry boundary
    - System clock changes affecting throttle timing
    - Very high frequency alerts with microsecond precision
    - Throttle reset during active throttling period
    
  - **Notification Edge Cases:**
    - Network partitions during webhook/email delivery
    - SMTP server downtime during email sending
    - Very large alert messages exceeding size limits
    - Unicode and special characters in alert content
    - Malformed email addresses in recipient lists
    
  - **Recovery Edge Cases:**
    - Recovery strategies that themselves throw exceptions
    - Circular recovery attempts for related errors
    - Recovery context with non-serializable objects
    - Multiple simultaneous recovery attempts for same error
    
  - **Alert Management Edge Cases:**
    - Alert resolution of already resolved alerts
    - Massive alert volumes causing memory pressure
    - Alert history deque overflow with 10000+ alerts
    - Concurrent alert triggering with same ID generation
    - Alert rules with invalid condition expressions
    
  - **Configuration Edge Cases:**
    - NotificationConfig with partial SMTP settings
    - Webhook URLs with invalid formats or protocols
    - MQTT configuration with missing broker details
    - Alert rules with empty or None channel lists

- Error Handling:
  - **Email Error Handling:**
    - SMTP authentication failures with invalid credentials
    - Network connectivity issues during email sending
    - Email size limits exceeded by large alert messages
    - Invalid MIME type handling in email formatting
    - Recipient email address validation failures
    
  - **Webhook Error Handling:**
    - HTTP connection timeouts and retries
    - Invalid webhook URL formats and DNS resolution
    - JSON serialization failures for complex context data
    - Webhook endpoint authentication failures
    - SSL/TLS certificate validation errors
    
  - **MQTT Error Handling:**
    - MQTT broker connection failures and reconnection
    - Topic publishing failures with QoS handling
    - MQTT client initialization without proper configuration
    - Message payload size limits and truncation
    - MQTT authentication and authorization failures
    
  - **Alert System Error Handling:**
    - Alert ID generation with hash collision scenarios  
    - Alert rule condition parsing and validation errors
    - Memory exhaustion with large alert history storage
    - Concurrent access to shared alert state
    - Alert manager initialization failures with invalid config
    
  - **Recovery System Error Handling:**
    - Recovery strategy registration with duplicate patterns
    - Recovery function execution timeouts and cancellation
    - Recovery history serialization failures
    - Error pattern matching with regex compilation errors
    - Recovery strategy removal during active recovery attempts
    
  - **Severity and Channel Error Handling:**
    - Invalid AlertSeverity enum values in alert creation
    - Unsupported AlertChannel types in notification sending
    - Channel-specific configuration validation failures
    - Alert rule channel list validation and sanitization
    - Notification handler initialization with missing dependencies

- Coverage Target: 85%+


### src/utils/time_utils.py - Time Utility Functions
**Classes Found:** TimeFrame (Enum), TimeRange, TimeUtils, AsyncTimeUtils, TimeProfiler
**Methods Analyzed:** TimeRange.__init__, _ensure_timezone, duration, contains, overlaps, intersection; TimeUtils.setup_local_timezone, now, utc_now, to_utc, to_timezone, parse_datetime, format_duration, time_until, time_since, round_to_interval, get_time_buckets, is_business_hours, get_cyclical_time_features, validate_timezone; AsyncTimeUtils.wait_until, periodic_task; TimeProfiler.__init__, __enter__, __exit__, duration, duration_seconds, __call__; Convenience functions: format_duration, time_until, time_since, cyclical_time_features

**Required Tests:**
- Unit Tests:
  - **TimeFrame Enum Testing:**
    - All enum values accessible and correct (MINUTE, HOUR, DAY, WEEK, MONTH)
    - String representation and comparisons
  
  - **TimeRange Class Testing:**
    - Initialization with timezone-aware and naive datetimes
    - Initialization with custom timezone strings
    - Start time before end time validation (ValueError testing)
    - _ensure_timezone method with various timezone scenarios
    - Duration property calculation for different time spans
    - Contains method with boundary conditions and timezone handling
    - Overlaps method with adjacent, overlapping, and non-overlapping ranges
    - Intersection method returning correct overlap or None
    - Edge cases: same start/end times, microsecond differences
  
  - **TimeUtils Static Methods Testing:**
    - setup_local_timezone with valid/invalid timezone strings and auto-detection
    - now() with different timezone parameters
    - utc_now() consistency and timezone validation
    - to_utc() with naive and timezone-aware inputs
    - to_timezone() with string and timezone object parameters
    - parse_datetime() with all supported formats and custom formats
    - parse_datetime() failure scenarios with invalid strings
    - format_duration() with various precisions and negative durations
    - time_until() and time_since() with timezone handling
    - round_to_interval() with up/down/nearest directions and different intervals
    - get_time_buckets() with various intervals and edge cases
    - is_business_hours() with different hours and weekend scenarios
    - get_cyclical_time_features() mathematical correctness for all components
    - validate_timezone() with valid and invalid timezone strings
  
  - **AsyncTimeUtils Testing:**
    - wait_until() behavior with past and future times
    - periodic_task() generator with different intervals and alignment
    - periodic_task() with max_iterations parameter
    - Async timeout scenarios and cancellation handling
  
  - **TimeProfiler Context Manager Testing:**
    - Context manager entry and exit timing accuracy
    - Duration property calculation before and after completion
    - Duration_seconds property with various operation lengths
    - Decorator functionality with different function return types
    - Exception handling during profiled operations
    - Nested profiler usage scenarios

- Integration Tests:
  - **Timezone Integration Testing:**
    - Cross-timezone calculations between TimeRange and TimeUtils
    - DST transition handling in time calculations
    - Real timezone data validation with pytz integration
    - System timezone detection and fallback behavior
  
  - **ML Feature Integration Testing:**
    - Cyclical features integration with actual datetime sequences
    - Business hours calculation across different timezones
    - Time bucket generation for real data analysis scenarios
    - Duration formatting in user-facing applications
  
  - **Async Integration Testing:**
    - AsyncTimeUtils integration with asyncio event loops
    - Periodic task execution timing accuracy over extended periods
    - wait_until integration with real-time event processing
    - Concurrent async time operations

- Edge Cases:
  - **Timezone Edge Cases:**
    - Invalid timezone strings and error handling
    - Timezone-naive datetime handling in all methods
    - DST transition boundaries and ambiguous times
    - UTC offset changes and historical timezone data
    - Leap year and leap second handling
  
  - **Boundary Value Testing:**
    - Minimum and maximum datetime values
    - Zero-duration time ranges and calculations
    - Extremely large time intervals (years, decades)
    - Microsecond precision in time calculations
    - Negative time deltas and duration calculations
  
  - **Cyclical Feature Edge Cases:**
    - End-of-month date handling (28, 29, 30, 31 days)
    - Year boundary calculations (December to January)
    - Leap year February 29th handling
    - Hour boundary calculations with DST
  
  - **Async Edge Cases:**
    - wait_until with target time in the past
    - periodic_task cancellation and cleanup
    - Very short intervals (milliseconds) in periodic tasks
    - Long-running periodic tasks memory usage
    - System clock changes during async operations
  
  - **Profiler Edge Cases:**
    - TimeProfiler with zero-duration operations
    - Profiler state when exceptions occur in context
    - Decorator usage with async functions
    - Nested profiler scenarios and timing accuracy
    - Context manager reuse and state reset

- Error Handling:
  - **TimeRange Error Handling:**
    - ValueError when start >= end time
    - Invalid timezone string handling in initialization
    - None datetime inputs and validation
    - Timezone conversion failures during operations
  
  - **TimeUtils Parsing Error Handling:**
    - parse_datetime with completely invalid strings
    - Timezone validation with malformed timezone data
    - Format list exhaustion in datetime parsing
    - pytz.UnknownTimeZoneError handling in validate_timezone
  
  - **Datetime Conversion Error Handling:**
    - to_timezone with invalid timezone parameters
    - to_utc with corrupted datetime objects
    - round_to_interval with zero or negative intervals
    - get_time_buckets with invalid start/end relationships
  
  - **AsyncTimeUtils Error Handling:**
    - wait_until with None target_time
    - periodic_task with negative intervals
    - Asyncio cancellation during time operations
    - System shutdown during async time waiting
  
  - **TimeProfiler Error Handling:**
    - Context manager usage without proper entry/exit
    - Exception propagation through profiled operations
    - Decorator usage with non-callable objects
    - Duration calculation with invalid start/end times
    - Nested context manager exception scenarios

- Coverage Target: 85%+
### src/utils/incident_response.py - Incident Response
**Classes Found:** IncidentSeverity, IncidentStatus, RecoveryActionType, RecoveryAction, Incident, IncidentResponseManager
**Methods Analyzed:** RecoveryAction.__init__(), RecoveryAction.can_attempt(), RecoveryAction.record_attempt(), RecoveryAction.reset_attempts(), Incident.__init__(), Incident.add_timeline_entry(), Incident.acknowledge(), Incident.resolve(), Incident.escalate(), Incident.needs_escalation(), Incident.to_dict(), IncidentResponseManager.__init__(), IncidentResponseManager._register_default_recovery_actions(), IncidentResponseManager.register_recovery_action(), IncidentResponseManager.start_incident_response(), IncidentResponseManager.stop_incident_response(), IncidentResponseManager._response_loop(), IncidentResponseManager._handle_health_incident(), IncidentResponseManager._create_incident_from_health(), IncidentResponseManager._attempt_automated_recovery(), IncidentResponseManager._check_recovery_conditions(), IncidentResponseManager._should_auto_resolve_incident(), IncidentResponseManager._auto_resolve_incident(), IncidentResponseManager._check_escalations(), IncidentResponseManager._cleanup_incidents(), IncidentResponseManager._update_incident_metrics(), IncidentResponseManager._recover_database_connection(), IncidentResponseManager._recover_mqtt_connection(), IncidentResponseManager._recover_memory_usage(), IncidentResponseManager._recover_api_endpoints(), IncidentResponseManager.get_active_incidents(), IncidentResponseManager.get_incident(), IncidentResponseManager.get_incident_history(), IncidentResponseManager.get_incident_statistics(), IncidentResponseManager.acknowledge_incident(), IncidentResponseManager.resolve_incident(), get_incident_response_manager()

**Required Tests:**
- Unit Tests:
  - **Enum Classes Tests:**
    - Test IncidentSeverity enum values (INFO, MINOR, MAJOR, CRITICAL, EMERGENCY) with string mapping
    - Test IncidentStatus enum values (NEW, ACKNOWLEDGED, INVESTIGATING, IN_PROGRESS, RESOLVED, CLOSED)
    - Test RecoveryActionType enum values (RESTART_SERVICE, CLEAR_CACHE, RESTART_COMPONENT, SCALE_RESOURCES, FAILOVER, NOTIFICATION, CUSTOM)
    - Test enum value access and validation
    
  - **RecoveryAction Dataclass Tests:**
    - Test __init__() with all required fields (action_type, component, description, function, conditions, max_attempts, cooldown_minutes)
    - Test default values for optional fields (conditions empty dict, max_attempts=3, cooldown_minutes=15, attempt_count=0, success_count=0)
    - Test can_attempt() method with max_attempts limit (returns False when attempts >= max_attempts)
    - Test can_attempt() cooldown logic with last_attempted timestamp (returns False during cooldown period)
    - Test can_attempt() returns True for first attempt (last_attempted=None)
    - Test record_attempt() with success=True (increments both attempt_count and success_count, sets last_attempted)
    - Test record_attempt() with success=False (increments attempt_count only, sets last_attempted)
    - Test reset_attempts() method (sets attempt_count=0, last_attempted=None)
    - Test datetime.now() usage in record_attempt method
    - Test timedelta cooldown calculation in can_attempt method
    
  - **Incident Dataclass Tests:**
    - Test __init__() with all required fields (incident_id, title, description, severity, status, component, component_type, created_at, updated_at, source_health)
    - Test default values for optional tracking fields (acknowledged_by=None, recovery_actions_attempted empty list, escalation_level=0)
    - Test add_timeline_entry() method with event string and optional details dict
    - Test timeline entry structure (timestamp ISO format, event, details)
    - Test updated_at timestamp updates in add_timeline_entry()
    - Test acknowledge() method status change to ACKNOWLEDGED
    - Test acknowledge() setting acknowledged_by and acknowledged_at timestamps
    - Test acknowledge() timeline entry creation with acknowledgment details
    - Test resolve() method status change to RESOLVED with resolution_notes
    - Test resolve() setting resolved_at timestamp and recovery_success flag
    - Test resolve() timeline entry with resolution details and action count
    - Test escalate() method increment of escalation_level and escalated_at timestamp
    - Test escalate() severity progression (MINOR->MAJOR->CRITICAL->EMERGENCY)
    - Test escalate() timeline entry with escalation details
    - Test needs_escalation() returns False for resolved/closed incidents
    - Test needs_escalation() time-based escalation logic (created_at + escalation_threshold_minutes)
    - Test to_dict() method comprehensive dictionary conversion with all fields
    - Test to_dict() ISO timestamp formatting for datetime fields
    - Test to_dict() source_health.to_dict() integration
    
  - **IncidentResponseManager Initialization Tests:**
    - Test __init__() with default check_interval=60
    - Test __init__() with custom check_interval parameter
    - Test dependency injection (get_logger, get_health_monitor, get_alert_manager, get_metrics_collector)
    - Test configuration defaults (auto_recovery_enabled=True, escalation_enabled=True)
    - Test data structure initialization (active_incidents dict, incident_history list, recovery_actions dict)
    - Test statistics initialization with all counter fields (incidents_created, incidents_resolved, etc.)
    - Test _register_default_recovery_actions() method call during initialization
    - Test response state initialization (_response_active=False, _response_task=None)

- Integration Tests:
  - **Health Monitor Integration:**
    - Test complete health incident workflow from ComponentHealth to Incident creation
    - Test health monitor callback registration and incident trigger
    - Test incident updates from health status changes
    - Test component health lookup for auto-resolution validation
    
  - **Alert Manager Integration:**
    - Test incident alert creation with proper severity mapping
    - Test escalation alert generation with escalation context
    - Test alert manager exception handling during incident creation
    - Test alert context population with incident metadata
    
  - **Metrics Collector Integration:**
    - Test incident metrics recording throughout lifecycle
    - Test startup/shutdown event metrics
    - Test recovery attempt statistics tracking
    - Test gauge metrics for active incident counts by status/severity

- Edge Cases:
  - **Recovery Action Edge Cases:**
    - Test recovery actions with zero max_attempts (should never execute)
    - Test recovery actions with negative cooldown periods
    - Test recovery action execution during system shutdown
    - Test recovery function exceptions during action execution
    - Test recovery condition evaluation with missing health metrics
    - Test concurrent recovery action attempts on same component
    
  - **Incident Management Edge Cases:**
    - Test incident creation with extremely long component names/descriptions
    - Test incident escalation beyond EMERGENCY severity
    - Test incident timeline with thousands of entries
    - Test incident serialization (to_dict) with complex nested data
    - Test incident acknowledgment/resolution of already processed incidents
    - Test rapid incident creation and resolution cycles

- Error Handling:
  - **Recovery Action Error Handling:**
    - Test recovery function exceptions with proper logging and attempt recording
    - Test recovery condition evaluation exceptions with graceful fallback
    - Test recovery action registration with invalid components or functions
    - Test cooldown period calculation with invalid timestamps
    - Test attempt tracking with counter overflow scenarios
    
  - **Incident Management Error Handling:**
    - Test incident creation with invalid ComponentHealth data
    - Test incident_id generation with timestamp formatting failures
    - Test timeline entry creation with non-serializable details
    - Test incident serialization failures during to_dict conversion
    - Test severity/status enum validation during incident operations

- Coverage Target: 85%+

### src/data/validation/schema_validator.py - Schema Validation
**Classes Found:** SchemaDefinition, SchemaValidationContext, JSONSchemaValidator, DatabaseSchemaValidator, APISchemaValidator
**Methods Analyzed:** SchemaDefinition.__init__(), SchemaValidationContext.__init__(), JSONSchemaValidator.__init__(), JSONSchemaValidator._initialize_format_checkers(), JSONSchemaValidator._load_standard_schemas(), JSONSchemaValidator._validate_sensor_id_format(), JSONSchemaValidator._validate_room_id_format(), JSONSchemaValidator._validate_entity_id_format(), JSONSchemaValidator._validate_iso_datetime_format(), JSONSchemaValidator._validate_sensor_state_format(), JSONSchemaValidator._validate_sensor_type_format(), JSONSchemaValidator._validate_coordinate_format(), JSONSchemaValidator._validate_duration_format(), JSONSchemaValidator.validate_json_schema(), JSONSchemaValidator._get_validation_suggestion(), JSONSchemaValidator.register_schema(), JSONSchemaValidator.get_schema_info(), DatabaseSchemaValidator.__init__(), DatabaseSchemaValidator.validate_database_schema(), DatabaseSchemaValidator._validate_table_columns(), DatabaseSchemaValidator._validate_sensor_events_columns(), DatabaseSchemaValidator._validate_room_states_columns(), DatabaseSchemaValidator._validate_predictions_columns(), DatabaseSchemaValidator._validate_indexes(), DatabaseSchemaValidator._validate_timescaledb_features(), APISchemaValidator.__init__(), APISchemaValidator.validate_api_request(), APISchemaValidator._validate_headers(), APISchemaValidator._validate_json_content(), APISchemaValidator._validate_form_content(), APISchemaValidator._validate_multipart_content(), APISchemaValidator._validate_query_params()

**Required Tests:**
- Unit Tests:
  - **SchemaDefinition Dataclass Tests:**
    - Test __init__() with all required fields (schema_id, name, version, schema)
    - Test default values for optional fields (description="", tags=[], created_at=datetime.now(), validators=[])
    - Test dataclass field validation and type checking
    - Test datetime.now() default factory behavior
    - Test mutable default factory for lists (tags, validators)
    
  - **SchemaValidationContext Dataclass Tests:**
    - Test __init__() with default UUID generation for validation_id
    - Test default values (strict_mode=True, allow_additional_properties=False)
    - Test mutable defaults for custom_formats and validation_metadata dicts
    - Test UUID validation for generated validation_id values
    - Test context modification and immutability checks
    
  - **JSONSchemaValidator Initialization Tests:**
    - Test __init__() creates empty schemas and custom_validators dicts
    - Test _initialize_format_checkers() creates all 8 format checkers
    - Test _load_standard_schemas() creates sensor_event, room_config, api_request schemas
    - Test format checker registration and availability
    - Test JSONSCHEMA_AVAILABLE flag handling with mock imports
    - Test fallback Draft7Validator class when jsonschema not available
    
  - **Format Validation Tests:**
    - Test _validate_sensor_id_format() with valid Home Assistant entity IDs (binary_sensor.motion, sensor.temperature)
    - Test _validate_sensor_id_format() with invalid formats (missing domain, special chars, empty string)
    - Test _validate_room_id_format() with valid room IDs (living_room, kitchen-1, bedroom_2)
    - Test _validate_room_id_format() with invalid formats (special chars, too long, empty string)
    - Test _validate_entity_id_format() with domain.object_id patterns
    - Test _validate_iso_datetime_format() with ISO 8601 formats and Z timezone
    - Test _validate_sensor_state_format() with SensorState enum values
    - Test _validate_sensor_type_format() with SensorType enum values
    - Test _validate_coordinate_format() with valid coordinates (-180 to 180) and invalid ranges
    - Test _validate_duration_format() with ISO 8601 duration patterns (P1Y2M3DT4H5M6S)
    - Test format validation with non-string inputs and None values
    
  - **Schema Validation Tests:**
    - Test validate_json_schema() with valid sensor_event data
    - Test validate_json_schema() with missing required fields returns ValidationResult with errors
    - Test validate_json_schema() with invalid schema_id returns schema not found error
    - Test validation context strict_mode behavior with additional properties
    - Test custom format validation integration in validation process
    - Test validation error severity mapping (CRITICAL for required, HIGH for type, MEDIUM for format)
    - Test _get_validation_suggestion() method for all validator types
    - Test confidence score calculation based on error and warning counts
    - Test custom validator execution and error handling
    - Test ValidationResult structure with validation_id from context
    
  - **Schema Management Tests:**
    - Test register_schema() adds SchemaDefinition to schemas dict
    - Test register_schema() logs schema registration
    - Test get_schema_info() returns complete schema metadata dict
    - Test get_schema_info() returns None for non-existent schema_id
    - Test schema_info JSON serialization with datetime isoformat
    
  - **DatabaseSchemaValidator Initialization Tests:**
    - Test __init__() with AsyncSession parameter and expected table/column definitions
    - Test expected_tables set contains all 5 required tables
    - Test expected_columns dict structure for sensor_events, room_states, predictions
    - Test column set validation for each table type
    
  - **Database Schema Validation Tests:**
    - Test validate_database_schema() with complete database structure
    - Test missing table detection and CRITICAL error severity
    - Test unexpected table warnings (but not errors)
    - Test column validation for each expected table
    - Test TimescaleDB extension and hypertable validation
    - Test database inspection error handling with connection issues
    - Test confidence score calculation for database validation
    
  - **Database Column Validation Tests:**
    - Test _validate_sensor_events_columns() timestamp and JSON column type validation
    - Test _validate_room_states_columns() confidence column numeric type validation
    - Test _validate_predictions_columns() predicted_time timestamp validation
    - Test column type string matching logic (lowercase, substring matching)
    - Test column validation with missing columns (should not crash)
    
  - **Database Index Validation Tests:**
    - Test _validate_indexes() checks for critical performance indexes
    - Test expected index mapping for sensor_events, room_states, predictions
    - Test missing index detection with MEDIUM severity warnings
    - Test index validation exception handling for non-existent tables
    
  - **TimescaleDB Validation Tests:**
    - Test _validate_timescaledb_features() extension availability check
    - Test hypertable validation for sensor_events table
    - Test TimescaleDB catalog query execution and result parsing
    - Test PostgreSQL extension query error handling
    
  - **APISchemaValidator Initialization Tests:**
    - Test __init__() creates JSONSchemaValidator instance
    - Test content_type_validators mapping for all 3 supported types
    - Test validator function mapping and availability
    
  - **API Request Validation Tests:**
    - Test validate_api_request() with valid HTTP methods and paths
    - Test HTTP method validation with invalid methods returns HIGH severity errors
    - Test path format validation requiring leading slash
    - Test header validation including security header requirements
    - Test content type validation and body format checking
    - Test query parameter format and length validation
    - Test confidence score calculation for API validation
    
  - **API Content Validation Tests:**
    - Test _validate_json_content() with valid JSON objects and arrays
    - Test _validate_json_content() with invalid JSON syntax returns HIGH severity errors
    - Test _validate_form_content() with dict and string body types
    - Test _validate_multipart_content() with dict and bytes validation
    - Test content validation type checking and error messages
    
  - **API Header and Query Validation Tests:**
    - Test _validate_headers() security header requirement (Authorization, X-API-Key)
    - Test _validate_headers() header format validation (string names/values)
    - Test _validate_headers() header length limits (8192 char limit)
    - Test _validate_query_params() parameter name format validation
    - Test _validate_query_params() parameter value length limits (2048 chars)

- Integration Tests:
  - **JSON Schema Integration:**
    - Test complete sensor event validation workflow from raw data to ValidationResult
    - Test schema loading and format checker integration
    - Test custom validator registration and execution
    - Test validation context modification effects on validation behavior
    - Test error aggregation from multiple validation stages
    
  - **Database Schema Integration:**
    - Test complete database validation against real PostgreSQL/TimescaleDB instance
    - Test schema validation with actual database inspector metadata
    - Test TimescaleDB hypertable detection with real database queries
    - Test index existence validation against actual database structure
    - Test database connection error scenarios and graceful degradation
    
  - **API Request Integration:**
    - Test complete API request validation with real HTTP request data
    - Test content type header to body validation mapping
    - Test security header validation with actual authentication tokens
    - Test multi-part form data validation with complex payloads

- Edge Cases:
  - **Format Validation Edge Cases:**
    - Test format validators with None, empty string, and non-string inputs
    - Test boundary values for coordinate validation (-180, 0, 180, 181)
    - Test complex ISO datetime formats with microseconds and various timezones
    - Test malformed duration strings with partial ISO 8601 patterns
    - Test sensor ID validation with edge cases (single char domains, numeric object_ids)
    - Test entity ID validation with maximum length entity names
    
  - **Schema Validation Edge Cases:**
    - Test deeply nested JSON objects with multiple validation errors
    - Test schema validation with circular references in data
    - Test validation with extremely large JSON payloads (memory limits)
    - Test schema with conflicting validation rules (required + additional properties false)
    - Test custom validator execution with exceptions and timeouts
    - Test confidence score edge cases (0.0 minimum, multiple error types)
    
  - **Database Schema Edge Cases:**
    - Test database validation with partial table existence
    - Test column type validation with complex PostgreSQL types
    - Test TimescaleDB validation on non-TimescaleDB PostgreSQL instances
    - Test database schema validation with read-only database connections
    - Test very large table structures with hundreds of columns
    
  - **API Validation Edge Cases:**
    - Test API request validation with binary body content
    - Test header validation with international characters and encoding issues
    - Test query parameter validation with URL-encoded special characters
    - Test request validation with missing Content-Type headers
    - Test API validation with non-standard HTTP methods

- Error Handling:
  - **JSON Schema Error Handling:**
    - Test jsonschema import failures and fallback behavior
    - Test schema validation with corrupted schema definitions
    - Test format checker exceptions during validation process
    - Test custom validator registration with invalid callable objects
    - Test validation context with invalid UUID generation
    - Test schema loading with malformed JSON schema syntax
    
  - **Database Schema Error Handling:**
    - Test database connection failures during schema validation
    - Test SQLAlchemy inspector exceptions with invalid database metadata
    - Test TimescaleDB catalog query failures with permission issues
    - Test column type inspection with unsupported database engines
    - Test index validation with corrupted database metadata
    
  - **API Schema Error Handling:**
    - Test content validation with unsupported Content-Type headers
    - Test JSON parsing errors with malformed request bodies
    - Test header validation with non-UTF-8 encoded header values
    - Test query parameter validation with malformed URL encoding
    - Test request validation with missing required components

- Coverage Target: 85%+

### src/main_system.py - Main System Orchestration
**Classes Found:** OccupancyPredictionSystem
**Methods Analyzed:** __init__, initialize, run, shutdown, run_occupancy_prediction_system

**Required Tests:**
- Unit Tests: 
  - Test OccupancyPredictionSystem.__init__() - verify proper attribute initialization (config, tracking_manager, database_manager, mqtt_manager, running flag)
  - Test initialize() success path - verify database_manager setup, MQTT manager initialization, tracking manager initialization with API config
  - Test initialize() failure scenarios - database connection failure, MQTT initialization failure, tracking manager failure
  - Test initialize() API server status logging - test all three scenarios (enabled+running, enabled+not running, disabled)
  - Test run() with uninitialized system - verify it calls initialize() first
  - Test run() main loop - verify system stays running and sleeps properly
  - Test run() KeyboardInterrupt handling - verify graceful shutdown on SIGINT
  - Test run() exception handling - verify proper error logging and shutdown
  - Test shutdown() component cleanup - verify tracking_manager.stop_tracking(), mqtt_manager.cleanup() calls
  - Test shutdown() partial initialization - verify graceful handling when some components are None
  - Test run_occupancy_prediction_system() - verify it creates system instance and calls run()

- Integration Tests:
  - Test complete system startup sequence - database → MQTT → tracking manager → API server
  - Test system integration with real config loading and component initialization
  - Test automatic API server startup through tracking manager integration
  - Test system shutdown sequence with all components properly stopped
  - Test system restart scenario - verify clean shutdown and re-initialization
  - Test concurrent system operations during main loop execution
  - Test system behavior with missing or invalid configuration

- Edge Cases:
  - Test initialize() when already initialized (running=True) - should not re-initialize
  - Test run() called multiple times concurrently - verify thread safety
  - Test shutdown() called multiple times - verify idempotent behavior
  - Test system behavior when API server is disabled vs enabled
  - Test system behavior when tracking_manager or mqtt_manager fail to initialize
  - Test main loop interruption at various sleep intervals
  - Test exception handling during component cleanup in shutdown()

- Error Handling:
  - Test database connection failures during initialize() - verify proper cleanup and exception re-raising
  - Test MQTT manager initialization failures - verify system state and cleanup
  - Test tracking manager initialization failures - verify proper error handling
  - Test API server status check failures - verify graceful handling
  - Test logging failures during system operations - verify system continues
  - Test shutdown() exceptions from component cleanup - verify other components still get cleaned up
  - Test run() exception propagation after shutdown completion
  - Test system behavior with corrupted or missing config during initialization

- Coverage Target: 85%+

### src/adaptation/drift_detector.py - Concept Drift Detection
**Classes Found:** 
- DriftType (Enum)
- DriftSeverity (Enum) 
- StatisticalTest (Enum)
- DriftMetrics (dataclass)
- FeatureDriftResult (dataclass)
- ConceptDriftDetector
- FeatureDriftDetector
- DriftDetectionError (Exception)

**Methods Analyzed:**
- DriftMetrics.__post_init__()
- DriftMetrics._calculate_overall_drift_score()
- DriftMetrics._determine_drift_severity()
- DriftMetrics._generate_recommendations()
- DriftMetrics.update_recommendations()
- DriftMetrics.to_dict()
- FeatureDriftResult.is_significant()
- ConceptDriftDetector.__init__()
- ConceptDriftDetector.detect_drift()
- ConceptDriftDetector._analyze_prediction_drift()
- ConceptDriftDetector._analyze_feature_drift()
- ConceptDriftDetector._test_feature_drift()
- ConceptDriftDetector._test_numerical_drift()
- ConceptDriftDetector._test_categorical_drift()
- ConceptDriftDetector._calculate_psi()
- ConceptDriftDetector._calculate_numerical_psi()
- ConceptDriftDetector._calculate_categorical_psi()
- ConceptDriftDetector._analyze_pattern_drift()
- ConceptDriftDetector._run_page_hinkley_test()
- ConceptDriftDetector._calculate_statistical_confidence()
- ConceptDriftDetector._get_feature_data()
- ConceptDriftDetector._get_occupancy_patterns()
- ConceptDriftDetector._compare_temporal_patterns()
- ConceptDriftDetector._compare_frequency_patterns()
- ConceptDriftDetector._get_recent_prediction_errors()
- FeatureDriftDetector.__init__()
- FeatureDriftDetector.start_monitoring()
- FeatureDriftDetector.stop_monitoring()
- FeatureDriftDetector.detect_feature_drift()
- FeatureDriftDetector._test_single_feature_drift()
- FeatureDriftDetector._test_numerical_feature_drift()
- FeatureDriftDetector._test_categorical_feature_drift()
- FeatureDriftDetector._monitoring_loop()
- FeatureDriftDetector._get_recent_feature_data()
- FeatureDriftDetector.add_drift_callback()
- FeatureDriftDetector.remove_drift_callback()
- FeatureDriftDetector._notify_drift_callbacks()

**Required Tests:**
- Unit Tests: 
  - Enum value validation for DriftType, DriftSeverity, StatisticalTest
  - DriftMetrics initialization with all parameters and post_init calculations
  - DriftMetrics drift score calculation with various input combinations
  - DriftMetrics severity determination logic with edge cases
  - DriftMetrics recommendation generation for different scenarios
  - DriftMetrics.to_dict() serialization correctness
  - FeatureDriftResult initialization and is_significant() method
  - ConceptDriftDetector initialization with default and custom parameters
  - ConceptDriftDetector.detect_drift() with mock prediction validator
  - Statistical test methods (_test_numerical_drift, _test_categorical_drift) with sample data
  - PSI calculation methods with numerical and categorical data
  - Page-Hinkley test implementation with various error sequences
  - Pattern drift analysis with mock occupancy data
  - Statistical confidence calculation logic
  - FeatureDriftDetector initialization and configuration
  - Feature drift detection with sample DataFrame inputs
  - Monitoring start/stop functionality
  - Callback system for drift notifications
  
- Integration Tests:
  - ConceptDriftDetector with real PredictionValidator instance
  - Database integration for occupancy pattern retrieval
  - Feature engineering integration for feature data retrieval
  - End-to-end drift detection workflow from raw data to recommendations
  - FeatureDriftDetector continuous monitoring with background tasks
  - Integration with database models (SensorEvent, Prediction)
  - Cross-component integration with prediction validation system
  
- Edge Cases:
  - Empty or insufficient data samples for statistical tests
  - NaN/infinite values in feature data
  - Missing timestamps in feature data
  - Zero variance in baseline or current data distributions
  - Extreme PSI values causing numerical instability
  - Page-Hinkley test with constant error values
  - Feature data with all categorical or all numerical features
  - Monitoring system behavior when database is unavailable
  - Callback exceptions during drift notifications
  - Memory management with large feature datasets
  
- Error Handling:
  - Database connection failures during pattern retrieval
  - Invalid room_id parameters
  - Malformed feature data inputs
  - Statistical test failures due to data issues
  - Asyncio task cancellation during monitoring
  - Exception handling in drift callback functions
  - Error propagation from prediction validator
  - Timeout handling for long-running statistical calculations
  - Memory overflow with very large datasets
  - Network connectivity issues affecting database operations
  
- Coverage Target: 85%+