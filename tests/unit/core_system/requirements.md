# Core System Testing Requirements

## Overview
This document contains detailed testing requirements for the ha-ml-predictor core system components to achieve 85%+ test coverage. Each component has been analyzed for actual implementation details and specific testing scenarios.

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

### src/core/exceptions.py - Custom Exception Classes
**Classes Found:** ErrorSeverity, OccupancyPredictionError, ConfigurationError, ConfigFileNotFoundError, ConfigValidationError, MissingConfigSectionError, ConfigParsingError, HomeAssistantError, HomeAssistantConnectionError, HomeAssistantAuthenticationError, HomeAssistantAPIError, EntityNotFoundError, WebSocketError, WebSocketConnectionError, WebSocketAuthenticationError, WebSocketValidationError, DatabaseError, DatabaseConnectionError, DatabaseQueryError, DatabaseMigrationError, DatabaseIntegrityError, FeatureEngineeringError, FeatureExtractionError, InsufficientDataError, FeatureValidationError, FeatureStoreError, ModelError, ModelTrainingError, ModelPredictionError, InsufficientTrainingDataError, ModelNotFoundError, ModelVersionMismatchError, MissingFeatureError, ModelValidationError, DataProcessingError, DataCorruptionError, IntegrationError, DataValidationError, MQTTError, MQTTConnectionError, MQTTPublishError, MQTTSubscriptionError, APIServerError, SystemInitializationError, SystemResourceError, SystemError, ResourceExhaustionError, ServiceUnavailableError, MaintenanceModeError, APIError, APIAuthenticationError, RateLimitExceededError, APIAuthorizationError, APISecurityError, APIResourceNotFoundError
**Methods Analyzed:** OccupancyPredictionError.__init__(), OccupancyPredictionError.__str__(), ConfigurationError.__init__(), HomeAssistantAuthenticationError.__init__(), DatabaseConnectionError._mask_password(), DatabaseQueryError.__init__(), DataValidationError.__init__(), validate_room_id(), validate_entity_id()

**Required Tests:**
- Unit Tests: Base exception functionality, error severity enumeration, context and cause handling, string formatting, inheritance hierarchy, specialized exception parameters, password masking, validation functions
- Integration Tests: Exception chaining, error propagation through system layers, configuration error scenarios, logging integration
- Edge Cases: Very long error messages, Unicode characters, nested context data, circular references, extreme parameter values
- Error Handling: Exception creation failures, serialization issues, logging failures
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