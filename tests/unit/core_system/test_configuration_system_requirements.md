# Configuration System Testing Requirements

## Coverage Target: 85%+

## Covered Source Files:
- src/core/config.py (Configuration Management)
- src/core/config_validator.py (Configuration Validation)
- src/core/environment.py (Environment Management)

## Classes and Methods Covered:

### src/core/config.py
**Classes Found:** HomeAssistantConfig, DatabaseConfig, MQTTConfig, PredictionConfig, FeaturesConfig, LoggingConfig, TrackingConfig, JWTConfig, APIConfig, SensorConfig, RoomConfig, SystemConfig, ConfigLoader

**Methods Analyzed:** DatabaseConfig.__post_init__(), RoomConfig.get_all_entity_ids(), RoomConfig.get_sensors_by_type(), SystemConfig.get_all_entity_ids(), SystemConfig.get_room_by_entity_id(), JWTConfig.__post_init__(), APIConfig.__post_init__(), TrackingConfig.__post_init__(), ConfigLoader.__init__(), ConfigLoader.load_config(), ConfigLoader._load_yaml(), ConfigLoader._create_system_config(), get_config(), reload_config()

## Required Tests:

### Unit Tests:

#### Dataclass Configuration Objects:
- Test HomeAssistantConfig initialization with required/optional parameters
- Test DatabaseConfig.__post_init__() with DATABASE_URL environment variable override
- Test MQTTConfig initialization with extensive MQTT configuration options
- Test PredictionConfig, FeaturesConfig, LoggingConfig with default values
- Test TrackingConfig.__post_init__() with default alert_thresholds creation
- Test SensorConfig with entity_id, sensor_type, room_id parameters

#### JWT Configuration Tests:
- Test JWTConfig.__post_init__() with JWT_ENABLED environment variable ("false", "0", "no", "off")
- Test JWTConfig secret key loading from JWT_SECRET_KEY environment variable
- Test JWTConfig test environment fallback with default test secret key
- Test JWTConfig secret key length validation (minimum 32 characters)
- Test JWTConfig ValueError for missing secret key in non-test environments
- Test JWTConfig environment detection (test, CI environments)

#### API Configuration Tests:
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

#### Room Configuration Tests:
- Test RoomConfig.get_all_entity_ids() with nested dictionary sensor structures
- Test RoomConfig.get_all_entity_ids() with list-based sensor structures
- Test RoomConfig.get_all_entity_ids() entity ID extraction (binary_sensor., sensor. prefixes)
- Test RoomConfig.get_sensors_by_type() with dictionary sensor type
- Test RoomConfig.get_sensors_by_type() with string sensor type
- Test RoomConfig.get_sensors_by_type() with missing sensor type
- Test RoomConfig recursive extraction from nested objects

#### System Configuration Tests:
- Test SystemConfig.get_all_entity_ids() aggregation from all rooms
- Test SystemConfig.get_all_entity_ids() duplicate removal (set conversion)
- Test SystemConfig.get_room_by_entity_id() entity lookup across rooms
- Test SystemConfig.get_room_by_entity_id() with non-existent entity (None return)

#### ConfigLoader Tests:
- Test ConfigLoader.__init__() with valid config directory
- Test ConfigLoader.__init__() with missing config directory (FileNotFoundError)
- Test ConfigLoader._load_yaml() with valid YAML files
- Test ConfigLoader._load_yaml() with missing files (FileNotFoundError)
- Test ConfigLoader._load_yaml() with invalid YAML content
- Test ConfigLoader._load_yaml() return type handling (dict vs non-dict)

#### Configuration Loading Tests:
- Test ConfigLoader.load_config() with environment parameter
- Test ConfigLoader.load_config() environment-specific config loading (config.{environment}.yaml)
- Test ConfigLoader.load_config() fallback to base config.yaml when environment config missing
- Test ConfigLoader.load_config() rooms.yaml integration
- Test ConfigLoader.load_config() nested room structure handling (hallways example)
- Test ConfigLoader.load_config() regular room structure processing
- Test ConfigLoader.load_config() dataclass object creation for all config sections

#### Global Configuration Tests:
- Test get_config() singleton behavior with global _config_instance
- Test get_config() environment manager integration (when available)
- Test get_config() ImportError fallback to direct ConfigLoader
- Test reload_config() forced reload behavior
- Test reload_config() environment manager vs direct loading paths

### Integration Tests:

#### File System Integration:
- Test complete configuration loading from actual YAML files
- Test environment-specific configuration override behavior
- Test configuration loading with complex nested room structures
- Test environment variable integration across all configuration objects
- Test configuration validation with environment manager integration

#### Cross-Configuration Integration:
- Test SystemConfig with all nested configuration objects
- Test entity ID aggregation across multiple room configurations
- Test room lookup functionality with realistic sensor mappings
- Test JWT and API configuration interaction

#### Environment Manager Integration:
- Test get_config() with environment manager secret injection
- Test configuration processing through environment manager
- Test fallback behavior when environment manager unavailable

### Edge Cases:

#### Configuration File Edge Cases:
- Test configuration loading with empty YAML files
- Test configuration loading with malformed YAML syntax
- Test configuration loading with missing required sections
- Test configuration loading with Unicode characters in configuration values
- Test configuration loading with very large configuration files
- Test configuration loading with deeply nested room structures

#### Environment Variable Edge Cases:
- Test environment variable handling with empty string values
- Test environment variable handling with case sensitivity variations
- Test environment variable parsing with invalid values (non-numeric for numeric fields)
- Test environment variable boolean parsing with various true/false representations
- Test environment variable list parsing (comma-separated values)

#### Room Configuration Edge Cases:
- Test room configuration with no sensors defined
- Test room configuration with empty sensor dictionaries
- Test room configuration with mixed sensor type definitions (strings vs dicts)
- Test room configuration with circular references in nested structures
- Test room configuration with null/None values in sensor definitions

#### Configuration Object Edge Cases:
- Test configuration objects with extreme parameter values (very large timeouts, pools)
- Test configuration objects with boundary values (minimum/maximum allowed)
- Test configuration objects with conflicting parameter combinations
- Test configuration objects with missing optional parameters

### Error Handling:

#### File System Errors:
- Test ConfigLoader behavior with permission denied errors
- Test ConfigLoader behavior with corrupted YAML files
- Test ConfigLoader behavior with symbolic link issues
- Test ConfigLoader behavior when config directory becomes unavailable

#### Configuration Validation Errors:
- Test JWTConfig validation failures (short secret key, missing secret)
- Test APIConfig validation failures (missing API key when enabled)
- Test DatabaseConfig validation with invalid connection strings
- Test configuration object validation with invalid parameter types

#### Environment Integration Errors:
- Test environment manager import failures
- Test environment manager configuration processing failures
- Test secret injection failures in environment manager integration

#### Data Processing Errors:
- Test room configuration processing with corrupted sensor data
- Test entity ID extraction with malformed entity identifiers
- Test configuration object creation with invalid parameter combinations

## Mock Requirements:
- Mock Path.exists() and Path() for filesystem operations
- Mock open() and file operations for YAML loading
- Mock yaml.safe_load() for configuration parsing
- Mock os.getenv() for environment variable testing
- Mock environment manager import and methods
- Mock print() for test environment JWT warning messages

## Test Fixtures Needed:
- Sample configuration YAML files (base and environment-specific)
- Sample rooms.yaml with various room structures (simple, nested, hallways)
- Environment variable fixtures for all configuration objects
- Invalid YAML content fixtures for error testing
- Complex sensor configuration fixtures for room testing

## Special Testing Considerations:
- Test dataclass field validation and default values
- Test __post_init__ method behavior across all configuration classes
- Test singleton pattern behavior for global configuration
- Test recursive sensor extraction from nested room structures
- Test environment-specific configuration override behavior
- Test secret key security validation in JWT configuration
- Test comprehensive environment variable parsing across all config objects