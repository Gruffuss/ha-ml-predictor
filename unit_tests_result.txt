========================== short test summary info ============================
FAILED tests/unit/test_adaptation/test_retrainer.py::TestBackgroundTasks::test_retraining_processor_loop - assert 0 > 0
FAILED tests/unit/test_adaptation/test_optimizer.py::TestOptimizationStrategies::test_bayesian_optimization - AssertionError: assert 0 > 0
 +  where 0 = OptimizationResult(success=False, optimization_time_seconds=0.000939, best_parameters={}, best_score=0.0, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=False, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No valid dimensions for optimization').total_evaluations
FAILED tests/unit/test_adaptation/test_optimizer.py::TestOptimizationConstraints::test_performance_constraint_validation - AssertionError: assert False
 +  where False = OptimizationResult(success=False, optimization_time_seconds=0.000586, best_parameters={}, best_score=0.0, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=False, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No valid dimensions for optimization').success
FAILED tests/unit/test_adaptation/test_retrainer.py::TestRetrainingNeedEvaluation::test_cooldown_period_enforcement - AssertionError: assert RetrainingRequest(request_id='bathroom_lstm_1755421953', room_id='bathroom', model_type=<ModelType.LSTM: 'lstm'>, trigger=<RetrainingTrigger.ACCURACY_DEGRADATION: 'accuracy_degradation'>, strategy=<RetrainingStrategy.INCREMENTAL: 'incremental'>, priority=6.0, created_time=datetime.datetime(2025, 8, 17, 9, 12, 33, 583663), accuracy_metrics=AccuracyMetrics(total_predictions=100, validated_predictions=85, accurate_predictions=45, expired_predictions=0, failed_predictions=0, accuracy_rate=52.9, mean_error_minutes=28.5, median_error_minutes=25.0, std_error_minutes=0.0, rmse_minutes=0.0, mae_minutes=0.0, error_percentiles={}, accuracy_by_level={}, mean_bias_minutes=0.0, bias_std_minutes=0.0, mean_confidence=0.0, confidence_accuracy_correlation=0.68, overconfidence_rate=0.0, underconfidence_rate=0.0, measurement_period_start=datetime.datetime(2025, 8, 16, 9, 12, 33, 582898), measurement_period_end=datetime.datetime(2025, 8, 17, 9, 12, 33, 582904), predictions_per_hour=0.0), drift_metrics=None, performance_degradation={}, retraining_parameters={'lookback_days': 14, 'validation_split': 0.2, 'feature_refresh': True, 'max_training_time_minutes': 60, 'early_stopping_patience': 10, 'min_improvement_threshold': 0.01}, model_hyperparameters={}, feature_engineering_config={}, validation_strategy=['time_series_split', 'holdout'], status=<RetrainingStatus.PENDING: 'pending'>, started_time=None, completed_time=None, error_message=None, execution_log=[], resource_usage_log=[], checkpoint_data={}, training_result=None, performance_improvement={}, prediction_results=[], validation_metrics={}, lookback_days=14, validation_split=0.2, feature_refresh=True) is None
FAILED tests/unit/test_adaptation/test_retrainer.py::TestRetrainingNeedEvaluation::test_retraining_strategy_selection - AssertionError: assert <RetrainingStrategy.INCREMENTAL: 'incremental'> == <RetrainingStrategy.FULL_RETRAIN: 'full_retrain'>
 +  where <RetrainingStrategy.INCREMENTAL: 'incremental'> = RetrainingRequest(request_id='living_room_xgboost_1755421953', room_id='living_room', model_type=<ModelType.XGBOOST: 'xgboost'>, trigger=<RetrainingTrigger.ACCURACY_DEGRADATION: 'accuracy_degradation'>, strategy=<RetrainingStrategy.INCREMENTAL: 'incremental'>, priority=6.0, created_time=datetime.datetime(2025, 8, 17, 9, 12, 33, 635746), accuracy_metrics=AccuracyMetrics(total_predictions=0, validated_predictions=0, accurate_predictions=0, expired_predictions=0, failed_predictions=0, accuracy_rate=45.0, mean_error_minutes=35.0, median_error_minutes=0.0, std_error_minutes=0.0, rmse_minutes=0.0, mae_minutes=0.0, error_percentiles={}, accuracy_by_level={}, mean_bias_minutes=0.0, bias_std_minutes=0.0, mean_confidence=0.0, confidence_accuracy_correlation=0.0, overconfidence_rate=0.0, underconfidence_rate=0.0, measurement_period_start=None, measurement_period_end=None, predictions_per_hour=0.0), drift_metrics=None, performance_degradation={}, retraining_parameters={'lookback_days': 14, 'validation_split': 0.2, 'feature_refresh': True, 'max_training_time_minutes': 60, 'early_stopping_patience': 10, 'min_improvement_threshold': 0.01}, model_hyperparameters={}, feature_engineering_config={}, validation_strategy=['time_series_split', 'holdout'], status=<RetrainingStatus.PENDING: 'pending'>, started_time=None, completed_time=None, error_message=None, execution_log=[], resource_usage_log=[], checkpoint_data={}, training_result=None, performance_improvement={}, prediction_results=[], validation_metrics={}, lookback_days=14, validation_split=0.2, feature_refresh=True).strategy
 +  and   <RetrainingStrategy.FULL_RETRAIN: 'full_retrain'> = RetrainingStrategy.FULL_RETRAIN
FAILED tests/unit/test_adaptation/test_retrainer.py::TestRetrainingRequestManagement::test_retraining_queue_priority_ordering - assert [9.0, 7.0, 3.0, 5.0] == [9.0, 7.0, 5.0, 3.0]
  At index 2 diff: 3.0 != 5.0
  Full diff:
  - [9.0, 7.0, 5.0, 3.0]
  ?               -----
  + [9.0, 7.0, 3.0, 5.0]
  ?            +++++
FAILED tests/unit/test_adaptation/test_retrainer.py::TestRetrainingExecution::test_retraining_with_optimization - AssertionError: assert <RetrainingStatus.FAILED: 'failed'> == <RetrainingStatus.COMPLETED: 'completed'>
 +  where <RetrainingStatus.FAILED: 'failed'> = RetrainingRequest(request_id='test_optimization_001', room_id='optimization_room', model_type=<ModelType.LSTM: 'lstm'>, trigger=<RetrainingTrigger.ACCURACY_DEGRADATION: 'accuracy_degradation'>, strategy=<RetrainingStrategy.FULL_RETRAIN: 'full_retrain'>, priority=8.0, created_time=datetime.datetime(2025, 8, 17, 9, 12, 34, 105089), accuracy_metrics=None, drift_metrics=None, performance_degradation={}, retraining_parameters={'lookback_days': 14, 'validation_split': 0.2, 'feature_refresh': True, 'max_training_time_minutes': 60, 'early_stopping_patience': 10, 'min_improvement_threshold': 0.01}, model_hyperparameters={}, feature_engineering_config={}, validation_strategy=['time_series_split', 'holdout'], status=<RetrainingStatus.FAILED: 'failed'>, started_time=datetime.datetime(2025, 8, 17, 9, 12, 34, 105128), completed_time=datetime.datetime(2025, 8, 17, 9, 12, 34, 107171), error_message='Model optimization_room_lstm not found in registry | Error Code: RETRAINING_ERROR', execution_log=[], resource_usage_log=[], checkpoint_data={}, training_result=None, performance_improvement={}, prediction_results=[], validation_metrics={}, lookback_days=14, validation_split=0.2, feature_refresh=True).status
 +  and   <RetrainingStatus.COMPLETED: 'completed'> = RetrainingStatus.COMPLETED
FAILED tests/unit/test_adaptation/test_retrainer.py::TestRetrainingProgressTracking::test_progress_reporting - KeyError: 'progress_percentage'
FAILED tests/unit/test_adaptation/test_validator.py::TestErrorHandlingAndEdgeCases::test_memory_usage_with_large_datasets - AttributeError: 'PredictionValidator' object has no attribute '_validation_history'. Did you mean: '_validation_records'?
FAILED tests/unit/test_core/test_config.py::TestSystemConfig::test_system_config_creation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestSystemConfig::test_get_all_entity_ids - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestSystemConfig::test_get_all_entity_ids_with_duplicates - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestSystemConfig::test_get_room_by_entity_id - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestConfigLoader::test_load_config_success - AssertionError: assert 'sqlite+aiosqlite:///:memory:' == 'postgresql+a...alhost/testdb'
  - postgresql+asyncpg://localhost/testdb
  + sqlite+aiosqlite:///:memory:
FAILED tests/unit/test_core/test_config.py::TestGlobalConfigFunctions::test_get_config_singleton - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestGlobalConfigFunctions::test_reload_config - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
FAILED tests/unit/test_core/test_config.py::TestConfigIntegration::test_load_real_config_structure - AssertionError: assert 'postgresql' in 'sqlite+aiosqlite:///:memory:'
 +  where 'sqlite+aiosqlite:///:memory:' = DatabaseConfig(connection_string='sqlite+aiosqlite:///:memory:', pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=3600).connection_string
 +    where DatabaseConfig(connection_string='sqlite+aiosqlite:///:memory:', pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=3600) = SystemConfig(home_assistant=HomeAssistantConfig(url='http://test-ha:8123', token='test_token_12345', websocket_timeout=30, api_timeout=10), database=DatabaseConfig(connection_string='sqlite+aiosqlite:///:memory:', pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=3600), mqtt=MQTTConfig(broker='test-mqtt', port=1883, username='test_user', password='test_pass', topic_prefix='test/occupancy', discovery_enabled=True, discovery_prefix='homeassistant', device_name='Occupancy Predictor', device_identifier='ha_ml_predictor', device_manufacturer='HA ML Predictor', device_model='Smart Room Occupancy Predictor', device_sw_version='1.0.0', publishing_enabled=True, publish_system_status=True, status_update_interval_seconds=300, prediction_qos=1, system_qos=0, retain_predictions=True, retain_system_status=True, keepalive=60, connection_timeout=30, reconnect_delay_seconds=5, max_reconnect_attempts=-1), prediction=PredictionConfig(interval_seconds=300, accuracy_threshold_minutes=15, confidence_threshold=0.7), features=FeaturesConfig(lookback_hours=24, sequence_length=50, temporal_features=True, sequential_features=True, contextual_features=True), logging=LoggingConfig(level='DEBUG', fo...ey='test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long', algorithm='HS256', access_token_expire_minutes=60, refresh_token_expire_days=30, issuer='ha-ml-predictor', audience='ha-ml-predictor-api', require_https=False, secure_cookies=False, blacklist_enabled=True), rate_limit_enabled=True, requests_per_minute=60, burst_limit=100, request_timeout_seconds=30, max_request_size_mb=10, include_docs=True, docs_url='/docs', redoc_url='/redoc', background_tasks_enabled=True, health_check_interval_seconds=60, access_log=True, log_requests=True, log_responses=False), rooms={'living_room': RoomConfig(room_id='living_room', name='Living Room', sensors={'climate': {'temperature': 'sensor.living_room_temperature'}, 'presence': {'couch': 'binary_sensor.living_room_couch', 'main': 'binary_sensor.living_room_presence'}}), 'test_room': RoomConfig(room_id='test_room', name='Test Room', sensors={'climate': {'humidity': 'sensor.test_room_humidity', 'temperature': 'sensor.test_room_temperature'}, 'door': 'binary_sensor.test_room_door', 'light': 'sensor.test_room_light', 'presence': {'main': 'binary_sensor.test_room_presence', 'secondary': 'binary_sensor.test_room_motion'}})}).database
FAILED tests/unit/test_core/test_exceptions.py::TestHomeAssistantErrors::test_entity_not_found_error - AssertionError: assert 'ENTITY_NOT_FOUND_ERROR' == 'ENTITY_NOT_FOUND'
  - ENTITY_NOT_FOUND
  + ENTITY_NOT_FOUND_ERROR
  ?                 ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestModelErrors::test_insufficient_training_data_error - AssertionError: assert 'INSUFFICIENT...NG_DATA_ERROR' == 'INSUFFICIENT_TRAINING_DATA'
  - INSUFFICIENT_TRAINING_DATA
  + INSUFFICIENT_TRAINING_DATA_ERROR
  ?                           ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestModelErrors::test_model_version_mismatch_error - AssertionError: assert 'MODEL_VERSION_MISMATCH_ERROR' == 'MODEL_VERSION_MISMATCH'
  - MODEL_VERSION_MISMATCH
  + MODEL_VERSION_MISMATCH_ERROR
  ?                       ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestMQTTAndIntegrationErrors::test_rate_limit_exceeded_error - AssertionError: assert 'RATE_LIMIT_EXCEEDED_ERROR' == 'RATE_LIMIT_EXCEEDED'
  - RATE_LIMIT_EXCEEDED
  + RATE_LIMIT_EXCEEDED_ERROR
  ?                    ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_resource_exhaustion_error - AssertionError: assert 'RESOURCE_EXHAUSTION_ERROR' == 'RESOURCE_EXHAUSTION'
  - RESOURCE_EXHAUSTION
  + RESOURCE_EXHAUSTION_ERROR
  ?                    ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_service_unavailable_error - AssertionError: assert 'SERVICE_UNAVAILABLE_ERROR' == 'SERVICE_UNAVAILABLE'
  - SERVICE_UNAVAILABLE
  + SERVICE_UNAVAILABLE_ERROR
  ?                    ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_maintenance_mode_error - AssertionError: assert 'MAINTENANCE_MODE_ERROR' == 'MAINTENANCE_MODE'
  - MAINTENANCE_MODE
  + MAINTENANCE_MODE_ERROR
  ?                 ++++++
FAILED tests/unit/test_core/test_exceptions.py::TestExceptionIntegration::test_error_code_uniqueness - AssertionError: assert False
 +  where False = <built-in method endswith of str object at 0x7f99fcca15b0>('_ERROR')
 +    where <built-in method endswith of str object at 0x7f99fcca15b0> = 'CONFIG_FILE_NOT_FOUND'.endswith
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_initialize_success - assert False
 +  where False = <src.data.storage.database.DatabaseManager object at 0x7f99c045ffe0>.is_initialized
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_verify_connection_success - AttributeError: __aenter__
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_get_session_non_connection_error - TypeError: DatabaseQueryError.__init__() got an unexpected keyword argument 'severity'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_execute_query_success - TypeError: DatabaseQueryError.__init__() got an unexpected keyword argument 'severity'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_execute_query_error - TypeError: DatabaseQueryError.__init__() got an unexpected keyword argument 'severity'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_healthy - AssertionError: assert 'unhealthy' == 'healthy'
  - healthy
  + unhealthy
  ? ++
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_timescaledb_available - AssertionError: assert None == 'available'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_timescaledb_unavailable - AssertionError: assert None == 'unavailable'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_timescaledb_version_parsing - AttributeError: 'TestDatabaseManager' object has no attribute 'subTest'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_health_check_timescaledb_version_parsing_error - AssertionError: assert None == 'available'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManager::test_close_cleanup - TypeError: object Mock can't be used in 'await' expression
FAILED tests/unit/test_data/test_database.py::TestGlobalDatabaseFunctions::test_get_db_session - TypeError: 'async_generator' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestTrackingManagerInitialization::test_manager_shutdown - assert not True
 +  where True = <src.adaptation.tracking_manager.TrackingManager object at 0x7f06c128e300>._tracking_active
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerEdgeCases::test_verify_connection_timescaledb_warning - AttributeError: __aenter__
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerEdgeCases::test_get_session_rollback_on_error - TypeError: DatabaseQueryError.__init__() got an unexpected keyword argument 'severity'
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerEdgeCases::test_health_check_with_previous_errors - AssertionError: assert 'unhealthy' == 'healthy'
  - healthy
  + unhealthy
  ? ++
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerIntegration::test_full_lifecycle - src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestPredictionRecording::test_prediction_recording - assert 0 > 0
 +  where 0 = <src.adaptation.tracking_manager.TrackingManager object at 0x7f06c12b8b00>._total_predictions_recorded
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerIntegration::test_concurrent_sessions - src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
FAILED tests/unit/test_data/test_database.py::TestDatabaseManagerIntegration::test_retry_mechanism_with_real_errors - src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
FAILED tests/unit/test_data/test_models.py::TestSensorEvent::test_sensor_event_minimal - assert None is True
 +  where None = <src.data.storage.models.SensorEvent object at 0x7f99b9c80770>.is_human_triggered
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestPredictionRecording::test_prediction_mqtt_integration - AssertionError: Expected 'publish_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_data/test_models.py::TestSensorEvent::test_get_recent_events - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id]
[parameters: ('2025-08-17 08:12:38.827731', 'test_room', 'binary_sensor.test_sensor_0', 'presence', 'on', 'of', '{"test": true, "sequence": 0}', 1, 0.8, '2025-08-17 09:12:38.827755', None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestSensorEvent::test_get_recent_events_with_sensor_filter - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id]
[parameters: ('2025-08-17 08:12:39.724428', 'test_room', 'binary_sensor.test_sensor_0', 'presence', 'on', 'of', '{"test": true, "sequence": 0}', 1, 0.8, '2025-08-17 09:12:39.724452', None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestRoomStateChangeHandling::test_room_state_change_handling - assert 0 > 0
 +  where 0 = <src.adaptation.tracking_manager.TrackingManager object at 0x7f06c188a420>._total_validations_performed
FAILED tests/unit/test_data/test_models.py::TestSensorEvent::test_get_state_changes - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('2025-08-17 08:12:40.600641', 'test_room', 'binary_sensor.test', 'motion', 'on', 'of', '{}', 1, None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestSensorEvent::test_get_transition_sequences - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('2025-08-17 08:12:41.456144', 'test_room', 'binary_sensor.sensor_0', 'motion', 'on', 'of', '{}', 1, None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestRoomState::test_get_current_state - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: room_states.id
[SQL: INSERT INTO room_states (room_id, timestamp, occupancy_session_id, is_occupied, occupancy_confidence, occupant_type, occupant_count, state_duration, transition_trigger, certainty_factors, detailed_analysis, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('office', '2025-08-17 07:12:42.334721', None, 0, 0.7, None, 1, None, None, '{}', None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestDriftDetectionIntegration::test_manual_drift_detection - assert None is not None
FAILED tests/unit/test_data/test_models.py::TestRoomState::test_get_occupancy_history - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: room_states.id
[SQL: INSERT INTO room_states (room_id, timestamp, occupancy_session_id, is_occupied, occupancy_confidence, occupant_type, occupant_count, state_duration, transition_trigger, certainty_factors, detailed_analysis, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('bedroom', '2025-08-16 08:12:43.371651', None, 1, 0.7, None, 1, None, None, '{}', None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestPrediction::test_get_pending_validations - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: predictions.id
[SQL: INSERT INTO predictions (room_id, prediction_time, predicted_transition_time, transition_type, confidence_score, prediction_interval_lower, prediction_interval_upper, model_type, model_version, feature_importance, alternatives, actual_transition_time, accuracy_minutes, is_accurate, validation_timestamp, triggering_event_id, room_state_id, created_at, processing_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('living_room', '2025-08-17 06:12:44.257529', '2025-08-17 08:42:44.257529', 'occupied_to_vacant', 0.8, None, None, 'xgboost', 'v1.0', '{}', '[]', None, None, None, None, None, None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestPrediction::test_get_accuracy_metrics - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: predictions.id
[SQL: INSERT INTO predictions (room_id, prediction_time, predicted_transition_time, transition_type, confidence_score, prediction_interval_lower, prediction_interval_upper, model_type, model_version, feature_importance, alternatives, actual_transition_time, accuracy_minutes, is_accurate, validation_timestamp, triggering_event_id, room_state_id, created_at, processing_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('office', '2025-08-16 09:12:45.149258', '2025-08-16 10:12:45.149258', 'occupied_to_vacant', 0.9, None, None, 'lstm', 'v1.0', '{}', '[]', '2025-08-16 10:07:45.149258', 5.0, 1, '2025-08-16 11:12:45.149258', None, None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestFeatureStore::test_get_latest_features - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: feature_store.id
[SQL: INSERT INTO feature_store (room_id, feature_timestamp, temporal_features, sequential_features, contextual_features, environmental_features, lookback_hours, feature_version, computation_time_ms, completeness_score, freshness_score, confidence_score, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('kitchen', '2025-08-17 01:12:46.009100', '{"hour_sin": 0.0}', '{}', '{}', '{}', 24, 'v1.0', None, None, None, None, '2025-08-17 07:12:46.009100')]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestRetrainingIntegration::test_manual_retraining_request - assert None is not None
FAILED tests/unit/test_data/test_models.py::TestModelApplicationLevelRelationships::test_sensor_event_prediction_application_relationship - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('2025-08-17 09:12:46.917730', 'test_room', 'binary_sensor.test', 'motion', 'on', None, '{}', 1, None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestRetrainingIntegration::test_retraining_status_tracking - assert None is not None
FAILED tests/unit/test_data/test_models.py::TestModelApplicationLevelRelationships::test_room_state_prediction_application_relationship - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: room_states.id
[SQL: INSERT INTO room_states (room_id, timestamp, occupancy_session_id, is_occupied, occupancy_confidence, occupant_type, occupant_count, state_duration, transition_trigger, certainty_factors, detailed_analysis, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('test_room', '2025-08-17 09:12:47.665995', None, 1, 0.9, None, 1, None, None, '{}', None, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_models.py::TestModelConstraints::test_confidence_score_constraints - Failed: Valid confidence score should not raise IntegrityError
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestRetrainingIntegration::test_retraining_cancellation - assert False
FAILED tests/unit/test_data/test_models.py::TestModelConstraints::test_model_accuracy_constraints - Failed: Valid accuracy data should not raise IntegrityError
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestSystemStatusAndMetrics::test_tracking_status_comprehensive - assert 'tracking_active' in {'error': "object Mock can't be used in 'await' expression"}
FAILED tests/unit/test_data/test_models.py::TestModelIntegration::test_complete_prediction_workflow - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: sensor_events.id
[SQL: INSERT INTO sensor_events (timestamp, room_id, sensor_id, sensor_type, state, previous_state, attributes, is_human_triggered, confidence_score, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('2025-08-17 08:42:50.363858', 'integration_test_room', 'binary_sensor.test_motion', 'motion', 'on', 'of', '{}', 1, 0.85, None)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestSystemStatusAndMetrics::test_real_time_metrics_retrieval - assert None is not None
FAILED tests/unit/test_data/test_models.py::TestModelIntegration::test_feature_store_lifecycle - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: feature_store.id
[SQL: INSERT INTO feature_store (room_id, feature_timestamp, temporal_features, sequential_features, contextual_features, environmental_features, lookback_hours, feature_version, computation_time_ms, completeness_score, freshness_score, confidence_score, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) RETURNING id, created_at]
[parameters: ('lifecycle_test_room', '2025-08-17 08:42:51.127758', '{"hour_sin": 0.5}', '{}', '{}', '{}', 24, 'v1.0', None, 0.95, 0.9, 0.85, '2025-08-17 11:12:51.127758')]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_features/test_contextual.py::TestContextualFeatureExtractor::test_error_handling - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestSystemStatusAndMetrics::test_active_alerts_retrieval - assert 0 == 2
 +  where 0 = len([])
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestSystemStatusAndMetrics::test_alert_acknowledgment - assert False
FAILED tests/unit/test_features/test_contextual.py::TestContextualFeatureExtractor::test_natural_light_patterns - assert 1.0 == 0.0
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_extract_features_parallel - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_extract_features_sequential - AssertionError: Expected 'extract_features' to have been called once. Called 0 times.
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_error_handling_invalid_room_id - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_error_handling_extractor_failure - Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_validate_configuration_no_config - assert True is False
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_parallel_vs_sequential_consistency - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_extractor_partial_failure_handling - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_large_feature_set_handling - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_performance_comparison[True] - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_engineering.py::TestFeatureEngineeringEngine::test_performance_comparison[False] - assert 0.014484882354736328 > 0.025
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractor::test_extract_features_multi_room - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractor::test_extract_features_single_room - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractor::test_empty_room_configs - AssertionError: assert 'human_movement_probability' in {'active_room_count': 2, 'avg_event_interval': 106.66666666666667, 'avg_room_dwell_time': 320.0, 'burst_ratio': 0.0, ...}
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractor::test_no_classifier_available - KeyError: 'human_movement_probability'
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractor::test_error_handling - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_sequential.py::TestSequentialFeatureExtractorMovementPatterns::test_cat_like_patterns - assert 0.4 > 0.5
FAILED tests/unit/test_features/test_store.py::TestFeatureCache::test_get_expired_item - AssertionError: assert {'feature_1': 1.0, 'feature_2': 2.0} is None
FAILED tests/unit/test_features/test_store.py::TestFeatureCache::test_get_stats - assert 0 == 1
FAILED tests/unit/test_features/test_store.py::TestFeatureCache::test_feature_type_order_independence - AssertionError: assert '930d6c874708...2f13c90934bc6' == 'f0bbfeccbd19...45689abb7d4f0'
  - f0bbfeccbd19d47e08545689abb7d4f0
  + 930d6c874708727f0ea2f13c90934bc6
FAILED tests/unit/test_features/test_store.py::TestFeatureStore::test_get_data_for_features_with_db - assert 0 == 2
 +  where 0 = len([])
FAILED tests/unit/test_features/test_store.py::TestFeatureStore::test_error_propagation - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_extract_features_with_sample_data - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_extract_features_empty_events - assert 0.0 == -0.7071067811865471
 +  where -0.7071067811865471 = <built-in function sin>((((2 * 3.141592653589793) * 15) / 24))
 +    where <built-in function sin> = math.sin
 +    and   3.141592653589793 = math.pi
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_extract_features_single_event - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_feature_consistency - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_batch_feature_extraction - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_edge_case_time_boundaries - assert -0.8660254037844384 == 0.0
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_large_event_sequences - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_error_handling - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_memory_efficiency - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_feature_value_ranges - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractor::test_concurrent_extraction - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorEdgeCases::test_events_in_future - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorEdgeCases::test_duplicate_timestamps - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorEdgeCases::test_extreme_time_differences - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorEdgeCases::test_rapid_state_changes - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorEdgeCases::test_missing_sensor_types - TypeError: FeatureExtractionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHAEvent::test_ha_event_is_valid_false_missing_entity_id - AssertionError: assert '' is False
 +  where '' = <bound method HAEvent.is_valid of HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 17, 9, 13, 2, 780185), attributes={}, event_type='state_changed')>()
 +    where <bound method HAEvent.is_valid of HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 17, 9, 13, 2, 780185), attributes={}, event_type='state_changed')> = HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 17, 9, 13, 2, 780185), attributes={}, event_type='state_changed').is_valid
FAILED tests/unit/test_ingestion/test_ha_client.py::TestRateLimiter::test_rate_limiter_init - AttributeError: 'RateLimiter' object has no attribute 'window_seconds'
FAILED tests/unit/test_ingestion/test_ha_client.py::TestRateLimiter::test_rate_limiter_acquire_at_limit - TypeError: RateLimitExceededError.__init__() got an unexpected keyword argument 'resource'
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_ha_client_init - AttributeError: 'RateLimiter' object has no attribute 'window_seconds'
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_test_authentication_success - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_test_authentication_401 - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_test_authentication_other_error - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_test_authentication_connection_error - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_connect_websocket_success - TypeError: object AsyncMock can't be used in 'await' expression
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_state_success - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_state_not_found - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_state_api_error - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_history_success - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_history_default_end_time - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_history_not_found - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_convert_ha_event_to_sensor_event - AssertionError: assert 'off' == 'of'
  - of
  + off
  ?   +
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientWebSocketHandling::test_handle_event_processing - assert 0 == 1
 +  where 0 = len([])
FAILED tests/unit/test_models/test_base_predictors.py::TestBasePredictor::test_prediction_history_management - AttributeError: type object 'ModelType' has no attribute 'GP'
FAILED tests/unit/test_models/test_base_predictors.py::TestLSTMPredictor::test_lstm_initialization - AssertionError: assert 'lstm_units' in {'alpha': 0.0001, 'dropout': 0.2, 'early_stopping': True, 'hidden_layers': [64, 32], ...}
 +  where {'alpha': 0.0001, 'dropout': 0.2, 'early_stopping': True, 'hidden_layers': [64, 32], ...} = LSTMPredictor(model_type=lstm, room_id=test_room, is_trained=False, version=v1.0).model_params
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestPerformanceAndConcurrency::test_concurrent_prediction_recording - AssertionError: assert 0 >= 10
 +  where 0 = <src.adaptation.tracking_manager.TrackingManager object at 0x7f06c1611f40>._total_predictions_recorded
 +  and   10 = len([PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 43, 4, 639252), transition_type='occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_0', 'prediction_id': 'pred_0'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 44, 4, 639274), transition_type='occupied', confidence_score=0.81, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_1', 'prediction_id': 'pred_1'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 45, 4, 639282), transition_type='occupied', confidence_score=0.8200000000000001, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_2', 'prediction_id': 'pred_2'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 46, 4, 639307), transition_type='occupied', confidence_score=0.8300000000000001, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_0', 'prediction_id': 'pred_3'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 47, 4, 639313), transition_type='occupied', confidence_score=0.8400000000000001, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_1', 'prediction_id': 'pred_4'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 17, 9, 48, 4, 639319), transition_type='occupied', confidence_score=0.8500000000000001, prediction_interval=None, alternatives=None, model_type=None, model_version=None, features_used=None, prediction_metadata={'room_id': 'room_2', 'prediction_id': 'pred_5'}), ...])
FAILED tests/unit/test_models/test_base_predictors.py::TestLSTMPredictor::test_lstm_training_convergence - src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [150, 200]
FAILED tests/unit/test_models/test_base_predictors.py::TestLSTMPredictor::test_lstm_prediction_format - src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [150, 10]
FAILED tests/unit/test_models/test_base_predictors.py::TestLSTMPredictor::test_lstm_confidence_calibration - src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [150, 200]
FAILED tests/unit/test_models/test_base_predictors.py::TestXGBoostPredictor::test_xgboost_initialization - AssertionError: assert 'objective' in {'colsample_bytree': 0.8, 'early_stopping_rounds': 20, 'eval_metric': 'rmse', 'learning_rate': 0.1, ...}
 +  where {'colsample_bytree': 0.8, 'early_stopping_rounds': 20, 'eval_metric': 'rmse', 'learning_rate': 0.1, ...} = XGBoostPredictor(model_type=xgboost, room_id=living_room, is_trained=False, version=v1.0).model_params
FAILED tests/unit/test_adaptation/test_tracking_manager.py::TestPerformanceAndConcurrency::test_background_task_management - assert 0 > 0
FAILED tests/unit/test_models/test_base_predictors.py::TestXGBoostPredictor::test_xgboost_incremental_update - AttributeError: 'XGBoostPredictor' object has no attribute 'incremental_update'
FAILED tests/unit/test_models/test_base_predictors.py::TestHMMPredictor::test_hmm_initialization - AssertionError: assert 'n_iter' in {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 4, ...}
 +  where {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 4, ...} = HMMPredictor(model_type=hmm, room_id=bedroom, is_trained=False, version=v1.0).model_params
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidatorInitialization::test_validator_initialization - AttributeError: 'PredictionValidator' object has no attribute 'accuracy_threshold_minutes'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidatorInitialization::test_validator_custom_configuration - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'max_pending_predictions'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionRecording::test_basic_prediction_recording - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c132d880> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionRecording::test_prediction_recording_with_metadata - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c132ca10> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionRecording::test_duplicate_prediction_handling - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12e8560> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionRecording::test_prediction_expiration_handling - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12eb290> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidation::test_successful_prediction_validation - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12e9ac0> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidation::test_prediction_validation_multiple_candidates - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12e8140> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidation::test_validation_with_no_pending_predictions - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12e9280> does not have the attribute '_update_validation_in_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestPredictionValidation::test_validation_time_window_enforcement - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c132f6b0> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestAccuracyMetricsRetrieval::test_time_filtered_accuracy_metrics - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c12ebc50> does not have the attribute '_get_predictions_from_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestValidationStatistics::test_validation_stats_collection - TypeError: object dict can't be used in 'await' expression
FAILED tests/unit/test_models/test_base_predictors.py::TestGaussianProcessPredictor::test_gp_initialization - AttributeError: type object 'ModelType' has no attribute 'GP'
FAILED tests/unit/test_adaptation/test_validator.py::TestValidationStatistics::test_validation_performance_metrics - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c163d760> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestValidationStatistics::test_total_predictions_counter - AttributeError: 'PredictionValidator' object has no attribute 'get_total_predictions'
FAILED tests/unit/test_adaptation/test_validator.py::TestValidationStatistics::test_validation_rate_calculation - AttributeError: 'PredictionValidator' object has no attribute 'get_validation_rate'. Did you mean: 'get_validation_stats'?
FAILED tests/unit/test_adaptation/test_validator.py::TestDatabaseIntegration::test_prediction_storage_to_database - AttributeError: 'PredictionValidator' object has no attribute '_store_prediction_to_db'. Did you mean: '_store_prediction_in_db'?
FAILED tests/unit/test_adaptation/test_validator.py::TestDatabaseIntegration::test_validation_update_in_database - AttributeError: 'PredictionValidator' object has no attribute '_update_validation_in_db'. Did you mean: '_update_predictions_in_db'?
FAILED tests/unit/test_adaptation/test_validator.py::TestDatabaseIntegration::test_predictions_retrieval_from_database - AttributeError: 'PredictionValidator' object has no attribute '_get_predictions_from_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestCleanupAndMaintenance::test_expired_predictions_cleanup - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c163cb60> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestCleanupAndMaintenance::test_validation_history_cleanup - AttributeError: 'PredictionValidator' object has no attribute '_validation_history'. Did you mean: '_validation_records'?
FAILED tests/unit/test_adaptation/test_validator.py::TestCleanupAndMaintenance::test_pending_predictions_size_limit - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c1628b00> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_adaptation/test_validator.py::TestErrorHandlingAndEdgeCases::test_invalid_prediction_data_handling - assert False
 +  where False = isinstance(TypeError("PredictionValidator.record_prediction() got an unexpected keyword argument 'predicted_time'"), (<class 'ValueError'>, <class 'src.core.exceptions.OccupancyPredictionError'>))
FAILED tests/unit/test_adaptation/test_validator.py::TestErrorHandlingAndEdgeCases::test_validation_with_invalid_actual_time - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c16aa300> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_models/test_base_predictors.py::TestPredictorSerialization::test_model_save_load_cycle - src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [10, 200]
FAILED tests/unit/test_adaptation/test_validator.py::TestErrorHandlingAndEdgeCases::test_concurrent_validation_operations - AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7f06c16a9fa0> does not have the attribute '_store_prediction_to_db'
FAILED tests/unit/test_models/test_base_predictors.py::TestPredictorErrorHandling::test_prediction_on_untrained_model - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_base_predictors.py::TestPredictorErrorHandling::test_training_with_insufficient_data - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePerformance::test_ensemble_training_performance - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_base_predictors.py::TestPredictorErrorHandling::test_invalid_feature_data - Failed: DID NOT RAISE <class 'src.core.exceptions.ModelPredictionError'>
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePerformance::test_ensemble_prediction_latency - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_model_serialization.py::TestBasicModelSerialization::test_save_load_trained_xgboost_model - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_model_serialization.py::TestEnsembleModelSerialization::test_ensemble_base_model_serialization - AssertionError: assert 'placeholder' == 'test_room'
  - test_room
  + placeholder
FAILED tests/unit/test_models/test_model_serialization.py::TestSerializationErrorHandling::test_partial_model_data_loading - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
FAILED tests/unit/test_models/test_model_serialization.py::TestMultipleModelSerialization::test_model_comparison_after_serialization - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_model_serialization.py::TestBackwardsCompatibility::test_version_compatibility_handling - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
FAILED tests/unit/test_models/test_ensemble.py::TestEnsembleTraining::test_ensemble_training_phases - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_training_config.py::TestTrainingConfigManager::test_profile_management - AssertionError: Regex pattern did not match.
 Regex: 'Training profile invalid_profile not available'
 Input: "'invalid_profile' is not a valid TrainingProfile"
FAILED tests/unit/test_models/test_training_config.py::TestTrainingConfigManager::test_profile_updates - AssertionError: Regex pattern did not match.
 Regex: 'Profile invalid_profile not found'
 Input: "'invalid_profile' is not a valid TrainingProfile"
FAILED tests/unit/test_models/test_training_pipeline.py::TestDataQualityValidation::test_data_quality_validation_good_data - assert True is True
 +  where True = DataQualityReport(passed=True, total_samples=721, valid_samples=721, sufficient_samples=True, data_freshness_ok=True, feature_completeness_ok=True, temporal_consistency_ok=True, missing_values_percent=0.0, duplicates_count=0, outliers_count=0, data_gaps=[], recommendations=[]).passed
FAILED tests/unit/test_models/test_training_pipeline.py::TestDataPreparationAndFeatures::test_feature_extraction - assert 0 > 0
 +  where 0 = len(Empty DataFrame\nColumns: []\nIndex: [])
FAILED tests/unit/test_models/test_training_pipeline.py::TestDataPreparationAndFeatures::test_data_splitting - assert 240 == 210
 +  where 240 = len(     temporal_hour  temporal_day_of_week  ...  contextual_temp  motion_count\n0               16                     0  ...        29.461543             1\n1                5                     5  ...        23.040160             1\n2               12                     1  ...         9.705222             3\n3                8                     1  ...        22.468294             2\n4               23                     2  ...        17.130579             4\n..             ...                   ...  ...              ...           ...\n235             22                     3  ...        19.363807             3\n236             23                     6  ...        23.282016             1\n237             14                     5  ...        18.638156             4\n238              8                     0  ...        24.747470             7\n239             13                     2  ...        19.204178             0\n\n[240 rows x 5 columns])
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelTraining::test_model_training_failure_handling - AssertionError: Regex pattern did not match.
 Regex: 'No models were successfully trained'
 Input: "Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: ModelTrainingError: Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room"
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelTraining::test_model_training_specific_type - src.core.exceptions.ModelTrainingError: Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: ModelTrainingError: Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelValidation::test_model_validation_prediction_failure - ValueError: could not convert string to float: 'in'
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelValidation::test_quality_threshold_checking - ValueError: could not convert string to float: 'in'
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelDeployment::test_model_deployment - assert 0 == 1
 +  where 0 = len([])
FAILED tests/unit/test_models/test_training_pipeline.py::TestModelDeployment::test_model_artifact_saving - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
FAILED tests/unit/test_models/test_training_pipeline.py::TestFullTrainingWorkflow::test_train_room_models_success - src.core.exceptions.ModelTrainingError: Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: ValueError: could not convert string to float: 'in'
FAILED tests/unit/test_models/test_training_pipeline.py::TestFullTrainingWorkflow::test_train_room_models_insufficient_data - AssertionError: Regex pattern did not match.
 Regex: 'Insufficient data'
 Input: "Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
FAILED tests/unit/test_models/test_ensemble.py::TestEnsembleTraining::test_ensemble_model_weight_calculation - AssertionError: assert 'lstm' == 'xgboost'
  - xgboost
  + lstm
FAILED tests/unit/test_models/test_training_pipeline.py::TestFullTrainingWorkflow::test_train_room_models_quality_failure - ValueError: If using all scalar values, you must pass an index
FAILED tests/unit/test_models/test_training_pipeline.py::TestFullTrainingWorkflow::test_initial_training_multiple_rooms - AttributeError: <module 'src.models.training_pipeline' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/models/training_pipeline.py'> does not have the attribute 'get_config'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsembleTraining::test_ensemble_training_error_handling - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_training_pipeline.py::TestTrainingPipelineErrorHandling::test_pipeline_exception_handling - AssertionError: Regex pattern did not match.
 Regex: 'Training pipeline failed'
 Input: "Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: Exception: Database connection failed"
FAILED tests/unit/test_models/test_training_pipeline.py::TestTrainingPipelineErrorHandling::test_retraining_pipeline_error_handling - AssertionError: Regex pattern did not match.
 Regex: 'Retraining pipeline failed'
 Input: "Model training failed for ensemble model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=test_room | Caused by: Exception: Retraining failed"
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePrediction::test_ensemble_prediction_generation - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePrediction::test_ensemble_confidence_with_gp_uncertainty - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePrediction::test_ensemble_prediction_combination_methods - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePrediction::test_ensemble_alternatives_generation - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsemblePrediction::test_ensemble_prediction_error_handling - TypeError: ModelPredictionError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsembleIncrementalUpdate::test_ensemble_incremental_update - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/test_models/test_ensemble.py::TestEnsembleIncrementalUpdate::test_incremental_update_error_handling - TypeError: ModelTrainingError.__init__() missing 1 required positional argument: 'room_id'
ERROR tests/unit/test_data/test_database.py::TestDatabaseManager::test_get_session_success - TypeError: Boolean value of this clause is not defined
ERROR tests/unit/test_features/test_contextual.py::TestContextualFeatureExtractorEdgeCases::test_no_room_states
ERROR tests/unit/test_adaptation/test_validator.py::TestAccuracyMetricsRetrieval::test_room_accuracy_metrics - TypeError: 'predicted_time' is an invalid keyword argument for Prediction
ERROR tests/unit/test_adaptation/test_validator.py::TestAccuracyMetricsRetrieval::test_overall_accuracy_metrics - TypeError: 'predicted_time' is an invalid keyword argument for Prediction
ERROR tests/unit/test_adaptation/test_validator.py::TestAccuracyMetricsRetrieval::test_model_specific_accuracy_metrics - TypeError: 'predicted_time' is an invalid keyword argument for Prediction
ERROR tests/unit/test_adaptation/test_validator.py::TestAccuracyMetricsRetrieval::test_accuracy_trend_analysis - TypeError: 'predicted_time' is an invalid keyword argument for Prediction
ERROR tests/unit/test_adaptation/test_validator.py::TestValidationStatistics::test_room_prediction_counts - TypeError: 'predicted_time' is an invalid keyword argument for Prediction
==== 200 failed, 546 passed, 53737 warnings, 7 errors in 101.70s (0:01:41) =====