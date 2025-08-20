==================================== ERRORS ====================================
_ ERROR at setup of TestContextualFeatureExtractorEdgeCases.test_no_room_states _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_contextual.py, line 813
      def test_no_room_states(self, extractor, target_time):
E       fixture 'target_time' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extractor, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_features/__init__.py::<event_loop>, tests/unit/test_features/test_contextual.py::<event_loop>, tests/unit/test_features/test_contextual.py::TestContextualFeat
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_contextual.py:813
__ ERROR at setup of TestAccuracyMetricsRetrieval.test_room_accuracy_metrics ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:118: in prediction_history
    prediction = Prediction(
<string>:4: in __init__
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:571: in _initialize_instance
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:569: in _initialize_instance
    manager.original_init(*mixed[1:], **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/decl_base.py:2179: in _declarative_constructor
    raise TypeError(
E   TypeError: 'predicted_time' is an invalid keyword argument for Prediction
_ ERROR at setup of TestAccuracyMetricsRetrieval.test_overall_accuracy_metrics _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:118: in prediction_history
    prediction = Prediction(
<string>:4: in __init__
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:571: in _initialize_instance
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:569: in _initialize_instance
    manager.original_init(*mixed[1:], **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/decl_base.py:2179: in _declarative_constructor
    raise TypeError(
E   TypeError: 'predicted_time' is an invalid keyword argument for Prediction
_ ERROR at setup of TestAccuracyMetricsRetrieval.test_model_specific_accuracy_metrics _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:118: in prediction_history
    prediction = Prediction(
<string>:4: in __init__
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:571: in _initialize_instance
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:569: in _initialize_instance
    manager.original_init(*mixed[1:], **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/decl_base.py:2179: in _declarative_constructor
    raise TypeError(
E   TypeError: 'predicted_time' is an invalid keyword argument for Prediction
_ ERROR at setup of TestAccuracyMetricsRetrieval.test_accuracy_trend_analysis __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:118: in prediction_history
    prediction = Prediction(
<string>:4: in __init__
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:571: in _initialize_instance
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:569: in _initialize_instance
    manager.original_init(*mixed[1:], **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/decl_base.py:2179: in _declarative_constructor
    raise TypeError(
E   TypeError: 'predicted_time' is an invalid keyword argument for Prediction
____ ERROR at setup of TestValidationStatistics.test_room_prediction_counts ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:118: in prediction_history
    prediction = Prediction(
<string>:4: in __init__
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:571: in _initialize_instance
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state.py:569: in _initialize_instance
    manager.original_init(*mixed[1:], **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/decl_base.py:2179: in _declarative_constructor
    raise TypeError(
E   TypeError: 'predicted_time' is an invalid keyword argument for Prediction
=================================== FAILURES ===================================
____________ TestOptimizationStrategies.test_bayesian_optimization _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_optimizer.py:293: in test_bayesian_optimization
    assert result.total_evaluations > 0
E   AssertionError: assert 0 > 0
E    +  where 0 = OptimizationResult(success=True, optimization_time_seconds=0.000215, best_parameters={}, best_score=0.7, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=True, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No optimization dimensions available').total_evaluations
------------------------------ Captured log call -------------------------------
WARNING  src.adaptation.optimizer:optimizer.py:636 No valid dimensions for optimization - using default parameters
__________ TestOptimizationHistory.test_optimization_history_tracking __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_optimizer.py:603: in test_optimization_history_tracking
    assert model_key in optimizer._optimization_history
E   AssertionError: assert 'test_room_test_model' in {'test_model': [OptimizationResult(success=True, optimization_time_seconds=0.000149, best_parameters={}, best_score=0.7, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=True, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No optimization dimensions available')]}
E    +  where {'test_model': [OptimizationResult(success=True, optimization_time_seconds=0.000149, best_parameters={}, best_score=0.7, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=True, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No optimization dimensions available')]} = <src.adaptation.optimizer.ModelOptimizer object at 0x7fb19f458380>._optimization_history
------------------------------ Captured log call -------------------------------
WARNING  src.adaptation.optimizer:optimizer.py:636 No valid dimensions for optimization - using default parameters
_____________ TestErrorHandling.test_model_training_error_handling _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_optimizer.py:845: in test_model_training_error_handling
    assert not result.success
E   AssertionError: assert not True
E    +  where True = OptimizationResult(success=True, optimization_time_seconds=0.000202, best_parameters={}, best_score=0.7, improvement_over_default=0.0, total_evaluations=0, convergence_achieved=True, optimization_history=[], validation_score=None, training_score=None, cross_validation_scores=None, model_complexity=None, prediction_latency_ms=None, memory_usage_mb=None, error_message='No optimization dimensions available').success
------------------------------ Captured log call -------------------------------
WARNING  src.adaptation.optimizer:optimizer.py:636 No valid dimensions for optimization - using default parameters
________ TestRetrainingNeedEvaluation.test_cooldown_period_enforcement _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_retrainer.py:491: in test_cooldown_period_enforcement
    assert request is None
E   AssertionError: assert RetrainingRequest(request_id='bathroom_lstm_1755552185', room_id='bathroom', model_type=<ModelType.LSTM: 'lstm'>, trigger=<RetrainingTrigger.ACCURACY_DEGRADATION: 'accuracy_degradation'>, strategy=<RetrainingStrategy.FULL_RETRAIN: 'full_retrain'>, priority=6.0, created_time=datetime.datetime(2025, 8, 18, 21, 23, 5, 370206), accuracy_metrics=AccuracyMetrics(total_predictions=100, validated_predictions=85, accurate_predictions=45, expired_predictions=0, failed_predictions=0, accuracy_rate=52.9, mean_error_minutes=28.5, median_error_minutes=25.0, std_error_minutes=0.0, rmse_minutes=0.0, mae_minutes=0.0, error_percentiles={}, accuracy_by_level={}, mean_bias_minutes=0.0, bias_std_minutes=0.0, mean_confidence=0.0, confidence_accuracy_correlation=0.68, overconfidence_rate=0.0, underconfidence_rate=0.0, measurement_period_start=datetime.datetime(2025, 8, 17, 21, 23, 5, 369129), measurement_period_end=datetime.datetime(2025, 8, 18, 21, 23, 5, 369137), predictions_per_hour=0.0), drift_metrics
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.retrainer:retrainer.py:917 Error checking cooldown for bathroom_lstm: can't compare offset-naive and offset-aware datetimes
_____ TestErrorHandlingAndEdgeCases.test_memory_usage_with_large_datasets ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_validator.py:1280: in test_memory_usage_with_large_datasets
    with patch.object(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <src.adaptation.validator.PredictionValidator object at 0x7fb197da5700> does not have the attribute '_get_predictions_from_db'
___________________ TestSensorState.test_sensor_state_values ___________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants.py:71: in test_sensor_state_values
    assert SensorState.OFF.value == "of"
E   AssertionError: assert 'off' == 'of'
E     - of
E     + off
E     ?   +
_________________ TestSensorState.test_sensor_state_membership _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants.py:85: in test_sensor_state_membership
    assert "of" in states
E   AssertionError: assert 'of' in ['on', 'off', 'open', 'closed', 'unknown', 'unavailable']
____________________ TestStateConstants.test_absence_states ____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants.py:149: in test_absence_states
    assert ABSENCE_STATES == ["of"]
E   AssertionError: assert ['off'] == ['of']
E     At index 0 diff: 'off' != 'of'
E     Full diff:
E     - ['of']
E     + ['off']
E     ?     +
_________________ TestDatabaseManager.test_initialize_success __________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:101: in test_initialize_success
    assert manager.is_initialized
E   assert False
E    +  where False = <src.data.storage.database.DatabaseManager object at 0x7fb197e717c0>.is_initialized
______________ TestDatabaseManager.test_verify_connection_success ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:258: in test_verify_connection_success
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:662: in __getattr__
    raise AttributeError(name)
E   AttributeError: __aenter__
________________ TestDatabaseManager.test_execute_query_success ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/data/storage/database.py:337: in execute_query
    return await asyncio.wait_for(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/tasks.py:520: in wait_for
    return await fut
src/data/storage/database.py:327: in _execute_query
    async with self.get_session() as session:
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
During handling of the above exception, another exception occurred:
tests/unit/test_data/test_database.py:373: in test_execute_query_success
    result = await test_db_manager.execute_query(
src/data/storage/database.py:363: in execute_query
    raise DatabaseQueryError(
E   src.core.exceptions.DatabaseQueryError: Database query failed: SELECT 1... | Error Code: DB_QUERY_ERROR | Context: query=SELECT 1, parameters={'param': 'value'} | Caused by: TypeError: 'coroutine' object does not support the asynchronous context manager protocol
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:362 Unexpected database error: 'coroutine' object does not support the asynchronous context manager protocol
________________ TestDatabaseManager.test_health_check_healthy _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:416: in test_health_check_healthy
    assert health["status"] == "healthy"
E   AssertionError: assert 'unhealthy' == 'healthy'
E     - healthy
E     + unhealthy
E     ? ++
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:735 Database health check failed: 'coroutine' object does not support the asynchronous context manager protocol
_________ TestDatabaseManager.test_health_check_timescaledb_available __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:464: in test_health_check_timescaledb_available
    assert health["timescale_status"] == "available"
E   AssertionError: assert None == 'available'
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:735 Database health check failed: 'coroutine' object does not support the asynchronous context manager protocol
________ TestDatabaseManager.test_health_check_timescaledb_unavailable _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:501: in test_health_check_timescaledb_unavailable
    assert health["timescale_status"] == "unavailable"
E   AssertionError: assert None == 'unavailable'
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:735 Database health check failed: 'coroutine' object does not support the asynchronous context manager protocol
______ TestDatabaseManager.test_health_check_timescaledb_version_parsing _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:526: in test_health_check_timescaledb_version_parsing
    with self.subTest(version_string=version_string):
E   AttributeError: 'TestDatabaseManager' object has no attribute 'subTest'
___ TestDatabaseManager.test_health_check_timescaledb_version_parsing_error ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:591: in test_health_check_timescaledb_version_parsing_error
    assert health["timescale_status"] == "available"
E   AssertionError: assert None == 'available'
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:735 Database health check failed: 'coroutine' object does not support the asynchronous context manager protocol
____________________ TestDatabaseManager.test_close_cleanup ____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:637: in test_close_cleanup
    await manager.close()
src/data/storage/database.py:754: in close
    await self._cleanup()
src/data/storage/database.py:762: in _cleanup
    await self._health_check_task
E   TypeError: object Mock can't be used in 'await' expression
_______________ TestGlobalDatabaseFunctions.test_get_db_session ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:725: in test_get_db_session
    async with get_db_session() as session:
E   TypeError: 'async_generator' object does not support the asynchronous context manager protocol
___ TestDatabaseManagerEdgeCases.test_verify_connection_timescaledb_warning ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:952: in test_verify_connection_timescaledb_warning
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:662: in __getattr__
    raise AttributeError(name)
E   AttributeError: __aenter__
_______ TestDatabaseManagerEdgeCases.test_get_session_rollback_on_error ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'close' to have been called once. Called 2 times.
E   Calls: [call(), call()].
During handling of the above exception, another exception occurred:
tests/unit/test_data/test_database.py:977: in test_get_session_rollback_on_error
    mock_session.close.assert_called_once()
E   AssertionError: Expected 'close' to have been called once. Called 2 times.
E   Calls: [call(), call()].
_____ TestDatabaseManagerEdgeCases.test_health_check_with_previous_errors ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_database.py:999: in test_health_check_with_previous_errors
    assert health["status"] == "healthy"
E   AssertionError: assert 'unhealthy' == 'healthy'
E     - healthy
E     + unhealthy
E     ? ++
------------------------------ Captured log call -------------------------------
ERROR    src.data.storage.database:database.py:735 Database health check failed: 'coroutine' object does not support the asynchronous context manager protocol
______________ TestDatabaseManagerIntegration.test_full_lifecycle ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/data/storage/database.py:87: in initialize
    await self._verify_connection()
src/data/storage/database.py:213: in _verify_connection
    async with self.engine.begin() as conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:210: in __aenter__
    return await anext(self.gen)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:1066: in begin
    async with conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/base.py:121: in __aenter__
    return await self.start(is_ctxmanager=True)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:274: in start
    await greenlet_spawn(self.sync_engine.connect)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3277: in connect
    return self._connection_cls(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:143: in __init__
    self._dbapi_connection = engine.raw_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3301: in raw_connection
    return self.pool.connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:447: in connect
    return _ConnectionFairy._checkout(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:1264: in _checkout
    fairy = _ConnectionRecord.checkout(pool)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:711: in checkout
    rec = pool._do_get()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:177: in _do_get
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:175: in _do_get
    return self._create_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:388: in _create_connection
    return _ConnectionRecord(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:673: in __init__
    self.__connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:899: in __connect
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:895: in __connect
    self.dbapi_connection = connection = pool._invoke_creator(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/create.py:661: in connect
    return dialect.connect(*cargs, **cparams)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:629: in connect
    return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py:964: in connect
    await_only(creator_fn(*arg, **kw)),
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connection.py:2421: in connect
    return await connect_utils._connect(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1075: in _connect
    raise last_error or exceptions.TargetServerAttributeNotMatched(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1049: in _connect
    conn = await _connect_addr(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:886: in _connect_addr
    return await __connect_addr(params, True, *args)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:931: in __connect_addr
    tr, pr = await connector
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:802: in _create_ssl_connection
    tr, pr = await loop.create_connection(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:1140: in create_connection
    raise OSError('Multiple exceptions: {}'.format(
E   OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
During handling of the above exception, another exception occurred:
tests/unit/test_data/test_database.py:1035: in test_full_lifecycle
    await manager.initialize()
src/data/storage/database.py:102: in initialize
    raise DatabaseConnectionError(
E   src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
___________ TestDatabaseManagerIntegration.test_concurrent_sessions ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/data/storage/database.py:87: in initialize
    await self._verify_connection()
src/data/storage/database.py:213: in _verify_connection
    async with self.engine.begin() as conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:210: in __aenter__
    return await anext(self.gen)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:1066: in begin
    async with conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/base.py:121: in __aenter__
    return await self.start(is_ctxmanager=True)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:274: in start
    await greenlet_spawn(self.sync_engine.connect)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3277: in connect
    return self._connection_cls(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:143: in __init__
    self._dbapi_connection = engine.raw_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3301: in raw_connection
    return self.pool.connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:447: in connect
    return _ConnectionFairy._checkout(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:1264: in _checkout
    fairy = _ConnectionRecord.checkout(pool)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:711: in checkout
    rec = pool._do_get()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:177: in _do_get
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:175: in _do_get
    return self._create_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:388: in _create_connection
    return _ConnectionRecord(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:673: in __init__
    self.__connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:899: in __connect
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:895: in __connect
    self.dbapi_connection = connection = pool._invoke_creator(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/create.py:661: in connect
    return dialect.connect(*cargs, **cparams)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:629: in connect
    return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py:964: in connect
    await_only(creator_fn(*arg, **kw)),
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connection.py:2421: in connect
    return await connect_utils._connect(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1075: in _connect
    raise last_error or exceptions.TargetServerAttributeNotMatched(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1049: in _connect
    conn = await _connect_addr(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:886: in _connect_addr
    return await __connect_addr(params, True, *args)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:931: in __connect_addr
    tr, pr = await connector
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:802: in _create_ssl_connection
    tr, pr = await loop.create_connection(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:1140: in create_connection
    raise OSError('Multiple exceptions: {}'.format(
E   OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
During handling of the above exception, another exception occurred:
tests/unit/test_data/test_database.py:1067: in test_concurrent_sessions
    await manager.initialize()
src/data/storage/database.py:102: in initialize
    raise DatabaseConnectionError(
E   src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
_____ TestDatabaseManagerIntegration.test_retry_mechanism_with_real_errors _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/data/storage/database.py:87: in initialize
    await self._verify_connection()
src/data/storage/database.py:213: in _verify_connection
    async with self.engine.begin() as conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:210: in __aenter__
    return await anext(self.gen)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:1066: in begin
    async with conn:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/base.py:121: in __aenter__
    return await self.start(is_ctxmanager=True)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:274: in start
    await greenlet_spawn(self.sync_engine.connect)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3277: in connect
    return self._connection_cls(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:143: in __init__
    self._dbapi_connection = engine.raw_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:3301: in raw_connection
    return self.pool.connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:447: in connect
    return _ConnectionFairy._checkout(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:1264: in _checkout
    fairy = _ConnectionRecord.checkout(pool)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:711: in checkout
    rec = pool._do_get()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:177: in _do_get
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/impl.py:175: in _do_get
    return self._create_connection()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:388: in _create_connection
    return _ConnectionRecord(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:673: in __init__
    self.__connect()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:899: in __connect
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/pool/base.py:895: in __connect
    self.dbapi_connection = connection = pool._invoke_creator(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/create.py:661: in connect
    return dialect.connect(*cargs, **cparams)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:629: in connect
    return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py:964: in connect
    await_only(creator_fn(*arg, **kw)),
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connection.py:2421: in connect
    return await connect_utils._connect(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1075: in _connect
    raise last_error or exceptions.TargetServerAttributeNotMatched(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:1049: in _connect
    conn = await _connect_addr(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:886: in _connect_addr
    return await __connect_addr(params, True, *args)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:931: in __connect_addr
    tr, pr = await connector
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/asyncpg/connect_utils.py:802: in _create_ssl_connection
    tr, pr = await loop.create_connection(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:1140: in create_connection
    raise OSError('Multiple exceptions: {}'.format(
E   OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
During handling of the above exception, another exception occurred:
tests/unit/test_data/test_database.py:1094: in test_retry_mechanism_with_real_errors
    await manager.initialize()
src/data/storage/database.py:102: in initialize
    raise DatabaseConnectionError(
E   src.core.exceptions.DatabaseConnectionError: Failed to connect to database: postgresql+asyncpg://localhost/testdb | Error Code: DB_CONNECTION_ERROR | Context: connection_string=postgresql+asyncpg://localhost/testdb | Caused by: OSError: Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5432)
____________________ TestSensorEvent.test_get_recent_events ____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1709: in _object_value_for_elem
    return self._object_lookup[elem]  # type: ignore[return-value]
E   KeyError: 'off'
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models.py:87: in test_get_recent_events
    recent_events = await SensorEvent.get_recent_events(
src/data/storage/models.py:204: in get_recent_events
    result = await session.execute(query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:309: in orm_execute_statement
    return cls.orm_setup_cursor_result(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:616: in orm_setup_cursor_result
    return loading.instances(result, querycontext)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/loading.py:262: in instances
    _prebuffered = list(chunks(None))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/loading.py:220: in chunks
    fetch = cursor._raw_all_rows()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/result.py:541: in _raw_all_rows
    return [make_row(row) for row in rows]
lib/sqlalchemy/cyextension/resultproxy.pyx:22: in sqlalchemy.cyextension.resultproxy.BaseRow.__init__
    ???
lib/sqlalchemy/cyextension/resultproxy.pyx:79: in sqlalchemy.cyextension.resultproxy._apply_processors
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1829: in process
    value = self._object_value_for_elem(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1711: in _object_value_for_elem
    raise LookupError(
E   LookupError: 'off' is not among the defined enum values. Enum name: sensor_state_enum. Possible values: on, of, open, ..., unknown
__________ TestSensorEvent.test_get_recent_events_with_sensor_filter ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1709: in _object_value_for_elem
    return self._object_lookup[elem]  # type: ignore[return-value]
E   KeyError: 'off'
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models.py:106: in test_get_recent_events_with_sensor_filter
    recent_events = await SensorEvent.get_recent_events(
src/data/storage/models.py:204: in get_recent_events
    result = await session.execute(query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:309: in orm_execute_statement
    return cls.orm_setup_cursor_result(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:616: in orm_setup_cursor_result
    return loading.instances(result, querycontext)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/loading.py:262: in instances
    _prebuffered = list(chunks(None))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/loading.py:220: in chunks
    fetch = cursor._raw_all_rows()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/result.py:541: in _raw_all_rows
    return [make_row(row) for row in rows]
lib/sqlalchemy/cyextension/resultproxy.pyx:22: in sqlalchemy.cyextension.resultproxy.BaseRow.__init__
    ???
lib/sqlalchemy/cyextension/resultproxy.pyx:79: in sqlalchemy.cyextension.resultproxy._apply_processors
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1829: in process
    value = self._object_value_for_elem(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/sqltypes.py:1711: in _object_value_for_elem
    raise LookupError(
E   LookupError: 'off' is not among the defined enum values. Enum name: sensor_state_enum. Possible values: on, of, open, ..., unknown
__________ TestContextualFeatureExtractor.test_natural_light_patterns __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_contextual.py:774: in test_natural_light_patterns
    assert features["natural_light_available"] == expected_light
E   assert 1.0 == 0.0
______ TestFeatureEngineeringEngine.test_error_handling_extractor_failure ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_engineering.py:332: in test_error_handling_extractor_failure
    with pytest.raises(FeatureExtractionError):
E   Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
------------------------------ Captured log call -------------------------------
ERROR    src.features.engineering:engineering.py:328 Failed to extract temporal features: Temporal extraction failed
______ TestFeatureEngineeringEngine.test_validate_configuration_no_config ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_engineering.py:473: in test_validate_configuration_no_config
    assert validation_results["valid"] is False
E   assert True is False
------------------------------ Captured log call -------------------------------
WARNING  src.features.engineering:engineering.py:690 No room configurations available - feature extraction may be limited
_________ TestFeatureEngineeringEngine.test_large_feature_set_handling _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_engineering.py:760: in test_large_feature_set_handling
    assert len(features) == total_expected
E   AssertionError: assert 231 == 230
E    +  where 231 = len({'contextual_contextual_feature_0': 200.0, 'contextual_contextual_feature_1': 201.0, 'contextual_contextual_feature_10': 210.0, 'contextual_contextual_feature_11': 211.0, ...})
__________ TestDriftDetectionIntegration.test_manual_drift_detection ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracking_manager.py:462: in test_manual_drift_detection
    assert result is not None
E   assert None is not None
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracking_manager:tracking_manager.py:849 Failed to check drift for living_room: object DriftMetrics can't be used in 'await' expression
_______ TestSequentialFeatureExtractor.test_extract_features_multi_room ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/sequential.py:111: in extract_features
    self._extract_movement_classification_features(
src/features/sequential.py:543: in _extract_movement_classification_features
    classification = self.classifier.classify_movement(
src/data/ingestion/event_processor.py:218: in classify_movement
    metrics = self._calculate_movement_metrics(sequence, room_config)
src/data/ingestion/event_processor.py:278: in _calculate_movement_metrics
    metrics["spatial_dispersion"] = self._calculate_spatial_dispersion(
src/data/ingestion/event_processor.py:449: in _calculate_spatial_dispersion
    for sensor_type, sensors in room_config.sensors.items():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'sensors'
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_sequential.py:155: in test_extract_features_multi_room
    features = extractor.extract_features(
src/features/sequential.py:125: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for sequential features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: AttributeError: Mock object has no attribute 'sensors'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:122 Failed to extract sequential features: Mock object has no attribute 'sensors'
_______ TestSequentialFeatureExtractor.test_extract_features_single_room _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/sequential.py:111: in extract_features
    self._extract_movement_classification_features(
src/features/sequential.py:543: in _extract_movement_classification_features
    classification = self.classifier.classify_movement(
src/data/ingestion/event_processor.py:218: in classify_movement
    metrics = self._calculate_movement_metrics(sequence, room_config)
src/data/ingestion/event_processor.py:278: in _calculate_movement_metrics
    metrics["spatial_dispersion"] = self._calculate_spatial_dispersion(
src/data/ingestion/event_processor.py:449: in _calculate_spatial_dispersion
    for sensor_type, sensors in room_config.sensors.items():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'sensors'
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_sequential.py:183: in test_extract_features_single_room
    features = extractor.extract_features(
src/features/sequential.py:125: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for sequential features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: AttributeError: Mock object has no attribute 'sensors'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:122 Failed to extract sequential features: Mock object has no attribute 'sensors'
____________ TestSequentialFeatureExtractor.test_empty_room_configs ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_sequential.py:426: in test_empty_room_configs
    assert "human_movement_probability" in features
E   AssertionError: assert 'human_movement_probability' in {'active_room_count': 2, 'avg_event_interval': 106.66666666666667, 'avg_room_dwell_time': 320.0, 'burst_ratio': 0.0, ...}
_________ TestSequentialFeatureExtractor.test_no_classifier_available __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_sequential.py:438: in test_no_classifier_available
    assert features["human_movement_probability"] == 0.5
E   KeyError: 'human_movement_probability'
____ TestSequentialFeatureExtractorMovementPatterns.test_cat_like_patterns _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_sequential.py:671: in test_cat_like_patterns
    assert features["room_revisit_ratio"] > 0.5  # Frequent returns to same room
E   assert 0.4 > 0.5
____________________ TestFeatureCache.test_get_expired_item ____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_store.py:188: in test_get_expired_item
    assert retrieved_features is None
E   AssertionError: assert {'feature_1': 1.0, 'feature_2': 2.0} is None
_______________________ TestFeatureCache.test_get_stats ________________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_store.py:314: in test_get_stats
    assert stats["hit_count"] == 1
E   assert 0 == 1
____________ TestFeatureCache.test_feature_type_order_independence _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_store.py:327: in test_feature_type_order_independence
    assert key1 == key2
E   AssertionError: assert '6c2226f0867d...439d7f7750289' == '75314a3e32a4...816ae4401a52f'
E     - 75314a3e32a4357884b816ae4401a52f
E     + 6c2226f0867d62d6048439d7f7750289
_____________ TestFeatureStore.test_get_data_for_features_with_db ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_store.py:629: in test_get_data_for_features_with_db
    assert len(events) == 2
E   assert 0 == 2
E    +  where 0 = len([])
------------------------------ Captured log call -------------------------------
ERROR    src.features.store:store.py:520 Failed to get data for features: 'coroutine' object does not support the asynchronous context manager protocol
_____ TestTemporalFeatureExtractor.test_extract_features_with_sample_data ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:98: in test_extract_features_with_sample_data
    features = extractor.extract_features(sample_events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
_______ TestTemporalFeatureExtractor.test_extract_features_empty_events ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:132: in test_extract_features_empty_events
    assert features["hour_sin"] == math.sin(2 * math.pi * 15 / 24)  # 3 PM
E   assert 0.0 == -0.7071067811865471
E    +  where -0.7071067811865471 = <built-in function sin>((((2 * 3.141592653589793) * 15) / 24))
E    +    where <built-in function sin> = math.sin
E    +    and   3.141592653589793 = math.pi
_______ TestTemporalFeatureExtractor.test_extract_features_single_event ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:137: in test_extract_features_single_event
    features = extractor.extract_features(single_event, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
_____________ TestTemporalFeatureExtractor.test_time_calculations ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:163: in test_time_calculations
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400804716304'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400804716304'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
___________ TestTemporalFeatureExtractor.test_cyclical_time_features ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:188: in test_cyclical_time_features
    assert abs(features["hour_sin"] - expected_sin) < 0.001
E   assert 1.0 < 0.001
E    +  where 1.0 = abs((0.0 - 1.0))
____________ TestTemporalFeatureExtractor.test_day_of_week_features ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:207: in test_day_of_week_features
    assert abs(features["day_sin"] - expected_sin) < 0.001
E   assert 0.7818314824680298 < 0.001
E    +  where 0.7818314824680298 = abs((0.0 - 0.7818314824680298))
____________ TestTemporalFeatureExtractor.test_work_hours_detection ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:225: in test_work_hours_detection
    assert features["is_work_hours"] == (1.0 if is_work_hours else 0.0)
E   assert 0.0 == 1.0
________ TestTemporalFeatureExtractor.test_state_duration_calculations _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:249: in test_state_duration_calculations
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400804717936'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400804717936'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
_____________ TestTemporalFeatureExtractor.test_activity_patterns ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:273: in test_activity_patterns
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400805169728'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400805169728'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
____________ TestTemporalFeatureExtractor.test_sensor_type_features ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:296: in test_sensor_type_features
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400804710496'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400804710496'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
__________ TestTemporalFeatureExtractor.test_recent_activity_features __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:328: in test_recent_activity_features
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400863051856'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400863051856'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
_____________ TestTemporalFeatureExtractor.test_timezone_handling ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:353: in test_timezone_handling
    assert abs(features["hour_sin"] - expected_sin) < 0.001
E   assert 0.7071067811865471 < 0.001
E    +  where 0.7071067811865471 = abs((0.0 - -0.7071067811865471))
____________ TestTemporalFeatureExtractor.test_feature_consistency _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:360: in test_feature_consistency
    features1 = extractor.extract_features(sample_events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
___________ TestTemporalFeatureExtractor.test_edge_case_no_on_events ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:383: in test_edge_case_no_on_events
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400804721872'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400804721872'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
__________ TestTemporalFeatureExtractor.test_edge_case_no_off_events ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:405: in test_edge_case_no_off_events
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400860057616'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400860057616'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
______________ TestTemporalFeatureExtractor.test_very_old_events _______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:428: in test_very_old_events
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_________ TestSystemStatusAndMetrics.test_real_time_metrics_retrieval __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracking_manager.py:621: in test_real_time_metrics_retrieval
    assert metrics is not None
E   assert None is not None
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracking_manager:tracking_manager.py:740 Failed to get real-time metrics: object Mock can't be used in 'await' expression
___________ TestSystemStatusAndMetrics.test_active_alerts_retrieval ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracking_manager.py:641: in test_active_alerts_retrieval
    assert len(alerts) == 2
E   assert 0 == 2
E    +  where 0 = len([])
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracking_manager:tracking_manager.py:760 Failed to get active alerts: object list can't be used in 'await' expression
_________ TestTemporalFeatureExtractor.test_performance_large_dataset __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:469: in test_performance_large_dataset
    features = extractor.extract_features(large_events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room '<Mock name='mock.room_id' id='140400719030336'>' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140400719030336'> | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
____________ TestTemporalFeatureExtractor.test_cache_functionality _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:491: in test_cache_functionality
    extractor.temporal_cache["test"] = "value"
E   AttributeError: 'TemporalFeatureExtractor' object has no attribute 'temporal_cache'
_______ TestTemporalFeatureExtractor.test_different_lookback_windows[1] ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:503: in test_different_lookback_windows
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_______ TestTemporalFeatureExtractor.test_different_lookback_windows[6] ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:503: in test_different_lookback_windows
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_______ TestTemporalFeatureExtractor.test_different_lookback_windows[12] _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:503: in test_different_lookback_windows
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_______ TestTemporalFeatureExtractor.test_different_lookback_windows[24] _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:503: in test_different_lookback_windows
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_______ TestTemporalFeatureExtractor.test_different_lookback_windows[48] _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:503: in test_different_lookback_windows
    features = extractor.extract_features(
E   TypeError: TemporalFeatureExtractor.extract_features() got an unexpected keyword argument 'lookback_hours'
_________ TestTemporalFeatureExtractor.test_month_and_season_features __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:563: in test_month_and_season_features
    assert abs(features["month_sin"] - expected_sin) < 0.001
E   assert 0.49999999999999994 < 0.001
E    +  where 0.49999999999999994 = abs((0.0 - 0.49999999999999994))
___________ TestTemporalFeatureExtractor.test_concurrent_extraction ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/features/temporal.py:97: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:274: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
E   TypeError: 'Mock' object is not iterable
During handling of the above exception, another exception occurred:
tests/unit/test_features/test_temporal.py:584: in test_concurrent_extraction
    results = await asyncio.gather(*tasks)
tests/unit/test_features/test_temporal.py:580: in extract_features
    return extractor.extract_features(sample_events, target_time)
src/features/temporal.py:106: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed for temporal features in room 'living_room' | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
ERROR    src.features.temporal:temporal.py:103 Failed to extract temporal features: 'Mock' object is not iterable
_______________ TestTemporalFeatureExtractor.test_stats_tracking _______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_features/test_temporal.py:596: in test_stats_tracking
    initial_stats = extractor.get_extraction_stats()
E   AttributeError: 'TemporalFeatureExtractor' object has no attribute 'get_extraction_stats'
__________ TestHAEvent.test_ha_event_is_valid_false_missing_entity_id __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_ingestion/test_ha_client.py:107: in test_ha_event_is_valid_false_missing_entity_id
    assert event.is_valid() is False
E   AssertionError: assert '' is False
E    +  where '' = <bound method HAEvent.is_valid of HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 18, 21, 23, 24, 682964, tzinfo=datetime.timezone.utc), attributes={}, event_type='state_changed')>()
E    +    where <bound method HAEvent.is_valid of HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 18, 21, 23, 24, 682964, tzinfo=datetime.timezone.utc), attributes={}, event_type='state_changed')> = HAEvent(entity_id='', state='on', previous_state='off', timestamp=datetime.datetime(2025, 8, 18, 21, 23, 24, 682964, tzinfo=datetime.timezone.utc), attributes={}, event_type='state_changed').is_valid
______ TestHomeAssistantClient.test_test_authentication_connection_error _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_ingestion/test_ha_client.py:394: in test_test_authentication_connection_error
    await client._test_authentication()
src/data/ingestion/ha_client.py:197: in _test_authentication
    async with self.session.get(api_url) as response:
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
_____________ TestSystemStatusAndMetrics.test_alert_acknowledgment _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracking_manager.py:660: in test_alert_acknowledgment
    assert success
E   assert False
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracking_manager:tracking_manager.py:774 Failed to acknowledge alert: object bool can't be used in 'await' expression
_____________ TestBasePredictor.test_prediction_history_management _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_models/test_base_predictors.py:297: in test_prediction_history_management
    assert len(predictor.prediction_history) == 500
E   AssertionError: assert 599 == 500
E    +  where 599 = len([(datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025,
E    +    where [(datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025, 8, 18, 21, 23, 26, 45046), PredictionResult(predicted_time=datetime.datetime(2025, 8, 18, 21, 53, 26, 45046), transition_type='vacant_to_occupied', confidence_score=0.8, prediction_interval=None, alternatives=None, model_type='gp', model_version=None, features_used=None, prediction_metadata=None)), (datetime.datetime(2025, 8, 18, 
_______________ TestLSTMPredictor.test_lstm_training_convergence _______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/lstm_predictor.py:174: in train
    training_score = r2_score(y_train_original, y_pred_train_original)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/utils/_param_validation.py:218: in wrapper
    return func(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:1276: in r2_score
    _check_reg_targets_with_floating_dtype(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:209: in _check_reg_targets_with_floating_dtype
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:114: in _check_reg_targets
    check_consistent_length(y_true, y_pred, sample_weight)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/utils/validation.py:473: in check_consistent_length
    raise ValueError(
E   ValueError: Found input variables with inconsistent numbers of samples: [151, 200]
During handling of the above exception, another exception occurred:
tests/unit/test_models/test_base_predictors.py:353: in test_lstm_training_convergence
    result = await predictor.train(
src/models/base/lstm_predictor.py:263: in train
    raise ModelTrainingError(model_type="lstm", room_id=self.room_id, cause=e)
E   src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [151, 200]
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.lstm_predictor:lstm_predictor.py:252 LSTM training failed: Found input variables with inconsistent numbers of samples: [151, 200]
________________ TestLSTMPredictor.test_lstm_prediction_format _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/lstm_predictor.py:174: in train
    training_score = r2_score(y_train_original, y_pred_train_original)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/utils/_param_validation.py:218: in wrapper
    return func(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:1276: in r2_score
    _check_reg_targets_with_floating_dtype(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:209: in _check_reg_targets_with_floating_dtype
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/metrics/_regression.py:114: in _check_reg_targets
    check_consistent_length(y_true, y_pred, sample_weight)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sklearn/utils/validation.py:473: in check_consistent_length
    raise ValueError(
E   ValueError: Found input variables with inconsistent numbers of samples: [151, 10]
During handling of the above exception, another exception occurred:
tests/unit/test_models/test_base_predictors.py:382: in test_lstm_prediction_format
    await predictor.train(train_features, train_targets)
src/models/base/lstm_predictor.py:263: in train
    raise ModelTrainingError(model_type="lstm", room_id=self.room_id, cause=e)
E   src.core.exceptions.ModelTrainingError: Model training failed for lstm model in room 'test_room' | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=test_room | Caused by: ValueError: Found input variables with inconsistent numbers of samples: [151, 10]