[gw0] [ 56%] FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorTimeSinceFeatures::test_extract_time_since_features_motion_sensor 

==================================== ERRORS ====================================
_____ ERROR collecting tests/unit/test_adaptation/test_tracking_manager.py _____
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracking_manager.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_adaptation/test_tracking_manager.py:21: in <module>
    from src.adaptation.retrainer import AdaptiveRetrainer, RetrainerError
E   ImportError: cannot import name 'RetrainerError' from 'src.adaptation.retrainer' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/adaptation/retrainer.py)
_ ERROR collecting tests/unit/test_core/test_config_validator_comprehensive.py _
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_core/test_config_validator_comprehensive.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_core/test_config_validator_comprehensive.py:32: in <module>
    from src.core.config_validator import (
E   ImportError: cannot import name 'ConfigValidator' from 'src.core.config_validator' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config_validator.py)
_____ ERROR collecting tests/unit/test_utils/test_metrics_comprehensive.py _____
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_utils/test_metrics_comprehensive.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_utils/test_metrics_comprehensive.py:38: in <module>
    from src.utils.metrics import (
E   ImportError: cannot import name 'MetricsCollector' from 'src.utils.metrics' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/utils/metrics.py)
______ ERROR at setup of TestAccuracyTracker.test_tracker_initialization _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
______ ERROR at setup of TestAccuracyTracker.test_record_validation_basic ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_____ ERROR at setup of TestAccuracyTracker.test_calculate_window_accuracy _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
___________ ERROR at setup of TestAccuracyTracker.test_detect_trend ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_________ ERROR at setup of TestAccuracyTracker.test_alert_generation __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_______ ERROR at setup of TestAccuracyTracker.test_get_real_time_metrics _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:567: in accuracy_tracker
    return AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
____ ERROR at setup of TestAdaptiveRetrainer.test_retrainer_initialization _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:789: in adaptive_retrainer
    return AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_accuracy_degradation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:789: in adaptive_retrainer
    return AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_insufficient_samples _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:789: in adaptive_retrainer
    return AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
_______ ERROR at setup of TestAdaptiveRetrainer.test_request_retraining ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:789: in adaptive_retrainer
    return AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
_______ ERROR at setup of TestAdaptiveRetrainer.test_perform_retraining ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:789: in adaptive_retrainer
    return AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
______ ERROR at setup of TestModelOptimizer.test_optimizer_initialization ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:910: in optimization_config
    return OptimizationConfig(
E   TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
______ ERROR at setup of TestModelOptimizer.test_optimize_hyperparameters ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:910: in optimization_config
    return OptimizationConfig(
E   TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
_ ERROR at setup of TestModelOptimizer.test_generate_hyperparameter_combinations _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:910: in optimization_config
    return OptimizationConfig(
E   TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
__ ERROR at setup of TestTrackingManager.test_tracking_manager_initialization __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_________ ERROR at setup of TestTrackingManager.test_start_monitoring __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
__________ ERROR at setup of TestTrackingManager.test_stop_monitoring __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
___ ERROR at setup of TestTrackingManager.test_record_prediction_integration ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
__ ERROR at setup of TestTrackingManager.test_validate_prediction_integration __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
____ ERROR at setup of TestTrackingManager.test_monitoring_loop_integration ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_________ ERROR at setup of TestTrackingManager.test_get_system_status _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
__________ ERROR at setup of TestTrackingManager.test_error_handling ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_________ ERROR at setup of TestTrackingManager.test_resource_cleanup __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_ ERROR at setup of TestAdaptationIntegrationScenarios.test_full_adaptation_workflow _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_ ERROR at setup of TestAdaptationIntegrationScenarios.test_concurrent_monitoring_operations _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_ ERROR at setup of TestAdaptationIntegrationScenarios.test_error_propagation __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_ ERROR at setup of TestAdaptationIntegrationScenarios.test_recovery_from_component_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:91: in tracking_config
    return TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
_ ERROR at setup of TestConceptDriftDetectorStatistical.test_detect_drift_complete_analysis _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_drift_detector_comprehensive.py:256: in mock_prediction_validator
    baseline_metrics = AccuracyMetrics(
E   TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'prediction_count'
_ ERROR at setup of TestConceptDriftDetectorStatistical.test_prediction_drift_analysis_mathematical_accuracy _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_drift_detector_comprehensive.py:256: in mock_prediction_validator
    baseline_metrics = AccuracyMetrics(
E   TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'prediction_count'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_success _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:90: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_type'. Did you mean 'predicted_time'?
___ ERROR at setup of TestIntegrationScenarios.test_full_prediction_workflow ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 535
      @pytest.mark.asyncio
      async def test_full_prediction_workflow(self, enhanced_manager, prediction_result):
          """Test complete prediction workflow with monitoring."""
          room_id = "living_room"

          # Record prediction
          prediction_id = await enhanced_manager._monitored_record_prediction(
              room_id, prediction_result, ModelType.ENSEMBLE
          )

          # Validate prediction later
          actual_time = datetime.now(timezone.utc)
          validation_result = await enhanced_manager._monitored_validate_prediction(
              room_id, actual_time
          )

          # Verify workflow
          assert prediction_id == "prediction_id_123"
          assert validation_result["accuracy_minutes"] == 5.2

          # Verify monitoring calls
          assert (
              enhanced_manager.monitoring_integration.record_prediction_accuracy.call_count
              == 2
          )
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 526
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:526
_ ERROR at setup of TestIntegrationScenarios.test_system_lifecycle_with_monitoring _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 561
      @pytest.mark.asyncio
      async def test_system_lifecycle_with_monitoring(self, enhanced_manager):
          """Test system start/stop lifecycle with monitoring."""
          # Start system
          await enhanced_manager._monitored_start_tracking()

          # Verify startup
          enhanced_manager.monitoring_integration.start_monitoring.assert_called_once()

          # Record some metrics during operation
          enhanced_manager.record_feature_computation("kitchen", "sequential", 0.032)
          enhanced_manager.record_database_operation("UPDATE", "predictions", 0.045)

          # Stop system
          await enhanced_manager._monitored_stop_tracking()

          # Verify shutdown
          enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 526
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:526
_ ERROR at setup of TestIntegrationScenarios.test_error_scenarios_with_alerts __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 580
      @pytest.mark.asyncio
      async def test_error_scenarios_with_alerts(
          self, enhanced_manager, mock_tracking_manager
      ):
          """Test various error scenarios with alert generation."""
          # Validation error
          validation_error = Exception("Database connection lost")
          enhanced_manager._original_methods["validate_prediction"].side_effect = (
              validation_error
          )

          with pytest.raises(Exception, match="Database connection lost"):
              await enhanced_manager._monitored_validate_prediction(
                  "bedroom", datetime.now(timezone.utc)
              )

          # Startup error
          startup_error = Exception("Configuration invalid")
          enhanced_manager._original_methods["start_tracking"].side_effect = startup_error

          with pytest.raises(Exception, match="Configuration invalid"):
              await enhanced_manager._monitored_start_tracking()

          # Verify alerts generated
          assert (
              enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_count
              == 2
          )
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 526
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:526
_ ERROR at setup of TestIntegrationScenarios.test_monitoring_integration_coverage _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 609
      def test_monitoring_integration_coverage(self, enhanced_manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 526
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:526
____ ERROR at setup of TestEdgeCases.test_prediction_with_string_model_type ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 638
      @pytest.mark.asyncio
      async def test_prediction_with_string_model_type(
          self, enhanced_manager, prediction_result
      ):
          """Test prediction recording with string model type."""
          result = await enhanced_manager._monitored_record_prediction(
              "room", prediction_result, "custom_model"
          )

          assert result == "prediction_id_123"

          # Verify string model type handled correctly
          call_args = (
              enhanced_manager.monitoring_integration.track_prediction_operation.call_args
          )
          assert call_args[1]["model_type"] == "custom_model"
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 629
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:629
_____ ERROR at setup of TestEdgeCases.test_validation_with_non_dict_result _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 655
      @pytest.mark.asyncio
      async def test_validation_with_non_dict_result(
          self, enhanced_manager, mock_tracking_manager
      ):
          """Test validation with non-dictionary result."""
          enhanced_manager._original_methods["validate_prediction"].return_value = (
              "simple_result"
          )

          result = await enhanced_manager._monitored_validate_prediction(
              "room", datetime.now(timezone.utc)
          )

          assert result == "simple_result"
          # Should not attempt to record accuracy with non-dict result
          enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 629
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:629
_ ERROR at setup of TestEdgeCases.test_concept_drift_without_tracking_manager_support _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 672
      def test_concept_drift_without_tracking_manager_support(
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 629
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:629
___ ERROR at setup of TestEdgeCases.test_context_manager_exception_handling ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 685
      @pytest.mark.asyncio
      async def test_context_manager_exception_handling(self, enhanced_manager):
          """Test model training context manager with exceptions."""
          enhanced_manager.monitoring_integration.track_training_operation.return_value = (
              MockAsyncContextManagerWithError()
          )

          with pytest.raises(RuntimeError, match="Context manager error"):
              async with enhanced_manager.track_model_training("room", "model", "type"):
                  raise RuntimeError("Training error")
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 629
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:629
___ ERROR at setup of TestPerformanceAndStress.test_high_volume_predictions ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 729
      @pytest.mark.asyncio
      async def test_high_volume_predictions(self, enhanced_manager):
          """Test handling high volume of predictions."""
          predictions = []
          for i in range(100):
              prediction = PredictionResult(
                  prediction_type="next_occupied",
                  predicted_time=datetime.now(timezone.utc),
                  confidence=0.8 + (i % 20) / 100,
                  metadata={"batch": i},
              )
              predictions.append(prediction)

          # Process all predictions
          tasks = []
          for i, prediction in enumerate(predictions):
              task = enhanced_manager._monitored_record_prediction(
                  f"room_{i % 10}", prediction, ModelType.ENSEMBLE
              )
              tasks.append(task)

          results = await asyncio.gather(*tasks)

          # Verify all processed
          assert len(results) == 100
          assert all(result == "prediction_id_123" for result in results)
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 720
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestPerformanceAndStress::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:720
____ ERROR at setup of TestPerformanceAndStress.test_rapid_metric_recording ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 756
      def test_rapid_metric_recording(self, enhanced_manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py, line 720
      @pytest.fixture
      def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
E       fixture 'mock_tracking_manager' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, enhanced_manager, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::<event_loop>, tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestPerformanceAndStress::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:720
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_success _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_with_kwargs _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_string_model_type _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_prediction_monitoring_timeout_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_concurrent_prediction_recording _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestMonitoringEnhancedTrackingManager.test_monitoring_context_manager_error _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:119: in sample_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
_________ ERROR at setup of TestAdaptiveRetrainer.test_initialization __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_____ ERROR at setup of TestAdaptiveRetrainer.test_initialize_and_shutdown _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_accuracy_trigger _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_error_trigger _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_drift_trigger _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_confidence_trigger _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_no_triggers _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_evaluate_retraining_need_cooldown _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_select_retraining_strategy_incremental _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_select_retraining_strategy_full_retrain _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_select_retraining_strategy_ensemble_rebalance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
____ ERROR at setup of TestAdaptiveRetrainer.test_request_manual_retraining ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_get_retraining_status_specific_request _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_get_retraining_status_all_requests _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
____ ERROR at setup of TestAdaptiveRetrainer.test_cancel_pending_retraining ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_______ ERROR at setup of TestAdaptiveRetrainer.test_get_retrainer_stats _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_queue_retraining_request_priority_ordering _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestAdaptiveRetrainer.test_queue_retraining_request_duplicate_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
__ ERROR at setup of TestRetrainingExecution.test_incremental_retrain_success __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_incremental_retrain_fallback_to_full _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_full_retrain_with_optimization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
______ ERROR at setup of TestRetrainingExecution.test_ensemble_rebalance _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
__ ERROR at setup of TestRetrainingExecution.test_ensemble_rebalance_fallback __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
____ ERROR at setup of TestRetrainingExecution.test_prepare_retraining_data ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_extract_features_for_retraining _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_retrain_model_with_optimization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_retrain_model_missing_from_registry _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingExecution.test_retrain_model_insufficient_data _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingProgress.test_progress_tracking_during_retraining _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
____ ERROR at setup of TestRetrainingProgress.test_get_retraining_progress _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestRetrainingProgress.test_execute_retraining_with_progress_tracking _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestIntegrationAndValidation.test_drift_detector_integration _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestIntegrationAndValidation.test_prediction_validator_integration _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestIntegrationAndValidation.test_model_optimizer_integration_status _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
__ ERROR at setup of TestIntegrationAndValidation.test_drift_detector_status ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestIntegrationAndValidation.test_prediction_validator_status _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestConcurrencyAndResourceManagement.test_concurrent_request_limit _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestConcurrencyAndResourceManagement.test_multiple_concurrent_requests _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestConcurrencyAndResourceManagement.test_resource_tracking_accuracy _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestConcurrencyAndResourceManagement.test_cooldown_tracking_thread_safety _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_retraining_execution_failure _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_model_training_failure __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_validation_failure_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_malformed_retraining_request _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_invalid_model_type_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_optimizer_failure_graceful_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestErrorHandlingAndEdgeCases.test_memory_cleanup_on_shutdown _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestPerformanceAndScalability.test_high_volume_request_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestPerformanceAndScalability.test_memory_efficiency_large_history _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestPerformanceAndScalability.test_concurrent_status_queries _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_retrainer_comprehensive.py:100: in mock_model_optimizer
    return_value=OptimizationResult(
E   TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
_ ERROR at setup of TestEventValidatorAdvanced.test_event_validator_comprehensive_initialization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_all_sensor_states _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_invalid_states_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_state_transition_matrix _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_timestamp_edge_cases _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_room_sensor_configuration_matching _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_sensor_state_enum_compliance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_confidence_score_calculation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventValidatorAdvanced.test_validate_event_performance_timing _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_classifier_initialization_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_movement_metrics_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_max_velocity_complex_patterns _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_door_interactions_multiple_doors _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_presence_sensor_ratio_complex_config _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_sensor_revisits_complex_patterns _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_avg_dwell_time_mathematical_precision _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_timing_variance_statistical_accuracy _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_movement_entropy_information_theory _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_calculate_spatial_dispersion_advanced_geometry _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_score_human_pattern_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_score_cat_pattern_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_classify_movement_comprehensive_human _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_classify_movement_comprehensive_cat _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_analyze_sequence_patterns_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_get_sequence_time_analysis_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_extract_movement_signature_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestMovementPatternClassifierAdvanced.test_compare_movement_patterns_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_event_processor_comprehensive_initialization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_process_event_comprehensive_flow _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_process_event_with_sequence_classification _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_process_event_batch_performance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_process_event_duplicate_detection_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_check_room_state_change_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_check_room_state_change_motion_patterns _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_validate_event_sequence_integrity_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_validate_event_sequence_integrity_with_anomalies _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_validate_room_configuration_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_validate_room_configuration_edge_cases _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_processing_stats_management _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_determine_sensor_type_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorAdvanced.test_movement_sequence_creation_edge_cases _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorIntegrationAdvanced.test_end_to_end_event_processing_workflow _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorIntegrationAdvanced.test_concurrent_event_processing _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorIntegrationAdvanced.test_high_volume_processing_stress _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorPerformance.test_validator_performance_benchmark _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEventProcessorPerformance.test_classifier_performance_benchmark _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:94: in comprehensive_system_config
    return SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_____ ERROR at setup of TestSchemaValidator.test_valid_sensor_event_schema _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
______ ERROR at setup of TestSchemaValidator.test_missing_required_fields ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_______ ERROR at setup of TestSchemaValidator.test_null_required_fields ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
________ ERROR at setup of TestSchemaValidator.test_invalid_field_types ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
___ ERROR at setup of TestSchemaValidator.test_valid_sensor_types_and_states ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
____ ERROR at setup of TestSchemaValidator.test_timestamp_format_validation ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
________ ERROR at setup of TestSchemaValidator.test_timezone_awareness _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
___ ERROR at setup of TestSchemaValidator.test_room_configuration_validation ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestPerformanceValidator.test_bulk_validation_small_dataset _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestPerformanceValidator.test_bulk_validation_large_dataset _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestPerformanceValidator.test_single_event_validation_performance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestPerformanceValidator.test_validation_with_errors_and_warnings _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:58: in schema_validator
    config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
____ ERROR at setup of TestComprehensiveEventValidator.test_initialization _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_validation_rules_initialization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_single_event_validation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_single_event_validation_with_security_issues _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_bulk_events_validation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_bulk_validation_with_duplicates _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_room_specific_validation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_event_data_sanitization _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
__ ERROR at setup of TestComprehensiveEventValidator.test_validation_summary ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_validation_error_context_and_suggestions _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_concurrent_validation_safety _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestComprehensiveEventValidator.test_large_scale_performance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_comprehensive_sql_injection_prevention _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_comprehensive_xss_prevention _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_path_traversal_prevention _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_sanitization_effectiveness _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_bypass_attempt_prevention _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestSecurityTestingIntegration.test_false_positive_prevention _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:91: in comprehensive_validator
    mock_config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestDatabaseManagerInitialization.test_initialize_success __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 314
      @patch.object(DatabaseManager, "_create_engine")
      @patch.object(DatabaseManager, "_setup_session_factory")
      @patch.object(DatabaseManager, "_verify_connection")
      def test_initialize_success(
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 309
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:309
_ ERROR at setup of TestDatabaseManagerInitialization.test_initialize_already_initialized _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 340
      def test_initialize_already_initialized(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 309
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:309
_ ERROR at setup of TestDatabaseManagerInitialization.test_initialize_error_handling _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 350
      @patch.object(DatabaseManager, "_create_engine")
      def test_initialize_error_handling(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 309
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:309
_ ERROR at setup of TestDatabaseManagerEngineCreation.test_create_engine_success _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 147
      @patch("src.data.storage.database.create_async_engine")
      def test_create_engine_success(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 142
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:142
_ ERROR at setup of TestDatabaseManagerEngineCreation.test_create_engine_error _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 167
      @patch("src.data.storage.database.create_async_engine")
      def test_create_engine_error(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 142
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:142
_ ERROR at setup of TestDatabaseManagerEngineCreation.test_create_engine_connection_pooling_config _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 175
      @patch("src.data.storage.database.create_async_engine")
      def test_create_engine_connection_pooling_config(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 142
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:142
_ ERROR at setup of TestDatabaseManagerEngineCreation.test_create_engine_event_listeners _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 191
      @patch("src.data.storage.database.create_async_engine")
      def test_create_engine_event_listeners(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 142
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:142
_ ERROR at setup of TestDatabaseManagerSessionFactory.test_setup_session_factory _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 214
      @patch("src.data.storage.database.async_sessionmaker")
      def test_setup_session_factory(self, mock_sessionmaker, manager_with_engine):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 207
      @pytest.fixture
      def manager_with_engine(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_engine, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionFactory::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:207
_ ERROR at setup of TestDatabaseManagerSessionFactory.test_setup_session_factory_no_engine _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 232
      def test_setup_session_factory_no_engine(self, manager_with_engine):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 207
      @pytest.fixture
      def manager_with_engine(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_engine, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionFactory::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:207
_ ERROR at setup of TestDatabaseManagerConnectionVerification.test_verify_connection_success _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 251
      @patch("src.data.storage.database.text")
      def test_verify_connection_success(self, mock_text, manager_with_session_factory):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 243
      @pytest.fixture
      def manager_with_session_factory(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session_factory, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:243
_ ERROR at setup of TestDatabaseManagerConnectionVerification.test_verify_connection_failure _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 273
      def test_verify_connection_failure(self, manager_with_session_factory):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 243
      @pytest.fixture
      def manager_with_session_factory(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session_factory, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:243
_ ERROR at setup of TestDatabaseManagerConnectionVerification.test_verify_connection_timeout _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 290
      def test_verify_connection_timeout(self, manager_with_session_factory):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 243
      @pytest.fixture
      def manager_with_session_factory(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session_factory, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:243
__ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_success __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 373
      @patch("src.data.storage.database.text")
      def test_health_check_success(self, mock_text, initialized_manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 365
      @pytest.fixture
      def initialized_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:365
__ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_failure __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 393
      def test_health_check_failure(self, initialized_manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 365
      @pytest.fixture
      def initialized_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:365
_ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_no_engine _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 410
      def test_health_check_no_engine(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:410
_ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_stats_update _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 418
      def test_health_check_stats_update(self, initialized_manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 365
      @pytest.fixture
      def initialized_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:365
___ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_loop ____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 435
      @pytest.mark.asyncio
      async def test_health_check_loop(self, initialized_manager):
          """Test background health check loop."""
          # Mock health check to avoid actual DB calls
          initialized_manager.health_check = AsyncMock(return_value={"status": "healthy"})

          # Start the health check loop
          task = asyncio.create_task(initialized_manager._health_check_loop())

          # Let it run briefly
          await asyncio.sleep(0.1)

          # Cancel the task
          task.cancel()

          try:
              await task
          except asyncio.CancelledError:
              pass

          # Verify health check was called
          initialized_manager.health_check.assert_called()
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 365
      @pytest.fixture
      def initialized_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:365
_ ERROR at setup of TestDatabaseManagerHealthCheck.test_health_check_loop_error_recovery _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 458
      @pytest.mark.asyncio
      async def test_health_check_loop_error_recovery(self, initialized_manager):
          """Test health check loop error recovery."""
          # Mock health check to fail then succeed
          initialized_manager.health_check = AsyncMock(
              side_effect=[Exception("Health check failed"), {"status": "healthy"}]
          )

          task = asyncio.create_task(initialized_manager._health_check_loop())

          # Let it run through error and recovery
          await asyncio.sleep(0.2)

          task.cancel()
          try:
              await task
          except asyncio.CancelledError:
              pass

          # Should have continued despite error
          assert initialized_manager.health_check.call_count >= 1
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 365
      @pytest.fixture
      def initialized_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, initialized_manager, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:365
_ ERROR at setup of TestDatabaseManagerSessionManagement.test_get_session_context_manager _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 492
      def test_get_session_context_manager(self, manager_with_session):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 484
      @pytest.fixture
      def manager_with_session(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:484
_ ERROR at setup of TestDatabaseManagerSessionManagement.test_get_session_no_factory _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 508
      def test_get_session_no_factory(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:508
_ ERROR at setup of TestDatabaseManagerSessionManagement.test_session_error_handling _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 519
      def test_session_error_handling(self, manager_with_session):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 484
      @pytest.fixture
      def manager_with_session(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:484
_ ERROR at setup of TestDatabaseManagerSessionManagement.test_concurrent_sessions _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 540
      def test_concurrent_sessions(self, manager_with_session):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 484
      @pytest.fixture
      def manager_with_session(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager_with_session, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:484
__ ERROR at setup of TestDatabaseManagerRetryLogic.test_calculate_retry_delay __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 587
      def test_calculate_retry_delay(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 577
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:577
_ ERROR at setup of TestDatabaseManagerRetryLogic.test_retry_with_exponential_backoff _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 602
      @patch.object(DatabaseManager, "_create_engine")
      def test_retry_with_exponential_backoff(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 577
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:577
_ ERROR at setup of TestDatabaseManagerRetryLogic.test_retry_max_retries_exceeded _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 624
      @patch.object(DatabaseManager, "_create_engine")
      def test_retry_max_retries_exceeded(self, mock_create_engine, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 577
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:577
_ ERROR at setup of TestDatabaseManagerRetryLogic.test_retry_non_retryable_error _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 640
      def test_retry_non_retryable_error(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 577
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:577
_ ERROR at setup of TestDatabaseManagerConnectionStatistics.test_stats_initialization _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 661
      def test_stats_initialization(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 656
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:656
_ ERROR at setup of TestDatabaseManagerConnectionStatistics.test_stats_update_on_success _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 671
      def test_stats_update_on_success(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 656
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:656
_ ERROR at setup of TestDatabaseManagerConnectionStatistics.test_stats_update_on_failure _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 683
      def test_stats_update_on_failure(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 656
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:656
_ ERROR at setup of TestDatabaseManagerConnectionStatistics.test_success_rate_calculation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 697
      def test_success_rate_calculation(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 656
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:656
__ ERROR at setup of TestDatabaseManagerConnectionStatistics.test_stats_reset __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 708
      def test_stats_reset(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 656
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:656
______ ERROR at setup of TestDatabaseManagerCleanup.test_cleanup_success _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 735
      @pytest.mark.asyncio
      async def test_cleanup_success(self, active_manager):
          """Test successful cleanup."""
          # Mock task cancellation
          active_manager._health_check_task.cancel = MagicMock()
          active_manager._health_check_task.done.return_value = False
          active_manager._health_check_task.cancelled.return_value = True

          # Mock engine disposal
          active_manager.engine.dispose = AsyncMock()

          await active_manager.cleanup()

          # Verify cleanup steps
          active_manager._health_check_task.cancel.assert_called_once()
          active_manager.engine.dispose.assert_called_once()

          assert active_manager.engine is None
          assert active_manager.session_factory is None
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 726
      @pytest.fixture
      def active_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, active_manager, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:726
_ ERROR at setup of TestDatabaseManagerCleanup.test_cleanup_no_active_components _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 755
      @pytest.mark.asyncio
      async def test_cleanup_no_active_components(self, mock_config):
          """Test cleanup when no active components."""
          manager = DatabaseManager(config=mock_config)

          # Should not raise errors
          await manager.cleanup()
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, active_manager, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:755
_ ERROR at setup of TestDatabaseManagerCleanup.test_cleanup_task_cancellation_error _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 763
      @pytest.mark.asyncio
      async def test_cleanup_task_cancellation_error(self, active_manager):
          """Test cleanup when task cancellation fails."""
          active_manager._health_check_task.cancel.side_effect = RuntimeError(
              "Cancel failed"
          )
          active_manager.engine.dispose = AsyncMock()

          # Should still complete cleanup
          await active_manager.cleanup()

          active_manager.engine.dispose.assert_called_once()
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 726
      @pytest.fixture
      def active_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, active_manager, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:726
_ ERROR at setup of TestDatabaseManagerCleanup.test_cleanup_engine_disposal_error _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 776
      @pytest.mark.asyncio
      async def test_cleanup_engine_disposal_error(self, active_manager):
          """Test cleanup when engine disposal fails."""
          active_manager._health_check_task = None  # No task to cancel
          active_manager.engine.dispose = AsyncMock(
              side_effect=RuntimeError("Dispose failed")
          )

          # Should handle error gracefully
          await active_manager.cleanup()

          # Engine should still be cleared
          assert active_manager.engine is None
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 726
      @pytest.fixture
      def active_manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, active_manager, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:726
_ ERROR at setup of TestDatabaseManagerSecurityAndTimeout.test_connection_string_sanitization _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 799
      def test_connection_string_sanitization(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 794
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:794
_ ERROR at setup of TestDatabaseManagerSecurityAndTimeout.test_query_timeout_enforcement _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 812
      @pytest.mark.asyncio
      async def test_query_timeout_enforcement(self, manager):
          """Test query timeout enforcement."""

          # Mock slow query that should timeout
          async def slow_query():
              await asyncio.sleep(2)  # Longer than timeout
              return "result"

          manager.query_timeout = timedelta(milliseconds=100)

          with pytest.raises(asyncio.TimeoutError):
              await asyncio.wait_for(
                  slow_query(), timeout=manager.query_timeout.total_seconds()
              )
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 794
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:794
_ ERROR at setup of TestDatabaseManagerSecurityAndTimeout.test_connection_pool_security _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 828
      def test_connection_pool_security(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 794
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:794
_ ERROR at setup of TestDatabaseManagerSecurityAndTimeout.test_sql_injection_prevention _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 834
      def test_sql_injection_prevention(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 794
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:794
_ ERROR at setup of TestDatabaseManagerErrorRecovery.test_connection_recovery_after_disconnect _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 849
      @pytest.mark.asyncio
      async def test_connection_recovery_after_disconnect(self, manager):
          """Test recovery after database disconnection."""
          manager.engine = MagicMock()
          manager.session_factory = MagicMock()

          # Mock session that fails with disconnection error
          mock_session = AsyncMock()
          mock_session.execute.side_effect = DisconnectionError(
              "Connection lost", None, None
          )

          manager.session_factory.return_value.__aenter__ = AsyncMock(
              return_value=mock_session
          )
          manager.session_factory.return_value.__aexit__ = AsyncMock()

          # Health check should detect and report disconnection
          result = await manager.health_check()

          assert result["status"] == "unhealthy"
          assert "error" in result
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 844
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:844
_ ERROR at setup of TestDatabaseManagerErrorRecovery.test_engine_recreation_on_failure _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 872
      def test_engine_recreation_on_failure(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 844
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:844
_ ERROR at setup of TestDatabaseManagerErrorRecovery.test_graceful_degradation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 878
      @pytest.mark.asyncio
      async def test_graceful_degradation(self, manager):
          """Test graceful degradation when database is unavailable."""
          manager.engine = None  # Simulate uninitialized state

          # Operations should fail gracefully
          with pytest.raises(DatabaseConnectionError):
              async with manager.get_session():
                  pass

          health_result = await manager.health_check()
          assert health_result["status"] == "not_initialized"
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 844
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:844
_ ERROR at setup of TestDatabaseManagerPerformanceMonitoring.test_connection_pool_metrics _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 900
      def test_connection_pool_metrics(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 895
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:895
_ ERROR at setup of TestDatabaseManagerPerformanceMonitoring.test_query_performance_tracking _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 919
      def test_query_performance_tracking(self, manager):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 895
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:895
_ ERROR at setup of TestDatabaseManagerPerformanceMonitoring.test_health_check_response_time _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 925
      @pytest.mark.asyncio
      async def test_health_check_response_time(self, manager):
          """Test health check response time measurement."""
          manager.engine = MagicMock()
          manager.session_factory = MagicMock()

          # Mock fast responding session
          mock_session = AsyncMock()
          mock_result = MagicMock()
          mock_result.scalar.return_value = 1
          mock_session.execute.return_value = mock_result

          manager.session_factory.return_value.__aenter__ = AsyncMock(
              return_value=mock_session
          )
          manager.session_factory.return_value.__aexit__ = AsyncMock()

          result = await manager.health_check()

          assert "response_time_ms" in result
          assert result["response_time_ms"] >= 0
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py, line 895
      @pytest.fixture
      def manager(self, mock_config):
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, large_sample_training_data, manager, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_validator, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_validation_data, sample_room_states_data, sample_sensor_events, sample_training_data, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_data/test_database_comprehensive.py::<event_loop>, tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_database_comprehensive.py:895
_ ERROR at setup of TestDatabaseManagerIntegrationScenarios.test_full_lifecycle _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_database_comprehensive.py:954: in realistic_config
    return DatabaseConfig(
E   TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
_ ERROR at setup of TestDatabaseManagerIntegrationScenarios.test_concurrent_operations _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_database_comprehensive.py:954: in realistic_config
    return DatabaseConfig(
E   TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
_ ERROR at setup of TestDatabaseManagerIntegrationScenarios.test_error_recovery_scenario _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_database_comprehensive.py:954: in realistic_config
    return DatabaseConfig(
E   TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
=================================== FAILURES ===================================
_________ TestConceptDriftDetector.test_detect_accuracy_drift_no_drift _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:276: in test_detect_accuracy_drift_no_drift
    drift_result = await drift_detector.detect_accuracy_drift(
E   AttributeError: 'ConceptDriftDetector' object has no attribute 'detect_accuracy_drift'
____ TestConceptDriftDetector.test_detect_accuracy_drift_significant_drift _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:289: in test_detect_accuracy_drift_significant_drift
    drift_result = await drift_detector.detect_accuracy_drift(
E   AttributeError: 'ConceptDriftDetector' object has no attribute 'detect_accuracy_drift'
__ TestConceptDriftDetector.test_calculate_population_stability_index_stable ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:306: in test_calculate_population_stability_index_stable
    psi = drift_detector._calculate_population_stability_index(baseline, current)
E   AttributeError: 'ConceptDriftDetector' object has no attribute '_calculate_population_stability_index'
___ TestConceptDriftDetector.test_calculate_population_stability_index_drift ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:314: in test_calculate_population_stability_index_drift
    psi = drift_detector._calculate_population_stability_index(baseline, current)
E   AttributeError: 'ConceptDriftDetector' object has no attribute '_calculate_population_stability_index'
_______ TestConceptDriftDetector.test_perform_page_hinkley_test_no_drift _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:321: in test_perform_page_hinkley_test_no_drift
    result = drift_detector._perform_page_hinkley_test(accuracies)
E   AttributeError: 'ConceptDriftDetector' object has no attribute '_perform_page_hinkley_test'. Did you mean: '_run_page_hinkley_test'?
______ TestConceptDriftDetector.test_perform_page_hinkley_test_with_drift ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:329: in test_perform_page_hinkley_test_with_drift
    result = drift_detector._perform_page_hinkley_test(accuracies)
E   AttributeError: 'ConceptDriftDetector' object has no attribute '_perform_page_hinkley_test'. Did you mean: '_run_page_hinkley_test'?
_________ TestFeatureDriftDetector.test_detect_feature_drift_no_drift __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:373: in test_detect_feature_drift_no_drift
    drift_result = await feature_drift_detector.detect_feature_drift(
src/adaptation/drift_detector.py:1264: in detect_feature_drift
    if "timestamp" in feature_data.columns:
E   AttributeError: 'str' object has no attribute 'columns'
________ TestFeatureDriftDetector.test_detect_feature_drift_with_drift _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:409: in test_detect_feature_drift_with_drift
    drift_result = await feature_drift_detector.detect_feature_drift(
src/adaptation/drift_detector.py:1264: in detect_feature_drift
    if "timestamp" in feature_data.columns:
E   AttributeError: 'str' object has no attribute 'columns'
_______________ TestFeatureDriftDetector.test_statistical_tests ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:426: in test_statistical_tests
    stable_ks = feature_drift_detector._perform_kolmogorov_smirnov_test(
E   AttributeError: 'FeatureDriftDetector' object has no attribute '_perform_kolmogorov_smirnov_test'
_________________ TestRealTimeMetrics.test_is_healthy_property _________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:509: in test_is_healthy_property
    assert healthy_metrics.is_healthy
E   AssertionError: assert False
E    +  where False = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=0.0, window_24h_accuracy=85.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=0, window_24h_predictions=50, accuracy_trend=<TrendDirection.STABLE: 'stable'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 7, 187902), measurement_start=None).is_healthy
_________________ TestAccuracyAlert.test_alert_initialization __________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:525: in test_alert_initialization
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
___________________ TestAccuracyAlert.test_alert_resolution ____________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:545: in test_alert_resolution
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
____________ TestPredictionValidator.test_validator_initialization _____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:672: in test_validator_initialization
    assert len(prediction_validator.pending_validations) == 0
E   AttributeError: 'PredictionValidator' object has no attribute 'pending_validations'. Did you mean: 'get_pending_validations'?
________________ TestPredictionValidator.test_record_prediction ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:679: in test_record_prediction
    await prediction_validator.record_prediction(
E   TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
__________ TestPredictionValidator.test_validate_prediction_accurate ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:696: in test_validate_prediction_accurate
    await prediction_validator.record_prediction(
E   TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
_________ TestPredictionValidator.test_validate_prediction_inaccurate __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:722: in test_validate_prediction_inaccurate
    await prediction_validator.record_prediction(
E   TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
______________ TestPredictionValidator.test_get_accuracy_metrics _______________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/adaptation/validator.py:723: in get_accuracy_metrics
    memory_records = self._get_filtered_records(
src/adaptation/validator.py:1729: in _get_filtered_records
    cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)
E   TypeError: unsupported type for timedelta hours component: datetime.datetime

During handling of the above exception, another exception occurred:
tests/unit/test_adaptation_consolidated.py:768: in test_get_accuracy_metrics
    metrics = await prediction_validator.get_accuracy_metrics(
src/adaptation/validator.py:782: in get_accuracy_metrics
    raise ValidationError("Failed to calculate accuracy metrics", cause=e)
E   src.adaptation.validator.ValidationError: Failed to calculate accuracy metrics | Error Code: VALIDATION_ERROR | Caused by: TypeError: unsupported type for timedelta hours component: datetime.datetime
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.validator:validator.py:781 Failed to calculate accuracy metrics: unsupported type for timedelta hours component: datetime.datetime
_ TestConfigurationCorruptionAndRecovery.test_yaml_with_circular_references_deep _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config.py:1196: in test_yaml_with_circular_references_deep
    config = loader.load_config()
src/core/config.py:425: in load_config
    ha_config = HomeAssistantConfig(**main_config["home_assistant"])
E   TypeError: HomeAssistantConfig.__init__() got an unexpected keyword argument 'reference'
_ TestConfigurationCorruptionAndRecovery.test_configuration_with_malformed_data_types _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config.py:1277: in test_configuration_with_malformed_data_types
    with pytest.raises(TypeError):
E   Failed: DID NOT RAISE <class 'TypeError'>
_ TestConfigurationRecoveryMechanisms.test_configuration_fallback_to_defaults __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config.py:1350: in test_configuration_fallback_to_defaults
    config = loader.load_config()
src/core/config.py:428: in load_config
    prediction_config = PredictionConfig(**main_config["prediction"])
E   KeyError: 'prediction'
__ TestConfigurationRecoveryMechanisms.test_configuration_validation_recovery __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config.py:1420: in test_configuration_validation_recovery
    with pytest.raises(TypeError):
E   Failed: DID NOT RAISE <class 'TypeError'>
_ TestAdaptationPerformanceEdgeCases.test_large_validation_history_performance _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:1282: in test_large_validation_history_performance
    tracker = AccuracyTracker(room_id="test_room", max_history_size=10000)
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_ TestAdaptationPerformanceEdgeCases.test_memory_management_with_history_limit _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:1309: in test_memory_management_with_history_limit
    tracker = AccuracyTracker(room_id="test_room", max_history_size=100)
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_________ TestAdaptationPerformanceEdgeCases.test_edge_case_empty_data _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:1331: in test_edge_case_empty_data
    tracker = AccuracyTracker(room_id="test_room")
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
_______ TestAdaptationPerformanceEdgeCases.test_extreme_accuracy_values ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:1343: in test_extreme_accuracy_values
    tracker = AccuracyTracker(room_id="test_room")
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
___ TestAdaptationPerformanceEdgeCases.test_concurrent_validation_recording ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation_consolidated.py:1381: in test_concurrent_validation_recording
    tracker = AccuracyTracker(room_id="test_room")
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
________ TestOccupancyPredictionSystem.test_run_without_initialization _________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_main_system.py:265: in test_run_without_initialization
    await run_with_timeout()
tests/unit/test_main_system.py:261: in run_with_timeout
    await asyncio.wait_for(system.run(), timeout=0.1)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/asyncio/tasks.py:507: in wait_for
    return await fut
src/main_system.py:114: in run
    await self.shutdown()
src/main_system.py:122: in shutdown
    await self.tracking_manager.stop_tracking()
E   TypeError: object MagicMock can't be used in 'await' expression
_ TestOccupancyPredictionSystem.test_shutdown_handles_tracking_manager_exception _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_main_system.py:444: in test_shutdown_handles_tracking_manager_exception
    assert system.running is False
E   assert True is False
E    +  where True = <src.main_system.OccupancyPredictionSystem object at 0x7fef0cf9e5b0>.running
_ TestOccupancyPredictionSystem.test_system_passes_correct_config_to_components _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_main_system.py:511: in test_system_passes_correct_config_to_components
    await system.initialize()
src/main_system.py:64: in initialize
    if api_status["enabled"] and api_status["running"]:
E   TypeError: 'coroutine' object is not subscriptable
------------------------------ Captured log call -------------------------------
ERROR    src.main_system:main_system.py:82 ❌ Failed to initialize system: 'coroutine' object is not subscriptable
Traceback (most recent call last):
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/main_system.py", line 64, in initialize
    if api_status["enabled"] and api_status["running"]:
       ~~~~~~~~~~^^^^^^^^^^^
TypeError: 'coroutine' object is not subscriptable
______ TestMainSystemErrorScenarios.test_api_server_status_check_failure _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_main_system.py:700: in test_api_server_status_check_failure
    await system.initialize()
src/main_system.py:64: in initialize
    if api_status["enabled"] and api_status["running"]:
E   TypeError: 'coroutine' object is not subscriptable

During handling of the above exception, another exception occurred:
tests/unit/test_main_system.py:699: in test_api_server_status_check_failure
    with pytest.raises(Exception, match="API status check failed"):
E   AssertionError: Regex pattern did not match.
E    Regex: 'API status check failed'
E    Input: "'coroutine' object is not subscriptable"
------------------------------ Captured log call -------------------------------
ERROR    src.main_system:main_system.py:82 ❌ Failed to initialize system: 'coroutine' object is not subscriptable
Traceback (most recent call last):
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/main_system.py", line 64, in initialize
    if api_status["enabled"] and api_status["running"]:
       ~~~~~~~~~~^^^^^^^^^^^
TypeError: 'coroutine' object is not subscriptable
_ TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_without_confidence _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py:168: in test_monitored_record_prediction_without_confidence
    prediction_result = PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_type'. Did you mean 'predicted_time'?
__________ TestMonitoringEnhancedTrackingManager.test_initialization ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:148: in test_initialization
    assert (
E   AssertionError: assert <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> != <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>>
E    +  where <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> = <MagicMock spec='TrackingManager' id='140664671116256'>.record_prediction
E    +    where <MagicMock spec='TrackingManager' id='140664671116256'> = <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>.tracking_manager
E    +  and   <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> = <MagicMock spec='TrackingManager' id='140664671116256'>.record_prediction
__________ TestMonitoringEnhancedTrackingManager.test_method_wrapping __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:159: in test_method_wrapping
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1481: in __enter__
    self.target = self.getter()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/pkgutil.py:528: in resolve_name
    result = getattr(result, p)
E   AttributeError: module 'src.adaptation' has no attribute 'monitoring_integration'
_ TestMonitoringEnhancedTrackingManager.test_monitored_record_prediction_no_confidence _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:241: in test_monitored_record_prediction_no_confidence
    prediction_result = PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
___ TestMonitoringEnhancedTrackingManager.test_track_model_training_context ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:668: in test_track_model_training_context
    async with enhanced_tracking_manager.track_model_training(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/contextlib.py:214: in __aenter__
    return await anext(self.gen)
src/adaptation/monitoring_enhanced_tracking.py:250: in track_model_training
    async with self.monitoring_integration.track_training_operation(
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
_ TestMonitoringEnhancedTrackingManager.test_track_model_training_default_type _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:684: in test_track_model_training_default_type
    async with enhanced_tracking_manager.track_model_training(room_id, model_type):
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/contextlib.py:214: in __aenter__
    return await anext(self.gen)
src/adaptation/monitoring_enhanced_tracking.py:250: in track_model_training
    async with self.monitoring_integration.track_training_operation(
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
_ TestMonitoringEnhancedTrackingManager.test_monitoring_integration_none_error _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:771: in test_monitoring_integration_none_error
    with pytest.raises(AttributeError):
E   Failed: DID NOT RAISE <class 'AttributeError'>
_ TestMonitoringEnhancedTrackingManager.test_empty_prediction_result_attributes _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:784: in test_empty_prediction_result_attributes
    asyncio.run(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/asyncio/runners.py:195: in run
    return runner.run(main)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/asyncio/base_events.py:725: in run_until_complete
    return future.result()
src/adaptation/monitoring_enhanced_tracking.py:62: in _monitored_record_prediction
    async with self.monitoring_integration.track_prediction_operation(
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
___________ TestFactoryFunctions.test_get_enhanced_tracking_manager ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:979: in assert_called_with
    raise AssertionError(_error_message()) from cause
E   AssertionError: expected call not found.
E   Expected: create_monitoring_enhanced_tracking_manager(config=<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
E     Actual: create_monitoring_enhanced_tracking_manager(<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')

During handling of the above exception, another exception occurred:
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:991: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
E   AssertionError: expected call not found.
E   Expected: create_monitoring_enhanced_tracking_manager(config=<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
E     Actual: create_monitoring_enhanced_tracking_manager(<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
E   
E   pytest introspection follows:
E   
E   Args:
E   assert (<MagicMock s...4673065616'>,) == ()
E     Left contains one more item: <MagicMock spec='TrackingConfig' id='140664673065616'>
E     Full diff:
E     - ()
E     + (<MagicMock spec='TrackingConfig' id='140664673065616'>,)
E   Kwargs:
E   assert {'test_param': 'value'} == {'config': <M...ram': 'value'}
E     Omitting 1 identical items, use -vv to show
E     Right contains 1 more item:
E     {'config': <MagicMock spec='TrackingConfig' id='140664673065616'>}
E     Full diff:
E       {
E     -  'config': <MagicMock spec='TrackingConfig' id='140664673065616'>,
E        'test_param': 'value',
E       }

During handling of the above exception, another exception occurred:
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:858: in test_get_enhanced_tracking_manager
    mock_create_enhanced.assert_called_once_with(
E   AssertionError: expected call not found.
E   Expected: create_monitoring_enhanced_tracking_manager(config=<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
E     Actual: create_monitoring_enhanced_tracking_manager(<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
E   
E   pytest introspection follows:
E   
E   Args:
E   assert (<MagicMock s...4673065616'>,) == ()
E     Left contains one more item: <MagicMock spec='TrackingConfig' id='140664673065616'>
E     Full diff:
E     - ()
E     + (<MagicMock spec='TrackingConfig' id='140664673065616'>,)
E   Kwargs:
E   assert {'test_param': 'value'} == {'config': <M...ram': 'value'}
E     Omitting 1 identical items, use -vv to show
E     Right contains 1 more item:
E     {'config': <MagicMock spec='TrackingConfig' id='140664673065616'>}
E     Full diff:
E       {
E     -  'config': <MagicMock spec='TrackingConfig' id='140664673065616'>,
E        'test_param': 'value',
E       }
_________ TestIntegrationScenarios.test_complete_prediction_lifecycle __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:910: in test_complete_prediction_lifecycle
    prediction_result = PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
____________ TestIntegrationScenarios.test_error_recovery_scenarios ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py:963: in test_error_recovery_scenarios
    tracking_manager.validate_prediction.side_effect = ConnectionError(
E   AttributeError: 'method' object has no attribute 'side_effect' and no __dict__ for setting new attributes
_ TestConfigurationStabilityAndResilience.test_configuration_loading_with_system_stress _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config.py:1513: in test_configuration_loading_with_system_stress
    assert successful_loads > 100  # At least 20 per second
E   assert 17 > 100
__________ TestRealTimeMetrics.test_overall_health_score_calculation ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:70: in test_overall_health_score_calculation
    assert metrics.overall_health_score > 80.0
E   AssertionError: assert 69.2 > 80.0
E    +  where 69.2 = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=92.0, window_24h_accuracy=88.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=15, window_24h_predictions=120, accuracy_trend=<TrendDirection.IMPROVING: 'improving'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 17, 45678), measurement_start=None).overall_health_score
_________________ TestRealTimeMetrics.test_is_healthy_criteria _________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:108: in test_is_healthy_criteria
    assert metrics_healthy.is_healthy
E   AssertionError: assert False
E    +  where False = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=70.0, window_24h_accuracy=70.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=5, window_24h_predictions=40, accuracy_trend=<TrendDirection.UNKNOWN: 'unknown'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 17, 70568), measurement_start=None).is_healthy
________________ TestRealTimeMetrics.test_to_dict_serialization ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:131: in test_to_dict_serialization
    assert result["model_type"] == "LSTM"
E   AssertionError: assert 'ModelType.LSTM' == 'LSTM'
E     - LSTM
E     + ModelType.LSTM
_____________ TestAccuracyAlert.test_accuracy_alert_initialization _____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:151: in test_accuracy_alert_initialization
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
____________________ TestAccuracyAlert.test_age_calculation ____________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:180: in test_age_calculation
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
________________ TestAccuracyAlert.test_escalation_requirements ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:198: in test_escalation_requirements
    critical_alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
___________________ TestAccuracyAlert.test_acknowledge_alert ___________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:230: in test_acknowledge_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_____________________ TestAccuracyAlert.test_resolve_alert _____________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:250: in test_resolve_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
____________________ TestAccuracyAlert.test_escalate_alert _____________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:271: in test_escalate_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_____________________ TestAccuracyAlert.test_alert_to_dict _____________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:292: in test_alert_to_dict
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_______________ TestAccuracyTracker.test_tracker_initialization ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:358: in test_tracker_initialization
    assert accuracy_tracker._prediction_validator == mock_validator
E   AttributeError: 'AccuracyTracker' object has no attribute '_prediction_validator'
____________ TestTrendAnalysis.test_analyze_trend_insufficient_data ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:684: in test_analyze_trend_insufficient_data
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
________________ TestTrendAnalysis.test_analyze_trend_improving ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:696: in test_analyze_trend_improving
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
________________ TestTrendAnalysis.test_analyze_trend_degrading ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:708: in test_analyze_trend_degrading
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
_________________ TestTrendAnalysis.test_analyze_trend_stable __________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:720: in test_analyze_trend_stable
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
________________ TestTrendAnalysis.test_calculate_global_trend _________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:737: in test_calculate_global_trend
    assert "overall_direction" in global_trend
E   AssertionError: assert 'overall_direction' in {'average_slope': 0.0, 'confidence': 0.0, 'direction': <TrendDirection.UNKNOWN: 'unknown'>}
_____________ TestTrendAnalysis.test_calculate_global_trend_empty ______________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:748: in test_calculate_global_trend_empty
    assert global_trend["overall_direction"] == TrendDirection.UNKNOWN
E   KeyError: 'overall_direction'
________ TestModelBackupManager.test_create_backup_uncompressed_success ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:463: in test_create_backup_uncompressed_success
    assert "test_models.tar" in call_args[-3]
E   AssertionError: assert 'test_models.tar' in '-C'
_______ TestModelBackupManager.test_create_backup_nonexistent_models_dir _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:514: in test_create_backup_nonexistent_models_dir
    assert Path(nonexistent_dir).exists()
E   AssertionError: assert False
E    +  where False = <bound method PathBase.exists of PosixPath('/path/that/does/not/exist')>()
E    +    where <bound method PathBase.exists of PosixPath('/path/that/does/not/exist')> = PosixPath('/path/that/does/not/exist').exists
E    +      where PosixPath('/path/that/does/not/exist') = Path('/path/that/does/not/exist')
________________ TestBackupManager.test_list_backups_with_data _________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:739: in test_list_backups_with_data
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/pathlib/_local.py:722: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpzsxzshs8/database'
_____________ TestBackupManager.test_list_backups_filtered_by_type _____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:771: in test_list_backups_filtered_by_type
    db_dir.mkdir()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/pathlib/_local.py:722: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpekluij2_/database'
_________________ TestBackupManager.test_get_backup_info_found _________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:809: in test_get_backup_info_found
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/pathlib/_local.py:722: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpoovz5ajt/database'
________________ TestBackupManager.test_cleanup_expired_backups ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_backup_manager.py:867: in test_cleanup_expired_backups
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/pathlib/_local.py:722: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpcv_cmbhh/database'
__________ TestBackupManager.test_run_scheduled_backups_single_cycle ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:958: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'create_backup' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/test_core/test_backup_manager.py:986: in test_run_scheduled_backups_single_cycle
    mock_config.assert_called_once()  # Config backup triggered due to hour 2
E   AssertionError: Expected 'create_backup' to have been called once. Called 0 times.
___________ test_event_processor_comprehensive_test_suite_completion ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_processor_comprehensive.py:2337: in test_event_processor_comprehensive_test_suite_completion
    assert total_methods >= 65, f"Expected 65+ test methods, found {total_methods}"
E   AssertionError: Expected 65+ test methods, found 54
E   assert 54 >= 65
_____________ TestSecurityValidator.test_path_traversal_detection ______________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:237: in test_path_traversal_detection
    assert (
E   AssertionError: Failed to detect path traversal: ..%252f..%252f..%252fetc%252fpasswd
E   assert 0 > 0
E    +  where 0 = len([])
___________ TestSecurityValidator.test_input_sanitization_aggressive ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:310: in test_input_sanitization_aggressive
    assert "=" not in sanitized
E   AssertionError: assert '=' not in '* FROM user... id=1 OR 1=1'
E     '=' is contained here:
E       * FROM users WHERE id=1 OR 1=1
E     ?                      +
_______________ TestIntegrityValidator.test_duplicate_detection ________________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:628: in test_duplicate_detection
    assert len(duplicate_errors) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_______ TestIntegrityValidator.test_cross_system_consistency_validation ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:685: in test_cross_system_consistency_validation
    assert len(transition_errors) > 0
E   assert 0 > 0
E    +  where 0 = len([])
___________ TestPerformanceValidator.test_performance_stats_tracking ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_event_validator_comprehensive.py:817: in test_performance_stats_tracking
    assert "batches_processed" in initial_stats
E   AssertionError: assert 'batches_processed' in {}
______ TestSensorEventAdvancedFeatures.test_get_sensor_efficiency_metrics ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.OperationalError: misuse of window function lag()

The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:190: in test_get_sensor_efficiency_metrics
    metrics = await SensorEvent.get_sensor_efficiency_metrics(
src/data/storage/models.py:420: in get_sensor_efficiency_metrics
    result = await session.execute(efficiency_query)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/context.py:306: in orm_execute_statement
    result = conn.execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) misuse of window function lag()
E   [SQL: SELECT sensor_events.sensor_id, sensor_events.sensor_type, count(*) AS total_events, count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state) AS state_changes, avg(sensor_events.confidence_score) AS avg_confidence, min(sensor_events.confidence_score) AS min_confidence, max(sensor_events.confidence_score) AS max_confidence, (count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state)) / (CAST(count(*) AS FLOAT) + 0.0) AS state_change_ratio, CAST(STRFTIME('%s', avg(sensor_events.timestamp - lag(sensor_events.timestamp) OVER (PARTITION BY sensor_events.sensor_id ORDER BY sensor_events.timestamp))) AS INTEGER) AS avg_interval_seconds 
E   FROM sensor_events 
E   WHERE sensor_events.room_id = ? AND sensor_events.timestamp >= ? GROUP BY sensor_events.sensor_id, sensor_events.sensor_type ORDER BY count(*) DESC]
E   [parameters: ('efficiency_room', '2025-08-22 07:57:24.004992')]
E   (Background on this error at: https://sqlalche.me/e/20/e3q8)
______ TestSensorEventAdvancedFeatures.test_efficiency_score_calculation _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_models_advanced.py:257: in test_efficiency_score_calculation
    assert (
E   AssertionError: Score 0.6200000000000001 not in range 0.2-0.6
E   assert 0.6200000000000001 <= 0.6
__________ TestTimescaleDBFunctions.test_create_timescale_hypertables __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_models_advanced.py:1407: in test_create_timescale_hypertables
    assert mock_execute.call_count >= 6
E   AssertionError: assert 5 >= 6
E    +  where 5 = <AsyncMock name='execute' id='140664672170208'>.call_count
____ TestDatabaseCompatibilityHelpers.test_get_json_column_type_postgresql _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_models_advanced.py:1512: in test_get_json_column_type_postgresql
    assert json_type == JSONB
E   AssertionError: assert <class 'sqlalchemy.sql.sqltypes.JSON'> == <class 'sqlalchemy.dialects.postgresql.json.JSONB'>
____________ TestModelsIntegration.test_model_relationships_cascade ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.IntegrityError: NOT NULL constraint failed: prediction_audits.prediction_id

The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:1563: in test_model_relationships_cascade
    await test_db_session.commit()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/ext/asyncio/session.py:1014: in commit
    await greenlet_spawn(self.sync_session.commit)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:2032: in commit
    trans.commit(_to_root=True)
<string>:2: in commit
    ???
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/state_changes.py:137: in _go
    ret_value = fn(self, *arg, **kw)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:1313: in commit
    self._prepare_impl()
<string>:2: in _prepare_impl
    ???
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/state_changes.py:137: in _go
    ret_value = fn(self, *arg, **kw)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:1288: in _prepare_impl
    self.session.flush()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:4345: in flush
    self._flush(objects)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:4480: in _flush
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/session.py:4441: in _flush
    flush_context.execute()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/unitofwork.py:466: in execute
    rec.execute(self)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/unitofwork.py:642: in execute
    util.preloaded.orm_persistence.save_obj(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/persistence.py:85: in save_obj
    _emit_update_statements(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/persistence.py:912: in _emit_update_statements
    c = connection.execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: prediction_audits.prediction_id
E   [SQL: UPDATE prediction_audits SET prediction_id=? WHERE prediction_audits.id = ?]
E   [parameters: (None, 1)]
E   (Background on this error at: https://sqlalche.me/e/20/gkpj)
_ TestStatisticalPatternAnalyzer.test_statistical_anomaly_detection_with_outliers _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:275: in test_statistical_anomaly_detection_with_outliers
    assert anomalies["anomaly_count"] == 3
E   assert 2 == 3
_ TestStatisticalPatternAnalyzer.test_detect_sensor_malfunction_high_frequency _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:333: in test_detect_sensor_malfunction_high_frequency
    assert len(anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_ TestStatisticalPatternAnalyzer.test_detect_sensor_malfunction_low_frequency __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:380: in test_detect_sensor_malfunction_low_frequency
    assert len(low_freq_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_ TestStatisticalPatternAnalyzer.test_detect_sensor_malfunction_unstable_timing _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:426: in test_detect_sensor_malfunction_unstable_timing
    assert len(timing_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
___________ TestCorruptionDetector.test_detect_timestamp_corruption ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:560: in test_detect_timestamp_corruption
    assert "COR003" in error_rules  # Duplicate timestamps
E   AssertionError: assert 'COR003' in {'COR001', 'COR002'}
__________ TestRealTimeQualityMonitor.test_accuracy_score_calculation __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:825: in test_accuracy_score_calculation
    assert 0.4 <= metrics.accuracy_score <= 0.8  # Approximately 60% accuracy
E   assert 0.8666666666666667 <= 0.8
E    +  where 0.8666666666666667 = DataQualityMetrics(completeness_score=0.0, consistency_score=0.5, accuracy_score=0.8666666666666667, timeliness_score=1.0, anomaly_count=0, corruption_indicators=[], quality_trends={}).accuracy_score
______ TestEdgeCasesAndErrorHandling.test_analyzer_with_malformed_events _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:1003: in test_analyzer_with_malformed_events
    analysis = pattern_analyzer.analyze_sensor_behavior(
src/data/validation/pattern_detector.py:120: in analyze_sensor_behavior
    if "timestamp" in event and "state" in event:
E   TypeError: argument of type 'NoneType' is not iterable
_ TestEdgeCasesAndErrorHandling.test_quality_monitor_with_inconsistent_data_types _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_pattern_detector_comprehensive.py:1039: in test_quality_monitor_with_inconsistent_data_types
    metrics = quality_monitor.calculate_quality_metrics(inconsistent_events, set())
src/data/validation/pattern_detector.py:563: in calculate_quality_metrics
    event.get("sensor_id") for event in events if event.get("sensor_id")
E   TypeError: unhashable type: 'list'
_________ TestJSONSchemaValidator.test_iso_datetime_format_validation __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:245: in test_iso_datetime_format_validation
    assert not validator(dt_str), f"Invalid datetime accepted: {dt_str}"
E   AssertionError: Invalid datetime accepted: 2024-01-15 10:30:00
E   assert not True
E    +  where True = <bound method JSONSchemaValidator._validate_iso_datetime_format of <src.data.validation.schema_validator.JSONSchemaValidator object at 0x7fef2892f1d0>>('2024-01-15 10:30:00')
___________ TestJSONSchemaValidator.test_duration_format_validation ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:349: in test_duration_format_validation
    assert not validator(duration), f"Invalid duration accepted: {duration}"
E   AssertionError: Invalid duration accepted: P
E   assert not True
E    +  where True = <bound method JSONSchemaValidator._validate_duration_format of <src.data.validation.schema_validator.JSONSchemaValidator object at 0x7fef28990e50>>('P')
_____ TestJSONSchemaValidator.test_validate_sensor_event_schema_valid_data _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:360: in test_validate_sensor_event_schema_valid_data
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='0f2dbbda-14a1-4102-8cfc-7a54bb65adad').is_valid
_ TestJSONSchemaValidator.test_validate_sensor_event_schema_missing_required_fields _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:380: in test_validate_sensor_event_schema_missing_required_fields
    assert len(result.errors) >= 3  # At least 3 missing required fields
E   assert 1 >= 3
E    +  where 1 = len([ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})])
E    +    where [ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})] = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='ee51bca3-3ce4-4964-8e22-bbb7c1a997c6').errors
___ TestJSONSchemaValidator.test_validate_sensor_event_schema_invalid_types ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:412: in test_validate_sensor_event_schema_invalid_types
    assert any("TYPE" in rule for rule in error_rules)
E   assert False
E    +  where False = any(<generator object TestJSONSchemaValidator.test_validate_sensor_event_schema_invalid_types.<locals>.<genexpr> at 0x7fef0d04e0c0>)
_____ TestJSONSchemaValidator.test_validate_room_config_schema_valid_data ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:424: in test_validate_room_config_schema_valid_data
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='390b42a5-6725-4607-80fb-f274d4db49c4').is_valid
__ TestJSONSchemaValidator.test_validate_room_config_schema_invalid_structure __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:455: in test_validate_room_config_schema_invalid_structure
    assert "format" in error_types
E   AssertionError: assert 'format' in set()
_________ TestJSONSchemaValidator.test_validation_context_strict_mode __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:478: in test_validation_context_strict_mode
    assert len(additional_prop_errors) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_______ TestJSONSchemaValidator.test_validation_context_allow_additional _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:495: in test_validation_context_allow_additional
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='658447e8-39ac-458e-8641-de59b86035a0').is_valid
_____________ TestJSONSchemaValidator.test_register_custom_schema ______________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:525: in test_register_custom_schema
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='ff56fd2c-87fc-493c-88e7-a546ca4c4263').is_valid
__________ TestJSONSchemaValidator.test_custom_validator_integration ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:628: in test_custom_validator_integration
    assert len(custom_errors) == 1
E   assert 0 == 1
E    +  where 0 = len([])
___ TestDatabaseSchemaValidator.test_validate_database_schema_complete_setup ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:724: in test_validate_database_schema_complete_setup
    assert result.is_valid
E   AssertionError: assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.model_type', message='Missing required column: predictions.model_type', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the model_type column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.model_version', message='Missing required column: predictions.model_version', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the model_version column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.predicted_time', message='Missing required column: predictions.predicted_time', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the predicted_time column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.confidence', message='Missing required column: predictions.confidence', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the confidence column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.features_hash', message='Missin...message='Missing required column: room_states.confidence', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the confidence column to room_states table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='room_states.occupancy_state', message='Missing required column: room_states.occupancy_state', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the occupancy_state column to room_states table', context={}), ValidationError(rule_id='DB_SCH_009', field='index', value='room_states.idx_room_timestamp', message='Missing critical index: idx_room_timestamp on room_states', severity=<ErrorSeverity.MEDIUM: 'medium'>, suggestion='Create index idx_room_timestamp for better query performance', context={}), ValidationError(rule_id='DB_SCH_009', field='index', value='predictions.idx_room_predicted_time', message='Missing critical index: idx_room_predicted_time on predictions', severity=<ErrorSeverity.MEDIUM: 'medium'>, suggestion='Create index idx_room_predicted_time for better query performance', context={})], warnings=[], confidence_score=0.0, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='15f4a84c-3e7e-4bc9-985e-25f5f20ba117').is_valid
__________ TestAPISchemaValidator.test_validate_headers_format_errors __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:1038: in test_validate_headers_format_errors
    errors = api_schema_validator._validate_headers(invalid_headers)
src/data/validation/schema_validator.py:938: in _validate_headers
    if len(value) > 8192:  # HTTP header value limit
E   TypeError: object of type 'int' has no len()
_ TestEdgeCasesAndErrorHandling.test_database_validator_with_connection_issues _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:1348: in test_database_validator_with_connection_issues
    assert len(connection_errors) > 0
E   assert 0 > 0
E    +  where 0 = len([])
------------------------------ Captured log call -------------------------------
WARNING  src.data.validation.schema_validator:schema_validator.py:810 Could not validate TimescaleDB features: Connection lost
____ TestEdgeCasesAndErrorHandling.test_schema_validator_memory_efficiency _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:1456: in test_schema_validator_memory_efficiency
    assert all(result.is_valid for result in results)
E   assert False
E    +  where False = all(<generator object TestEdgeCasesAndErrorHandling.test_schema_validator_memory_efficiency.<locals>.<genexpr> at 0x7fef2891d540>)
_______ TestEdgeCasesAndErrorHandling.test_concurrent_schema_validation ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:1498: in test_concurrent_schema_validation
    assert all(result.is_valid for result in results)
E   assert False
E    +  where False = all(<generator object TestEdgeCasesAndErrorHandling.test_concurrent_schema_validation.<locals>.<genexpr> at 0x7fef2891db40>)
___ TestIntegrationScenarios.test_complete_sensor_event_validation_pipeline ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_schema_validator_comprehensive.py:1554: in test_complete_sensor_event_validation_pipeline
    assert (
E   AssertionError: Completely valid data: expected True, got False
E   assert False == True
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='a9ba47be-fe76-4df1-ad07-de000396c7f3').is_valid
_________ TestDatabaseConfigValidator.test_validate_pool_size_warnings _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:410: in test_validate_pool_size_warnings
    assert any(
E   assert False
E    +  where False = any(<generator object TestDatabaseConfigValidator.test_validate_pool_size_warnings.<locals>.<genexpr> at 0x7fec866f1700>)
___________ TestDatabaseConfigValidator.test_connection_test_success ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:447: in test_connection_test_success
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["❌ Connection failed: 'coroutine' object has no attribute 'fetchval'"], warnings=[], info=[]).is_valid
________ TestDatabaseConfigValidator.test_connection_test_auth_failure _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:469: in test_connection_test_auth_failure
    assert any(
E   assert False
E    +  where False = any(<generator object TestDatabaseConfigValidator.test_connection_test_auth_failure.<locals>.<genexpr> at 0x7fec856536b0>)
______ TestStatisticalPatternAnalyzer.test_statistical_anomaly_detection _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:629: in test_statistical_anomaly_detection
    assert "anomaly_count" in analysis
E   AssertionError: assert 'anomaly_count' in {'event_count': 11, 'mean_interval': 300.1, 'median_interval': 300.0, 'state_distribution': {'on': 1.0}, ...}
_____ TestDatabaseConfigValidator.test_connection_test_database_not_found ______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:493: in test_connection_test_database_not_found
    assert any("Database does not exist" in error for error in result.errors)
E   assert False
E    +  where False = any(<generator object TestDatabaseConfigValidator.test_connection_test_database_not_found.<locals>.<genexpr> at 0x7fec85650c70>)
_______ TestStatisticalPatternAnalyzer.test_sensor_malfunction_detection _______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:673: in test_sensor_malfunction_detection
    assert len(high_freq_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_________ TestJSONSchemaValidator.test_sensor_event_schema_validation __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:868: in test_sensor_event_schema_validation
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='68707c54-6b16-4b82-bd6c-b4b534f89f54').is_valid
___________ TestJSONSchemaValidator.test_invalid_sensor_event_schema ___________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:888: in test_invalid_sensor_event_schema
    assert len(required_errors) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_______ TestSystemRequirementsValidator.test_validate_sufficient_system ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:904: in test_validate_sufficient_system
    result = validator.validate(config)
src/core/config_validator.py:527: in validate
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
E   AttributeError: 'tuple' object has no attribute 'major'
_______________ TestJSONSchemaValidator.test_schema_registration _______________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:929: in test_schema_registration
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='e9924eb5-be19-40ac-8c7c-e0187d54162b').is_valid
_______ TestSystemRequirementsValidator.test_validate_old_python_version _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:917: in test_validate_old_python_version
    result = validator.validate(config)
src/core/config_validator.py:527: in validate
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
E   AttributeError: 'tuple' object has no attribute 'major'
___________ TestComprehensiveEventValidator.test_event_sanitization ____________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_validation.py:1118: in test_event_sanitization
    assert "DROP TABLE" not in sanitized["sensor_id"]
E   AssertionError: assert 'DROP TABLE' not in '&#x27;; DRO...LE users; --'
E     'DROP TABLE' is contained here:
E       &#x27;; DROP TABLE users; --
E     ?         ++++++++++
___ TestSystemRequirementsValidator.test_validate_old_python_version_warning ___
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:929: in test_validate_old_python_version_warning
    result = validator.validate(config)
src/core/config_validator.py:527: in validate
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
E   AttributeError: 'tuple' object has no attribute 'major'
_____ TestSystemRequirementsValidator.test_validate_low_disk_space_warning _____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:966: in test_validate_low_disk_space_warning
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['Low disk space (< 5 GB available)'], info=['Python version: 3.13.7', 'Available disk space: 3.0 GB', 'Total memory: 15.6 GB, Available: 14.2 GB']).is_valid
___ TestAdvancedCacheInvalidation.test_cache_invalidation_on_data_freshness ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:81: in test_cache_invalidation_on_data_freshness
    cache.put("fresh_key", fresh_features, max_age_seconds=300)  # 5 minutes
tests/unit/test_features/test_cache_invalidation_advanced.py:59: in put
    self._evict_oldest(0.5)  # Evict 50% of entries
tests/unit/test_features/test_cache_invalidation_advanced.py:66: in _evict_oldest
    while self.size() > target_size and self._cache:
E   AttributeError: 'MemoryAwareCache' object has no attribute 'size'
_______ TestConfigurationValidator.test_validate_configuration_all_valid _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:1049: in test_validate_configuration_all_valid
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', 'Very few sensors configured, predictions may be less accurate'], info=['Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room living_room: 3 sensor types, 3 entities', 'Rooms configured: 1', 'Total sensors: 3', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
_______ TestAdvancedCacheInvalidation.test_cascading_cache_invalidation ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:126: in test_cascading_cache_invalidation
    cache.put("base_features", {"sensor_value": 25.0})
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
__ TestAdvancedCacheInvalidation.test_selective_cache_invalidation_by_pattern __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:162: in test_selective_cache_invalidation_by_pattern
    cache.put(key, features)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
_ TestConfigurationValidator.test_validate_configuration_with_connection_tests _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:1100: in test_validate_configuration_with_connection_tests
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', 'Very few sensors configured, predictions may be less accurate'], info=['Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room living_room: 3 sensor types, 3 entities', 'Rooms configured: 1', 'Total sensors: 3', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', 'Testing external connections...', 'Connection successful', 'Connection successful', 'Connection successful', '💥 Configuration validation failed with 1 errors']).is_valid
___ TestAdvancedCacheInvalidation.test_cache_invalidation_on_config_changes ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:192: in test_cache_invalidation_on_config_changes
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
________ TestConfigurationValidator.test_validate_config_files_success _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:1156: in test_validate_config_files_success
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', "Room test_room missing essential sensor types: ['occupancy', 'door']", 'Very few sensors configured, predictions may be less accurate'], info=['Loaded configuration from: /tmp/tmpeki6adew/config.yaml', 'Loaded rooms configuration from: /tmp/tmpeki6adew/rooms.yaml', 'Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room test_room: 1 sensor types, 1 entities', 'Rooms configured: 1', 'Total sensors: 1', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
_ TestAdvancedCacheInvalidation.test_time_based_cache_invalidation_with_sliding_window _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:263: in test_time_based_cache_invalidation_with_sliding_window
    cache.put("recent_features", {"value": 1.0})
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
______ TestCacheMemoryEfficiency.test_memory_efficient_large_feature_sets ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:289: in test_memory_efficient_large_feature_sets
    cache.put(key, large_features)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
__ TestConfigurationValidator.test_validate_config_files_environment_specific __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_config_validator.py:1205: in test_validate_config_files_environment_specific
    assert result.is_valid is True
E   assert False is True
E    +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', "Room test_room missing essential sensor types: ['occupancy', 'door']", 'Very few sensors configured, predictions may be less accurate'], info=['Loaded configuration from: /tmp/tmph2ejcvs8/config.staging.yaml', 'Loaded rooms configuration from: /tmp/tmph2ejcvs8/rooms.yaml', 'Home Assistant URL: http://staging:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: staging-mqtt', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room test_room: 1 sensor types, 1 entities', 'Rooms configured: 1', 'Total sensors: 1', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
_________ TestCacheMemoryEfficiency.test_weak_reference_cache_cleanup __________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:339: in test_weak_reference_cache_cleanup
    cache.put("key1", features1)
tests/unit/test_features/test_cache_invalidation_advanced.py:317: in put
    self._refs[key] = weakref.ref(features, lambda ref: self._cleanup(key))
E   TypeError: cannot create weak reference to 'dict' object
__ TestCacheConcurrencyAndCoherence.test_cache_eviction_under_concurrent_load __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_cache_invalidation_advanced.py:637: in test_cache_eviction_under_concurrent_load
    assert cache.size() <= 20, "Cache should respect size limit under load"
E   AttributeError: 'FeatureCache' object has no attribute 'size'
__________ TestSecretsManager.test_get_or_create_key_creates_new_key ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_environment.py:174: in test_get_or_create_key_creates_new_key
    assert len(decoded_key) == 32  # Fernet keys are 32 bytes
E   AssertionError: assert 44 == 32
E    +  where 44 = len(b'68F-TlOEc9wu65W_ZVSei9YWn6OfsIhgBsgpolkAj8c=')
__________________ TestEnvironmentManager.test_inject_secrets __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_environment.py:488: in test_inject_secrets
    assert "user:test_db_pass@" in config["database"]["connection_string"]
E   AssertionError: assert 'user:test_db_pass@' in 'postgresql://user@localhost/db'
_ TestExceptionContextPreservation.test_exception_context_filtering_sensitive_data _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:196: in test_exception_context_filtering_sensitive_data
    assert filtered_context["user_data"] == "[FILTERED]"
E   AssertionError: assert {'name': 'John', 'ssn': '123-...'} == '[FILTERED]'
____ TestExceptionHierarchyValidation.test_all_exceptions_inherit_from_base ____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:228: in test_all_exceptions_inherit_from_base
    instance = exc_class("Test error")
E   TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
_ TestSystemLayerErrorPropagation.test_external_service_to_internal_error_propagation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:325: in test_external_service_to_internal_error_propagation
    feature_error = MissingFeatureError(
E   TypeError: MissingFeatureError.__init__() got an unexpected keyword argument 'cause'
_____ TestValidationFunctionEdgeCases.test_validate_room_id_comprehensive ______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:611: in test_validate_room_id_comprehensive
    validate_room_id(invalid_room_id)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
____ TestValidationFunctionEdgeCases.test_validate_entity_id_comprehensive _____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:660: in test_validate_entity_id_comprehensive
    validate_entity_id(invalid_entity_id)
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
______ TestExceptionLoggingIntegration.test_error_alerting_classification ______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:835: in test_error_alerting_classification
    assert (
E   AssertionError: Classification mismatch for ModelPredictionError: alert_level
E   assert 'info' == 'warning'
E     - warning
E     + info
_______ TestProductionErrorScenarios.test_memory_pressure_error_handling _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:863: in test_memory_pressure_error_handling
    assert system_error.resource_type == "memory"
E   AttributeError: 'ResourceExhaustionError' object has no attribute 'resource_type'
_________ TestProductionErrorScenarios.test_cascading_failure_scenario _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:904: in test_cascading_failure_scenario
    assert len(degraded_error.context["model_predictions"]) == 3
E   KeyError: 'model_predictions'
___ TestProductionErrorScenarios.test_data_corruption_detection_and_handling ___
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:982: in test_data_corruption_detection_and_handling
    assert "corruption" in str(error).lower() or "invalid" in str(error).lower()
E   assert ('corruption' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" or 'invalid' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}")
E    +  where "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fec85391c30>()
E    +    where <built-in method lower of str object at 0x7fec85391c30> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
E    +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
E    +  and   "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fec8527df30>()
E    +    where <built-in method lower of str object at 0x7fec8527df30> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
E    +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
____ TestProductionErrorScenarios.test_graceful_degradation_error_patterns _____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:1020: in test_graceful_degradation_error_patterns
    assert any(
E   assert False
E    +  where False = any(<generator object TestProductionErrorScenarios.test_graceful_degradation_error_patterns.<locals>.<genexpr> at 0x7fec851be880>)
_ TestHomeAssistantErrors.test_home_assistant_authentication_error_with_string_token _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:286: in test_home_assistant_authentication_error_with_string_token
    assert "very_long_token..." in error.context["token_hint"]
E   AssertionError: assert 'very_long_token...' in 'very_long_...'
_ TestHomeAssistantErrors.test_home_assistant_authentication_error_with_short_token _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:304: in test_home_assistant_authentication_error_with_short_token
    assert error.context["token_hint"] == "short_token"
E   AssertionError: assert 'short_toke...' == 'short_token'
E     - short_token
E     ?           ^
E     + short_toke...
E     ?           ^^^
___________________ TestSystemErrors.test_system_error_basic ___________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1040: in test_system_error_basic
    assert str(error) == "Something went wrong"
E   AssertionError: assert 'Something we... SYSTEM_ERROR' == 'Something went wrong'
E     - Something went wrong
E     + Something went wrong | Error Code: SYSTEM_ERROR
______________ TestSystemErrors.test_system_error_with_operation _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1049: in test_system_error_with_operation
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...ta_processing' == 'System error...ed to process'
E     - System error during data_processing: Failed to process
E     + System error during data_processing: Failed to process | Error Code: SYSTEM_ERROR | Context: operation=data_processing
_______ TestSystemErrors.test_system_error_with_component_and_operation ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1064: in test_system_error_with_component_and_operation
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...Invalid input' == 'System error...dation failed'
E     - System error in input_validator during user_input_validation: Validation failed
E     + System error in input_validator during user_input_validation: Validation failed | Error Code: SYSTEM_ERROR | Context: component=input_validator, operation=user_input_validation | Caused by: ValueError: Invalid input
_______________ TestSystemErrors.test_resource_exhaustion_error ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1074: in test_resource_exhaustion_error
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...=90.0, unit=%' == 'Resource exh...limit: 90.0%)'
E     - Resource exhaustion: memory at 95.5% (limit: 90.0%)
E     + System error in memory during resource_monitoring: Resource exhaustion: memory at 95.5% (limit: 90.0%) | Error Code: RESOURCE_EXHAUSTION_ERROR | Context: component=memory, operation=resource_monitoring, resource_type=memory, current_usage=95.5, limit=90.0, unit=%
____________ TestSystemErrors.test_service_unavailable_error_basic _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1090: in test_service_unavailable_error_basic
    assert error.context["operation"] == "service_access"
E   KeyError: 'operation'
______________ TestSystemErrors.test_maintenance_mode_error_basic ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1114: in test_maintenance_mode_error_basic
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...CE_MODE_ERROR' == 'System in maintenance mode'
E     - System in maintenance mode
E     + System error in system during maintenance: System in maintenance mode | Error Code: MAINTENANCE_MODE_ERROR
_________ TestSystemErrors.test_maintenance_mode_error_with_component __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1125: in test_maintenance_mode_error_with_component
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...nent=database' == 'System compo...ode: database'
E     - System component in maintenance mode: database
E     + System error in database during maintenance: System component in maintenance mode: database | Error Code: MAINTENANCE_MODE_ERROR | Context: component=database
__________ TestSystemErrors.test_maintenance_mode_error_with_end_time __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1133: in test_maintenance_mode_error_with_end_time
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...-15 14:00 UTC' == 'System compo...15 14:00 UTC)'
E     - System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC)
E     + System error in search_engine during maintenance: System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC) | Error Code: MAINTENANCE_MODE_ERROR | Context: component=search_engine, estimated_end_time=2024-01-15 14:00 UTC
______________ TestAPIErrors.test_rate_limit_exceeded_error_basic ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1184: in test_rate_limit_exceeded_error_basic
    assert str(error) == expected_msg
E   AssertionError: assert 'Rate limit e..._seconds=3600' == 'Rate limit e...sts per 3600s'
E     - Rate limit exceeded for api: 100 requests per 3600s
E     + Rate limit exceeded for api: 100 requests per 3600s | Error Code: RATE_LIMIT_EXCEEDED_ERROR | Context: service=api, limit=100, window_seconds=3600
_________ TestValidationFunctions.test_validate_room_id_invalid_empty __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1307: in test_validate_room_id_invalid_empty
    validate_room_id("")
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
__________ TestValidationFunctions.test_validate_room_id_invalid_none __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1316: in test_validate_room_id_invalid_none
    validate_room_id(None)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
__________ TestValidationFunctions.test_validate_room_id_invalid_type __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1324: in test_validate_room_id_invalid_type
    validate_room_id(123)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_______ TestValidationFunctions.test_validate_room_id_invalid_characters _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1345: in test_validate_room_id_invalid_characters
    validate_room_id(room_id)
src/core/exceptions.py:1278: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
________ TestValidationFunctions.test_validate_entity_id_invalid_empty _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1370: in test_validate_entity_id_invalid_empty
    validate_entity_id("")
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_________ TestValidationFunctions.test_validate_entity_id_invalid_none _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1378: in test_validate_entity_id_invalid_none
    validate_entity_id(None)
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_________ TestValidationFunctions.test_validate_entity_id_invalid_type _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1385: in test_validate_entity_id_invalid_type
    validate_entity_id(["sensor", "temperature"])
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
________ TestValidationFunctions.test_validate_entity_id_invalid_format ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1419: in test_validate_entity_id_invalid_format
    validate_entity_id(entity_id)
src/core/exceptions.py:1303: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_ TestExceptionInheritanceAndCompatibility.test_all_exceptions_are_exceptions __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1485: in test_all_exceptions_are_exceptions
    instance = exc_class("test message")
E   TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
___ TestExceptionInheritanceAndCompatibility.test_error_context_preservation ___
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_exceptions.py:1493: in test_error_context_preservation
    config_error = ConfigValidationError("Test error", context=context)
E   TypeError: ConfigValidationError.__init__() got an unexpected keyword argument 'context'
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_minimum_length_validation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:45: in test_jwt_secret_key_minimum_length_validation
    with pytest.raises(
E   Failed: DID NOT RAISE <class 'ValueError'>
----------------------------- Captured stdout call -----------------------------
Warning: Using default test JWT secret key in production environment
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_acceptable_lengths _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:67: in test_jwt_secret_key_acceptable_lengths
    config = JWTConfig()
<string>:13: in __init__
    ???
src/core/config.py:210: in __post_init__
    raise ValueError("JWT secret key must be at least 32 characters long")
E   ValueError: JWT secret key must be at least 32 characters long
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_missing_in_production _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:108: in test_jwt_secret_key_missing_in_production
    with pytest.raises(
E   Failed: DID NOT RAISE <class 'ValueError'>
----------------------------- Captured stdout call -----------------------------
Warning: Using default test JWT secret key in production environment
____ TestJWTEnvironmentHandling.test_jwt_test_environment_fallback_behavior ____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:191: in test_jwt_test_environment_fallback_behavior
    config = JWTConfig()
<string>:13: in __init__
    ???
src/core/config.py:205: in __post_init__
    raise ValueError(
E   ValueError: JWT is enabled but JWT_SECRET_KEY environment variable is not set
----------------------------- Captured stdout call -----------------------------
Warning: Using default test JWT secret key in test environment
Warning: Using default test JWT secret key in test environment
Warning: Using default test JWT secret key in test environment
Warning: Using default test JWT secret key in testing environment
Warning: Using default test JWT secret key in testing environment
_______________ TestDatabaseConfig.test_database_config_creation _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_data/test_database_comprehensive.py:49: in test_database_config_creation
    config = DatabaseConfig(
E   TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
_ TestTemporalFeatureExtractorDurationFeatures.test_extract_duration_features_basic _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal.py:435: in test_extract_duration_features_basic
    assert features["avg_off_duration"] == 600.0  # Average of 10-minute off periods
E   assert 1800.0 == 600.0
___ TestContextualFeatureExtractorComprehensive.test_multiple_doors_handling ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:624: in test_multiple_doors_handling
    assert 0.0 <= features["door_open_ratio"] <= 1.0
E   assert 1.25 <= 1.0
____ TestContextualFeatureExtractorComprehensive.test_single_room_handling _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:880: in test_single_room_handling
    assert (
E   assert 0.0 == 1.0
_ TestContextualFeatureExtractorComprehensive.test_multi_sensor_event_ratio_accuracy _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:1127: in test_multi_sensor_event_ratio_accuracy
    assert 0.4 < features["multi_sensor_event_ratio"] < 0.6
E   assert 0.6666666666666666 < 0.6
_ TestContextualFeatureExtractorComprehensive.test_sensor_correlation_empty_data _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:1153: in test_sensor_correlation_empty_data
    assert features_single["sensor_activation_correlation"] == 1.0  # 1 sensor
E   assert 0.0 == 1.0
_ TestContextualFeatureExtractorComprehensive.test_feature_extraction_error_handling _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:1164: in test_feature_extraction_error_handling
    with pytest.raises(FeatureExtractionError):
E   Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
__ TestContextualFeatureExtractorComprehensive.test_get_feature_names_method ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_contextual_comprehensive.py:1587: in test_get_feature_names_method
    assert len(feature_names) > 50  # Should have many features
E   AssertionError: assert 50 > 50
E    +  where 50 = len(['current_temperature', 'avg_temperature', 'temperature_trend', 'temperature_variance', 'temperature_change_rate', 'temperature_stability', ...])
_ TestTemporalCoverageGaps.test_temporal_extract_room_state_features_edge_cases _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:34: in test_temporal_extract_room_state_features_edge_cases
    assert "room_state_confidence_avg" in features_empty
E   AssertionError: assert 'room_state_confidence_avg' in {'avg_occupancy_confidence': 0.5, 'recent_occupancy_ratio': 0.5, 'state_stability': 0.5}
_ TestTemporalCoverageGaps.test_temporal_extract_historical_patterns_insufficient_data _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:66: in test_temporal_extract_historical_patterns_insufficient_data
    assert "historical_trend" in features
E   AssertionError: assert 'historical_trend' in {'activity_variance': 0.0, 'day_activity_rate': 1.0, 'hour_activity_rate': 1.0, 'overall_activity_rate': 1.0, ...}
_ TestTemporalCoverageGaps.test_temporal_extract_cyclical_features_edge_times __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:77: in test_temporal_extract_cyclical_features_edge_times
    assert "cyclical_hour_sin" in features_midnight
E   AssertionError: assert 'cyclical_hour_sin' in {'day_of_month_cos': -0.994869323391895, 'day_of_month_sin': 0.10116832198743272, 'day_of_week_cos': 1.0, 'day_of_week_sin': 0.0, ...}
_ TestContextualCoverageGaps.test_contextual_environmental_features_with_invalid_values _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:193: in test_contextual_environmental_features_with_invalid_values
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestContextualCoverageGaps.test_contextual_multi_room_correlation_single_room _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:213: in test_contextual_multi_room_correlation_single_room
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestEngineeringCoverageGaps.test_engineering_get_default_features_all_types __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:229: in test_engineering_get_default_features_all_types
    defaults = engine.get_default_features()
E   AttributeError: 'FeatureEngineeringEngine' object has no attribute 'get_default_features'
_ TestEngineeringCoverageGaps.test_engineering_validate_configuration_invalid __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:259: in test_engineering_validate_configuration_invalid
    result = engine.validate_configuration(invalid_config)
E   TypeError: FeatureEngineeringEngine.validate_configuration() takes 1 positional argument but 2 were given
_ TestEngineeringCoverageGaps.test_engineering_parallel_vs_sequential_extraction _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:272: in test_engineering_parallel_vs_sequential_extraction
    features_sequential = engine.extract_features(
E   TypeError: FeatureEngineeringEngine.extract_features() got an unexpected keyword argument 'parallel'
_______ TestStoreCoverageGaps.test_feature_cache_expired_record_cleanup ________
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:294: in test_feature_cache_expired_record_cleanup
    cache.put("expire_test", {"value": 1.0})
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
__ TestStoreCoverageGaps.test_feature_store_compute_training_data_edge_cases ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:308: in test_feature_store_compute_training_data_edge_cases
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
____ TestStoreCoverageGaps.test_feature_store_cache_key_generation_complex _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_coverage_validation.py:327: in test_feature_store_cache_key_generation_complex
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
___ TestFeatureEngineeringEngineComprehensive.test_create_feature_dataframe ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_engineering_comprehensive.py:889: in test_create_feature_dataframe
    assert "sequential_room_count" in df.columns
E   AssertionError: assert 'sequential_room_count' in Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138)
E    +  where Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138) =    temporal_time_since_last_event  ...  meta_data_quality_score\n0                             0.0  ...                      0.0\n1                             0.0  ...                      0.0\n\n[2 rows x 138 columns].columns
____ TestFeatureEngineeringEngineComprehensive.test_end_to_end_integration _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_engineering_comprehensive.py:1482: in test_end_to_end_integration
    assert len(df.columns) == len(features)
E   AssertionError: assert 138 == 157
E    +  where 138 = len(Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138))
E    +    where Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138) =    temporal_time_since_last_event  ...  meta_data_quality_score\n0                        -14400.0  ...                     0.05\n\n[1 rows x 138 columns].columns
E    +  and   157 = len({'contextual_active_rooms_count': 1, 'contextual_avg_door_open_duration': 0.0, 'contextual_avg_humidity': 50.0, 'contextual_avg_light': 500.0, ...})
_ TestComponentFailureRecovery.test_temporal_extractor_partial_method_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:48: in _extract_cyclical_features
    raise RuntimeError("Cyclical feature extraction failed")
E   RuntimeError: Cyclical feature extraction failed

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_error_recovery_fault_tolerance.py:72: in test_temporal_extractor_partial_method_failures
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:114: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140664671286816'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140664671286816'> | Caused by: RuntimeError: Cyclical feature extraction failed
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: Cyclical feature extraction failed
_ TestComponentFailureRecovery.test_sequential_extractor_classifier_failure_fallback _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_error_recovery_fault_tolerance.py:126: in test_sequential_extractor_classifier_failure_fallback
    features = extractor.extract_features(events, room_configs)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room room1 | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=room1 | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestComponentFailureRecovery.test_contextual_extractor_environmental_sensor_degradation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:184: in test_contextual_extractor_environmental_sensor_degradation
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestComponentFailureRecovery.test_feature_engineering_engine_extractor_orchestration_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:282: in test_feature_engineering_engine_extractor_orchestration_failures
    assert len(successes) > 5, "Should have some successful extractions"
E   AssertionError: Should have some successful extractions
E   assert 0 > 5
E    +  where 0 = len([])
_____ TestResourceExhaustionRecovery.test_cpu_exhaustion_timeout_handling ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:423: in test_cpu_exhaustion_timeout_handling
    features = extractor.extract_features(events, room_configs)
tests/unit/test_features/test_error_recovery_fault_tolerance.py:395: in extract_features
    return future.result(timeout=self.timeout_seconds)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/_base.py:449: in result
    return self.__get_result()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/_base.py:401: in __get_result
    raise self._exception
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/thread.py:59: in run
    result = self.fn(*self.args, **self.kwargs)
tests/unit/test_features/test_error_recovery_fault_tolerance.py:389: in safe_extraction
    return super().extract_features(events, room_configs, target_time)
E   RuntimeError: super(): no arguments
______ TestResourceExhaustionRecovery.test_disk_space_exhaustion_fallback ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:471: in test_disk_space_exhaustion_fallback
    store = DiskAwareFeatureStore()
tests/unit/test_features/test_error_recovery_fault_tolerance.py:439: in __init__
    super().__init__(db_manager=Mock())
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_ TestNetworkAndDatabaseFailureRecovery.test_database_connection_failure_with_exponential_backoff _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:545: in test_database_connection_failure_with_exponential_backoff
    store = ResilientFeatureStore()
tests/unit/test_features/test_error_recovery_fault_tolerance.py:505: in __init__
    super().__init__(db_manager=Mock())
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
___ TestNetworkAndDatabaseFailureRecovery.test_database_corruption_recovery ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_error_recovery_fault_tolerance.py:740: in test_database_corruption_recovery
    store = CorruptionResilientStore()
tests/unit/test_features/test_error_recovery_fault_tolerance.py:669: in __init__
    super().__init__(db_manager=Mock())
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_ TestMissingSensorDataScenarios.test_temporal_features_with_significant_data_gaps _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:114: in test_temporal_features_with_significant_data_gaps
    assert "time_since_last_change" in features
E   AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 3540.0, 'avg_transition_interval': 1006.1538461538462, ...}
_ TestMissingSensorDataScenarios.test_sequential_features_with_incomplete_room_sequences _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_missing_data_scenarios.py:155: in test_sequential_features_with_incomplete_room_sequences
    features = extractor.extract_features(all_events, room_configs, target_time)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room bedroom | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=bedroom | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestMissingSensorDataScenarios.test_contextual_features_with_missing_environmental_sensors _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:214: in test_contextual_features_with_missing_environmental_sensors
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestMissingSensorDataScenarios.test_feature_extraction_with_corrupted_timestamps _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/temporal.py:74: in extract_features
    sorted_events = sorted(events, key=lambda e: e.timestamp)
E   TypeError: '<' not supported between instances of 'str' and 'NoneType'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_missing_data_scenarios.py:265: in test_feature_extraction_with_corrupted_timestamps
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:114: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room office | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=office | Caused by: TypeError: '<' not supported between instances of 'str' and 'NoneType'
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: '<' not supported between instances of 'str' and 'NoneType'
_ TestMissingSensorDataScenarios.test_feature_extraction_with_malformed_attributes _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:310: in test_feature_extraction_with_malformed_attributes
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestMissingSensorDataScenarios.test_feature_store_with_database_connection_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:339: in test_feature_store_with_database_connection_failures
    store = FeatureStore(db_manager=FlakyDatabaseManager())
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_ TestMissingSensorDataScenarios.test_feature_engineering_with_partial_extractor_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:392: in test_feature_engineering_with_partial_extractor_failures
    assert "sequential_feature_1" in features, "Working extractor should contribute"
E   TypeError: argument of type 'coroutine' is not iterable
_ TestFeatureValidationEdgeCases.test_feature_extraction_with_single_sensor_type _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_missing_data_scenarios.py:418: in test_feature_extraction_with_single_sensor_type
    features = extractor.extract_features(events, room_configs)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room single_sensor_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=single_sensor_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestFeatureValidationEdgeCases.test_feature_extraction_with_rapid_state_changes _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:446: in test_feature_extraction_with_rapid_state_changes
    assert "transition_rate" in features
E   AssertionError: assert 'transition_rate' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 1.0, 'avg_transition_interval': 1.0, ...}
_ TestFeatureValidationEdgeCases.test_feature_extraction_with_extreme_temporal_ranges _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:476: in test_feature_extraction_with_extreme_temporal_ranges
    assert "time_since_last_change" in features
E   AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.2245359891353553, 'avg_off_duration': 1800.0, 'avg_on_duration': 604800.0, 'avg_transition_interval': 604800.0, ...}
__ TestFeatureValidationEdgeCases.test_feature_extraction_memory_constraints ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_missing_data_scenarios.py:552: in test_feature_extraction_memory_constraints
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestFeatureValidationEdgeCases.test_feature_extraction_with_duplicate_events _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_missing_data_scenarios.py:600: in test_feature_extraction_with_duplicate_events
    features = extractor.extract_features(events, room_configs)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room duplicate_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=duplicate_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestTemporalFeatureExtractorComprehensive.test_time_since_calculation_accuracy _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:344: in test_time_since_calculation_accuracy
    assert (
E   assert 2700.0 < 1.0
E    +  where 2700.0 = abs((3600.0 - 900.0))
_ TestTemporalFeatureExtractorComprehensive.test_duration_mathematical_accuracy _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:540: in test_duration_mathematical_accuracy
    assert abs(features["avg_off_duration"] - expected_avg_off) < 1.0
E   assert 1200.0 < 1.0
E    +  where 1200.0 = abs((1800.0 - 600.0))
_ TestTemporalFeatureExtractorComprehensive.test_duration_statistical_features_numpy _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:579: in test_duration_statistical_features_numpy
    assert abs(features["median_off_duration"] - expected_median_off) < 1.0
E   assert 600.0 < 1.0
E    +  where 600.0 = abs((1800.0 - 1200.0))
__ TestTemporalFeatureExtractorComprehensive.test_duration_ratio_calculation ___
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:622: in test_duration_ratio_calculation
    assert abs(features["duration_ratio"] - expected_ratio) < 0.1
E   assert 1.6666666666666665 < 0.1
E    +  where 1.6666666666666665 = abs((0.8333333333333334 - 2.5))
_ TestTemporalFeatureExtractorComprehensive.test_duration_percentile_calculations _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:659: in test_duration_percentile_calculations
    assert (
E   assert 150.0 < 60.0
E    +  where 150.0 = abs((750.0 - 600.0))
_ TestTemporalFeatureExtractorComprehensive.test_transition_timing_comprehensive _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:1442: in test_transition_timing_comprehensive
    assert features[feature] >= 0.0
E   assert -43.51648351648352 >= 0.0
_ TestTemporalFeatureExtractorComprehensive.test_transition_timing_edge_cases __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:1643: in test_transition_timing_edge_cases
    assert single_features[key] == expected
E   KeyError: 'transition_regularity'
_ TestTemporalFeatureExtractorComprehensive.test_error_handling_comprehensive __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal_comprehensive.py:1890: in test_error_handling_comprehensive
    with pytest.raises(FeatureExtractionError):
E   Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
_ TestDaylightSavingTimeTransitions.test_temporal_features_spring_forward_transition _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_timezone_dst_handling.py:129: in test_temporal_features_spring_forward_transition
    assert "time_since_last_change" in features
E   AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.24987654320987648, 'avg_off_duration': 1800.0, 'avg_on_duration': 337.77777777777777, 'avg_transition_interval': 80.11173184357541, ...}
_ TestDaylightSavingTimeTransitions.test_temporal_features_fall_back_transition _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_timezone_dst_handling.py:163: in test_temporal_features_fall_back_transition
    assert "duration_in_current_state" in features
E   AssertionError: assert 'duration_in_current_state' in {'activity_variance': 0.23040000000000002, 'avg_off_duration': 1800.0, 'avg_on_duration': 47.64705882352941, 'avg_transition_interval': 35.83892617449664, ...}
_ TestDaylightSavingTimeTransitions.test_sequential_features_across_dst_boundaries _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_timezone_dst_handling.py:203: in test_sequential_features_across_dst_boundaries
    features = extractor.extract_features(all_events, room_configs, target_time)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room living_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestDaylightSavingTimeTransitions.test_contextual_features_dst_environmental_correlation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_timezone_dst_handling.py:272: in test_contextual_features_dst_environmental_correlation
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestCrossTimezoneScenarios.test_temporal_features_cross_timezone_correlation _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_timezone_dst_handling.py:346: in test_temporal_features_cross_timezone_correlation
    assert "cyclical_hour_sin" in features
E   AssertionError: assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 360.0, 'avg_transition_interval': 652.7272727272727, ...}
_ TestCrossTimezoneScenarios.test_sequential_features_global_room_transitions __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_timezone_dst_handling.py:369: in test_sequential_features_global_room_transitions
    features = extractor.extract_features(
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room pst_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=pst_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestTimezoneEdgeCasesAndErrorHandling.test_mixed_timezone_aware_naive_events _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/temporal.py:74: in extract_features
    sorted_events = sorted(events, key=lambda e: e.timestamp)
E   TypeError: can't compare offset-naive and offset-aware datetimes

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_timezone_dst_handling.py:599: in test_mixed_timezone_aware_naive_events
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:114: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room mixed_tz_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=mixed_tz_room | Caused by: TypeError: can't compare offset-naive and offset-aware datetimes
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: can't compare offset-naive and offset-aware datetimes
_______ TestEventProcessor.test_check_room_state_change_presence_sensor ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:990: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_event_processor.py:1219: in test_check_room_state_change_presence_sensor
    mock_tracking_manager.handle_room_state_change.assert_called_once_with(
E   AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
________ TestEventProcessor.test_check_room_state_change_motion_sensor _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:990: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_event_processor.py:1293: in test_check_room_state_change_motion_sensor
    mock_tracking_manager.handle_room_state_change.assert_called_once_with(
E   AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
_ TestHomeAssistantClientIntegration.test_get_bulk_history_with_rate_limit_handling _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_ingestion/test_ha_client.py:1364: in mock_get_history
    raise rate_error
E   src.core.exceptions.RateLimitExceededError: Rate limit exceeded for test: 100 requests per 60s | Error Code: RATE_LIMIT_EXCEEDED_ERROR | Context: service=test, limit=100, window_seconds=60, reset_time=1

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_ha_client.py:1378: in test_get_bulk_history_with_rate_limit_handling
    async for batch in client.get_bulk_history(
src/data/ingestion/ha_client.py:656: in get_bulk_history
    f"Rate limited during bulk history fetch, waiting {e.reset_time}s"
E   AttributeError: 'RateLimitExceededError' object has no attribute 'reset_time'
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
_ TestHomeAssistantClientIntegration.test_validate_and_normalize_state_edge_cases _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_ingestion/test_ha_client.py:1464: in test_validate_and_normalize_state_edge_cases
    assert client._validate_and_normalize_state("motion_clear") == "off"
E   AssertionError: assert 'on' == 'off'
E     - off
E     + on
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
_ TestHomeAssistantClientIntegration.test_handle_websocket_messages_json_error _
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:970: in assert_called_with
    raise AssertionError(error_message)
E   AssertionError: expected call not found.
E   Expected: warning('Received invalid JSON: invalid json')
E     Actual: not called.

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_ha_client.py:1644: in test_handle_websocket_messages_json_error
    mock_logger.warning.assert_called_with(
E   AssertionError: expected call not found.
E   Expected: warning('Received invalid JSON: invalid json')
E     Actual: not called.
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
__ TestHomeAssistantClientIntegration.test_reconnect_with_exponential_backoff __
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:979: in assert_called_with
    raise AssertionError(_error_message()) from cause
E   AssertionError: expected call not found.
E   Expected: sleep(4)
E     Actual: sleep(8)

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_ha_client.py:1700: in test_reconnect_with_exponential_backoff
    mock_sleep.assert_called_with(expected_delay)
E   AssertionError: expected call not found.
E   Expected: sleep(4)
E     Actual: sleep(8)
E   
E   pytest introspection follows:
E   
E   Args:
E   assert (8,) == (4,)
E     At index 0 diff: 8 != 4
E     Full diff:
E     - (4,)
E     ?  ^
E     + (8,)
E     ?  ^
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
________ TestHomeAssistantClientIntegration.test_reconnect_delay_capped ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:970: in assert_called_with
    raise AssertionError(error_message)
E   AssertionError: expected call not found.
E   Expected: sleep(300)
E     Actual: not called.

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_ha_client.py:1718: in test_reconnect_delay_capped
    mock_sleep.assert_called_with(300)
E   AssertionError: expected call not found.
E   Expected: sleep(300)
E     Actual: not called.
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
------------------------------ Captured log call -------------------------------
ERROR    src.data.ingestion.ha_client:ha_client.py:429 Max reconnection attempts reached
___ TestHomeAssistantClientIntegration.test_reconnect_failure_triggers_retry ___
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:958: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'create_task' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/test_ingestion/test_ha_client.py:1737: in test_reconnect_failure_triggers_retry
    mock_create_task.assert_called_once()
E   AssertionError: Expected 'create_task' to have been called once. Called 0 times.
---------------------------- Captured stdout setup -----------------------------
Warning: Using default test JWT secret key in  environment
_____________________ TestAPIEndpoints.test_root_endpoint ______________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:515: in test_root_endpoint
    assert data["status"] == "running"
E   KeyError: 'status'
_______________ TestAPIEndpoints.test_health_component_endpoint ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:553: in test_health_component_endpoint
    assert response.status_code == 200
E   assert 400 == 200
E    +  where 400 = <Response [400 Bad Request]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:11,214 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/health/components/database - ID: 1cb582a3-35bd-4a23-9841-6937b0b41609
2025-08-23 07:58:11,214 [INFO] src.integration.auth.middleware: dispatch:435 - Request started: GET /health/components/database
Error: -23 07:58:11,217 [ERROR] src.integration.api_server: api_error_handler:663 - API Error: Component with ID 'database' not found
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/routing.py", line 75, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 302, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 213, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/api_server.py", line 1098, in get_component_health
    raise APIResourceNotFoundError("Component", component_name)
src.core.exceptions.APIResourceNotFoundError: Component with ID 'database' not found | Error Code: API_RESOURCE_NOT_FOUND | Context: resource_type=Component, resource_id=database
2025-08-23 07:58:11,220 [INFO] src.integration.auth.middleware: dispatch:462 - Request completed: 400
2025-08-23 07:58:11,222 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/health/components/database "HTTP/1.1 400 Bad Request"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/health/components/database - ID: 1cb582a3-35bd-4a23-9841-6937b0b41609
INFO     src.integration.auth.middleware:middleware.py:435 Request started: GET /health/components/database
ERROR    src.integration.api_server:api_server.py:663 API Error: Component with ID 'database' not found
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/routing.py", line 75, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 302, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 213, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/api_server.py", line 1098, in get_component_health
    raise APIResourceNotFoundError("Component", component_name)
src.core.exceptions.APIResourceNotFoundError: Component with ID 'database' not found | Error Code: API_RESOURCE_NOT_FOUND | Context: resource_type=Component, resource_id=database
INFO     src.integration.auth.middleware:middleware.py:462 Request completed: 400
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/health/components/database "HTTP/1.1 400 Bad Request"
__________ TestAPIEndpoints.test_health_component_endpoint_not_found ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:564: in test_health_component_endpoint_not_found
    assert response.status_code == 404
E   assert 400 == 404
E    +  where 400 = <Response [400 Bad Request]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:11,265 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/health/components/nonexistent - ID: acc4b0a3-391f-427a-af78-5e9b6e049ee2
2025-08-23 07:58:11,266 [INFO] src.integration.auth.middleware: dispatch:435 - Request started: GET /health/components/nonexistent
Error: -23 07:58:11,267 [ERROR] src.integration.api_server: api_error_handler:663 - API Error: Component with ID 'nonexistent' not found
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/routing.py", line 75, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 302, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 213, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/api_server.py", line 1098, in get_component_health
    raise APIResourceNotFoundError("Component", component_name)
src.core.exceptions.APIResourceNotFoundError: Component with ID 'nonexistent' not found | Error Code: API_RESOURCE_NOT_FOUND | Context: resource_type=Component, resource_id=nonexistent
2025-08-23 07:58:11,270 [INFO] src.integration.auth.middleware: dispatch:462 - Request completed: 400
2025-08-23 07:58:11,272 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/health/components/nonexistent "HTTP/1.1 400 Bad Request"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/health/components/nonexistent - ID: acc4b0a3-391f-427a-af78-5e9b6e049ee2
INFO     src.integration.auth.middleware:middleware.py:435 Request started: GET /health/components/nonexistent
ERROR    src.integration.api_server:api_server.py:663 API Error: Component with ID 'nonexistent' not found
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/starlette/routing.py", line 75, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 302, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/fastapi/routing.py", line 213, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/api_server.py", line 1098, in get_component_health
    raise APIResourceNotFoundError("Component", component_name)
src.core.exceptions.APIResourceNotFoundError: Component with ID 'nonexistent' not found | Error Code: API_RESOURCE_NOT_FOUND | Context: resource_type=Component, resource_id=nonexistent
INFO     src.integration.auth.middleware:middleware.py:462 Request completed: 400
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/health/components/nonexistent "HTTP/1.1 400 Bad Request"
__________________ TestAPIEndpoints.test_get_predictions_room __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:607: in test_get_predictions_room
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:11,931 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/living_room - ID: 415b60c5-03cf-410c-b44a-a39238b5059c
Warning: 3 07:58:11,932 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:11,933 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/living_room - ID: 415b60c5-03cf-410c-b44a-a39238b5059c
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
_____________ TestAPIEndpoints.test_get_predictions_room_not_found _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:619: in test_get_predictions_room_not_found
    assert response.status_code == 404
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:11,976 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/nonexistent_room - ID: 9d32a77f-e27f-400f-94ff-1fe3083eba6b
Warning: 3 07:58:11,977 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:11,979 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/nonexistent_room "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/nonexistent_room - ID: 9d32a77f-e27f-400f-94ff-1fe3083eba6b
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/nonexistent_room "HTTP/1.1 401 Unauthorized"
__________________ TestAPIEndpoints.test_get_predictions_all ___________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:625: in test_get_predictions_all
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,020 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions - ID: 0893f222-3ebc-4037-8ac2-465812f80274
Warning: 3 07:58:12,021 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,022 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions - ID: 0893f222-3ebc-4037-8ac2-465812f80274
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions "HTTP/1.1 401 Unauthorized"
__________________ TestAPIEndpoints.test_get_accuracy_metrics __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:634: in test_get_accuracy_metrics
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,064 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/accuracy - ID: e33d7833-a768-41b7-9b23-598f5992122c
Warning: 3 07:58:12,064 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,066 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/accuracy "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/accuracy - ID: e33d7833-a768-41b7-9b23-598f5992122c
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/accuracy "HTTP/1.1 401 Unauthorized"
____________ TestAPIEndpoints.test_get_accuracy_metrics_with_params ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:646: in test_get_accuracy_metrics_with_params
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,107 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/accuracy?room_id=living_room&hours=48 - ID: 2b5ae462-de2b-49d9-a0be-901a549b6a10
Warning: 3 07:58:12,108 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,109 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/accuracy?room_id=living_room&hours=48 "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/accuracy?room_id=living_room&hours=48 - ID: 2b5ae462-de2b-49d9-a0be-901a549b6a10
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/accuracy?room_id=living_room&hours=48 "HTTP/1.1 401 Unauthorized"
_________________ TestAPIEndpoints.test_trigger_manual_retrain _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:662: in test_trigger_manual_retrain
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,156 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: ffc53123-7c82-4f95-8b0c-4a60bdb77abd
Warning: 3 07:58:12,157 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,158 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: ffc53123-7c82-4f95-8b0c-4a60bdb77abd
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
_________________ TestAPIEndpoints.test_refresh_mqtt_discovery _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:671: in test_refresh_mqtt_discovery
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,199 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/mqtt/refresh - ID: 4a8a3973-8a57-476e-928a-916737a53b75
Warning: 3 07:58:12,200 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,202 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/mqtt/refresh "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/mqtt/refresh - ID: 4a8a3973-8a57-476e-928a-916737a53b75
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/mqtt/refresh "HTTP/1.1 401 Unauthorized"
____________________ TestAPIEndpoints.test_get_system_stats ____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:680: in test_get_system_stats
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,244 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/stats - ID: 5c9f1c4c-c8ae-44ce-a92c-00be6b6d2505
Warning: 3 07:58:12,244 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,246 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/stats "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/stats - ID: 5c9f1c4c-c8ae-44ce-a92c-00be6b6d2505
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/stats "HTTP/1.1 401 Unauthorized"
_______________ TestIncidentEndpoints.test_get_active_incidents ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:699: in test_get_active_incidents
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,288 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents - ID: 9f939be7-3aa9-42ae-95f0-e69a690d2ece
Warning: 3 07:58:12,288 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,290 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents - ID: 9f939be7-3aa9-42ae-95f0-e69a690d2ece
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents "HTTP/1.1 401 Unauthorized"
_______________ TestIncidentEndpoints.test_get_incident_details ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:709: in test_get_incident_details
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,332 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/incident_001 - ID: 056cec04-e231-4145-b937-bc5115086f3d
Warning: 3 07:58:12,332 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,334 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/incident_001 "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/incident_001 - ID: 056cec04-e231-4145-b937-bc5115086f3d
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/incident_001 "HTTP/1.1 401 Unauthorized"
__________ TestIncidentEndpoints.test_get_incident_details_not_found ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:721: in test_get_incident_details_not_found
    assert response.status_code == 404
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,375 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/nonexistent - ID: 7dde1f21-5122-486c-af61-ce31702824fc
Warning: 3 07:58:12,376 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,377 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/nonexistent "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/nonexistent - ID: 7dde1f21-5122-486c-af61-ce31702824fc
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/nonexistent "HTTP/1.1 401 Unauthorized"
_______________ TestIncidentEndpoints.test_get_incident_history ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:727: in test_get_incident_history
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,422 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/history?hours=24 - ID: 8ce1bef0-7107-40c7-b073-8fd4470f9001
Warning: 3 07:58:12,422 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,424 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/history?hours=24 "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/history?hours=24 - ID: 8ce1bef0-7107-40c7-b073-8fd4470f9001
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/history?hours=24 "HTTP/1.1 401 Unauthorized"
________ TestIncidentEndpoints.test_get_incident_history_invalid_hours _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:737: in test_get_incident_history_invalid_hours
    assert response.status_code == 400
E   assert 401 == 400
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,466 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/history?hours=200 - ID: 57496786-5023-49f9-8624-d99d33ed2af4
Warning: 3 07:58:12,466 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,468 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/history?hours=200 "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/history?hours=200 - ID: 57496786-5023-49f9-8624-d99d33ed2af4
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/history?hours=200 "HTTP/1.1 401 Unauthorized"
______________ TestIncidentEndpoints.test_get_incident_statistics ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:743: in test_get_incident_statistics
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,511 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/statistics - ID: 724e7a32-4da1-4978-81c9-1a6141c8ac85
Warning: 3 07:58:12,511 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,513 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/statistics "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/statistics - ID: 724e7a32-4da1-4978-81c9-1a6141c8ac85
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/statistics "HTTP/1.1 401 Unauthorized"
_______________ TestIncidentEndpoints.test_acknowledge_incident ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:754: in test_acknowledge_incident
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,554 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/incident_001/acknowledge?acknowledged_by=test_user - ID: 4f92c1fe-6e6d-46c9-8504-2a425703c012
Warning: 3 07:58:12,555 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,557 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/incident_001/acknowledge?acknowledged_by=test_user "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/incident_001/acknowledge?acknowledged_by=test_user - ID: 4f92c1fe-6e6d-46c9-8504-2a425703c012
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/incident_001/acknowledge?acknowledged_by=test_user "HTTP/1.1 401 Unauthorized"
__________ TestIncidentEndpoints.test_acknowledge_incident_not_found ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:766: in test_acknowledge_incident_not_found
    assert response.status_code == 404
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,598 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/nonexistent/acknowledge - ID: 56b76e8d-4c7e-40d5-8388-894faa3591db
Warning: 3 07:58:12,598 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,600 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/nonexistent/acknowledge "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/nonexistent/acknowledge - ID: 56b76e8d-4c7e-40d5-8388-894faa3591db
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/nonexistent/acknowledge "HTTP/1.1 401 Unauthorized"
_________________ TestIncidentEndpoints.test_resolve_incident __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:774: in test_resolve_incident
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,641 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/incident_001/resolve?resolution_notes=Resolved%20manually - ID: 863c80a7-a2ef-40d5-a8ae-5b4e52d1d148
Warning: 3 07:58:12,642 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,643 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/incident_001/resolve?resolution_notes=Resolved%20manually "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/incident_001/resolve?resolution_notes=Resolved%20manually - ID: 863c80a7-a2ef-40d5-a8ae-5b4e52d1d148
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/incident_001/resolve?resolution_notes=Resolved%20manually "HTTP/1.1 401 Unauthorized"
____________ TestIncidentEndpoints.test_resolve_incident_not_found _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:786: in test_resolve_incident_not_found
    assert response.status_code == 404
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,691 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/nonexistent/resolve - ID: 5dce60a2-f60f-44cf-9f4a-e5eb081ac27d
Warning: 3 07:58:12,692 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,693 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/nonexistent/resolve "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/nonexistent/resolve - ID: 5dce60a2-f60f-44cf-9f4a-e5eb081ac27d
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/nonexistent/resolve "HTTP/1.1 401 Unauthorized"
______________ TestIncidentEndpoints.test_start_incident_response ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:792: in test_start_incident_response
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,735 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/response/start - ID: 8ca74d44-1344-438d-996e-ace5ed7cf86c
Warning: 3 07:58:12,735 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,737 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/response/start "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/response/start - ID: 8ca74d44-1344-438d-996e-ace5ed7cf86c
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/response/start "HTTP/1.1 401 Unauthorized"
______________ TestIncidentEndpoints.test_stop_incident_response _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:801: in test_stop_incident_response
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,778 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/incidents/response/stop - ID: 1cd2f599-2545-4c9b-bdf3-33c59df08a05
Warning: 3 07:58:12,778 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,780 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/incidents/response/stop "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/incidents/response/stop - ID: 1cd2f599-2545-4c9b-bdf3-33c59df08a05
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/incidents/response/stop "HTTP/1.1 401 Unauthorized"
___________________ TestErrorHandling.test_api_error_handler ___________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:822: in test_api_error_handler
    assert response.status_code == 400
E   assert 401 == 400
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,822 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/nonexistent - ID: b8ad2811-2453-47c3-ab5c-e066b19701a8
Warning: 3 07:58:12,822 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,824 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/nonexistent "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/nonexistent - ID: b8ad2811-2453-47c3-ab5c-e066b19701a8
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/nonexistent "HTTP/1.1 401 Unauthorized"
_________________ TestErrorHandling.test_system_error_handler __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:837: in test_system_error_handler
    assert response.status_code == 500
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,865 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/living_room - ID: 17cc106e-a0c1-45e0-933b-16c17adaff51
Warning: 3 07:58:12,865 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,867 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/living_room - ID: 17cc106e-a0c1-45e0-933b-16c17adaff51
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
_______________ TestErrorHandling.test_general_exception_handler _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:851: in test_general_exception_handler
    assert response.status_code == 500
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:12,908 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/living_room - ID: 8a63fbf7-f918-4f1b-b294-2ef8698a0e35
Warning: 3 07:58:12,908 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:12,910 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/living_room - ID: 8a63fbf7-f918-4f1b-b294-2ef8698a0e35
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
_______________ TestErrorHandling.test_health_endpoint_exception _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:863: in test_health_endpoint_exception
    assert response.status_code == 500
E   assert 200 == 500
E    +  where 200 = <Response [200 OK]>.status_code
---------------------------- Captured stdout setup -----------------------------
Error: -23 07:58:13,116 [ERROR] asyncio: default_exception_handler:1879 - Task was destroyed but it is pending!
task: <Task pending name='Task-191' coro=<HomeAssistantClient._reconnect() running at /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/ha_client.py:441> wait_for=<Future pending cb=[Task.task_wakeup()]>>
------------------------------ Captured log setup ------------------------------
ERROR    asyncio:base_events.py:1879 Task was destroyed but it is pending!
task: <Task pending name='Task-191' coro=<HomeAssistantClient._reconnect() running at /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/ha_client.py:441> wait_for=<Future pending cb=[Task.task_wakeup()]>>
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,126 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/health - ID: 76cf0035-880f-49b3-9d8f-57ff26859efb
2025-08-23 07:58:13,127 [INFO] src.integration.auth.middleware: dispatch:435 - Request started: GET /health
Error: -23 07:58:13,127 [ERROR] src.data.storage.database: health_check:760 - Database health check failed: Database manager not initialized
2025-08-23 07:58:13,128 [INFO] src.integration.auth.middleware: dispatch:462 - Request completed: 200
2025-08-23 07:58:13,130 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/health "HTTP/1.1 200 OK"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/health - ID: 76cf0035-880f-49b3-9d8f-57ff26859efb
INFO     src.integration.auth.middleware:middleware.py:435 Request started: GET /health
ERROR    src.data.storage.database:database.py:760 Database health check failed: Database manager not initialized
INFO     src.integration.auth.middleware:middleware.py:462 Request completed: 200
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/health "HTTP/1.1 200 OK"
___________________ TestErrorHandling.test_validation_error ____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:874: in test_validation_error
    assert response.status_code == 422  # Validation error
E   assert 401 == 422
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,173 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: 794635e0-9b12-485f-9d7b-6659209a3d38
Warning: 3 07:58:13,173 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,175 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: 794635e0-9b12-485f-9d7b-6659209a3d38
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
__________ TestResponseModels.test_manual_retrain_request_validation ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:978: in test_manual_retrain_request_validation
    request = ManualRetrainRequest(**valid_data)
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for ManualRetrainRequest
E   room_id
E     Value error, Room 'living_room' not found in configuration [type=value_error, input_value='living_room', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
_____________ TestAPIPerformance.test_multiple_prediction_requests _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1153: in test_multiple_prediction_requests
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,405 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/living_room - ID: 29ecff00-e44d-46d0-82dd-c89a4e3c7ed7
Warning: 3 07:58:13,405 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,407 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/living_room - ID: 29ecff00-e44d-46d0-82dd-c89a4e3c7ed7
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 401 Unauthorized"
_________________ TestAPIPerformance.test_large_stats_response _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1170: in test_large_stats_response
    assert response.status_code == 200
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,450 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/stats - ID: bd04d8c8-2afc-48ef-8cb2-7f8620a6ec12
Warning: 3 07:58:13,450 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,452 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/stats "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/stats - ID: bd04d8c8-2afc-48ef-8cb2-7f8620a6ec12
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/stats "HTTP/1.1 401 Unauthorized"
_________________ TestAPIEdgeCases.test_malformed_json_request _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1202: in test_malformed_json_request
    assert response.status_code == 422  # Unprocessable Entity
E   assert 401 == 422
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,526 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: b1cb938a-71b9-4efe-bb4a-a6703f419cec
Warning: 3 07:58:13,526 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,528 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: b1cb938a-71b9-4efe-bb4a-a6703f419cec
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
________________ TestAPIEdgeCases.test_missing_required_fields _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1213: in test_missing_required_fields
    assert response.status_code == 422
E   assert 401 == 422
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,570 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: 3425d2ef-c068-416a-a6de-2c998201402a
Warning: 3 07:58:13,570 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,572 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: 3425d2ef-c068-416a-a6de-2c998201402a
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
________________ TestAPIEdgeCases.test_invalid_query_parameters ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1219: in test_invalid_query_parameters
    assert response.status_code == 422
E   assert 401 == 422
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,613 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/accuracy?hours=invalid - ID: 54b25910-c6af-4bc2-bf5f-a85814041dcb
Warning: 3 07:58:13,613 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,615 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/accuracy?hours=invalid "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/accuracy?hours=invalid - ID: 54b25910-c6af-4bc2-bf5f-a85814041dcb
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/accuracy?hours=invalid "HTTP/1.1 401 Unauthorized"
___________________ TestAPIEdgeCases.test_oversized_request ____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1229: in test_oversized_request
    assert response.status_code in [200, 413, 422]
E   assert 401 in [200, 413, 422]
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,658 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: ba51de9b-610b-4ef9-ae4b-7c7bbbe0e9e8
Warning: 3 07:58:13,659 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,661 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: ba51de9b-610b-4ef9-ae4b-7c7bbbe0e9e8
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
_____________ TestAPIEdgeCases.test_special_characters_in_room_id ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1237: in test_special_characters_in_room_id
    assert response.status_code == 404
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,702 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/room with spaces - ID: 9021f584-a5cc-48b6-91b9-7c50c5c732f8
Warning: 3 07:58:13,703 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,704 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/room%20with%20spaces "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/room with spaces - ID: 9021f584-a5cc-48b6-91b9-7c50c5c732f8
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/room%20with%20spaces "HTTP/1.1 401 Unauthorized"
____________________ TestAPIEdgeCases.test_long_incident_id ____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1245: in test_long_incident_id
    assert response.status_code in [404, 414]  # Not found or URI too long
E   assert 401 in [404, 414]
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,747 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/incidents/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx - ID: 2585835f-0849-41bd-a50d-a96faba51ade
Warning: 3 07:58:13,747 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,749 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/incidents/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/incidents/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx - ID: 2585835f-0849-41bd-a50d-a96faba51ade
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/incidents/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx "HTTP/1.1 401 Unauthorized"
______________ TestAPIEdgeCases.test_concurrent_retrain_requests _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1275: in test_concurrent_retrain_requests
    assert all(status in [200, 400, 500] for status in results)
E   assert False
E    +  where False = all(<generator object TestAPIEdgeCases.test_concurrent_retrain_requests.<locals>.<genexpr> at 0x7fec86295630>)
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,792 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: fda05402-d6dc-448b-b0c7-be1c33b4fdd2
2025-08-23 07:58:13,795 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: f4488674-17c6-4d95-9d6e-2239ff0600f5
Warning: 3 07:58:13,796 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
Warning: 3 07:58:13,796 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,797 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: 7b42310c-8bc4-437c-847e-bca4891d4668
2025-08-23 07:58:13,798 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: 2d7d5a79-8daf-48f9-be97-3880ecf1ef37
2025-08-23 07:58:13,803 [INFO] src.integration.api_server: request_middleware:753 - API Request: POST http://testserver/model/retrain - ID: 95f2b61c-8fb3-467a-acfd-15c95278f005
Warning: 3 07:58:13,804 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
Warning: 3 07:58:13,805 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
Warning: 3 07:58:13,805 [WARNING] src.integration.auth.middleware: dispatch:218 - Authentication failed: Missing Authorization header
2025-08-23 07:58:13,805 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
2025-08-23 07:58:13,806 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
2025-08-23 07:58:13,811 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
2025-08-23 07:58:13,812 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
2025-08-23 07:58:13,813 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: fda05402-d6dc-448b-b0c7-be1c33b4fdd2
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: f4488674-17c6-4d95-9d6e-2239ff0600f5
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: 7b42310c-8bc4-437c-847e-bca4891d4668
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: 2d7d5a79-8daf-48f9-be97-3880ecf1ef37
INFO     src.integration.api_server:api_server.py:753 API Request: POST http://testserver/model/retrain - ID: 95f2b61c-8fb3-467a-acfd-15c95278f005
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
WARNING  src.integration.auth.middleware:middleware.py:218 Authentication failed: Missing Authorization header
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/model/retrain "HTTP/1.1 401 Unauthorized"
______________ TestAPIEdgeCases.test_api_with_no_tracking_manager ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_api_server.py:1311: in test_api_with_no_tracking_manager
    assert response.status_code in [404, 500]
E   assert 429 in [404, 500]
E    +  where 429 = <Response [429 Too Many Requests]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:13,886 [INFO] src.integration.api_server: request_middleware:753 - API Request: GET http://testserver/predictions/living_room - ID: 74465e0b-3345-4cf2-a176-68d6b56c3361
2025-08-23 07:58:13,887 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 429 Too Many Requests"
------------------------------ Captured log call -------------------------------
INFO     src.integration.api_server:api_server.py:753 API Request: GET http://testserver/predictions/living_room - ID: 74465e0b-3345-4cf2-a176-68d6b56c3361
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/predictions/living_room "HTTP/1.1 429 Too Many Requests"
_____________ TestGetJWTManager.test_get_jwt_manager_jwt_disabled ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:70: in test_get_jwt_manager_jwt_disabled
    with pytest.raises(HTTPException) as excinfo:
E   Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>
__________ TestGetJWTManager.test_get_jwt_manager_singleton_behavior ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:958: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'JWTManager' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/test_integration/test_auth_dependencies.py:98: in test_get_jwt_manager_singleton_behavior
    mock_jwt_class.assert_called_once()  # Only called once
E   AssertionError: Expected 'JWTManager' to have been called once. Called 0 times.
____________ TestRequirePermission.test_require_permission_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:269: in test_require_permission_success
    result = await permission_checker()
src/integration/auth/dependencies.py:154: in permission_checker
    if not user.has_permission(permission):
E   AttributeError: 'Depends' object has no attribute 'has_permission'
__________ TestRequirePermission.test_require_permission_admin_bypass __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:281: in test_require_permission_admin_bypass
    result = await permission_checker()
src/integration/auth/dependencies.py:154: in permission_checker
    if not user.has_permission(permission):
E   AttributeError: 'Depends' object has no attribute 'has_permission'
____________ TestRequirePermission.test_require_permission_failure _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:294: in test_require_permission_failure
    await permission_checker()
src/integration/auth/dependencies.py:154: in permission_checker
    if not user.has_permission(permission):
E   AttributeError: 'Depends' object has no attribute 'has_permission'
__________________ TestRequireRole.test_require_role_success ___________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:317: in test_require_role_success
    result = await role_checker()
src/integration/auth/dependencies.py:188: in role_checker
    if not user.has_role(role):
E   AttributeError: 'Depends' object has no attribute 'has_role'
__________________ TestRequireRole.test_require_role_failure ___________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:330: in test_require_role_failure
    await role_checker()
src/integration/auth/dependencies.py:188: in role_checker
    if not user.has_role(role):
E   AttributeError: 'Depends' object has no attribute 'has_role'
_________________ TestRequireAdmin.test_require_admin_success __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:358: in test_require_admin_success
    result = await admin_checker()
src/integration/auth/dependencies.py:218: in admin_checker
    if not user.is_admin:
E   AttributeError: 'Depends' object has no attribute 'is_admin'
_________________ TestRequireAdmin.test_require_admin_failure __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:371: in test_require_admin_failure
    await admin_checker()
src/integration/auth/dependencies.py:218: in admin_checker
    if not user.is_admin:
E   AttributeError: 'Depends' object has no attribute 'is_admin'
_________ TestRequirePermissions.test_require_permissions_all_success __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:406: in test_require_permissions_all_success
    result = await permissions_checker()
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_________ TestRequirePermissions.test_require_permissions_all_failure __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:419: in test_require_permissions_all_failure
    await permissions_checker()
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_________ TestRequirePermissions.test_require_permissions_any_success __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:433: in test_require_permissions_any_success
    result = await permissions_checker()
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_________ TestRequirePermissions.test_require_permissions_any_failure __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:448: in test_require_permissions_any_failure
    await permissions_checker()
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_________ TestRequirePermissions.test_require_permissions_admin_bypass _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:462: in test_require_permissions_admin_bypass
    result = await permissions_checker()
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_________ TestGetRequestContext.test_get_request_context_without_user __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:618: in test_get_request_context_without_user
    assert context["request_id"] == "req_456"
E   AssertionError: assert None == 'req_456'
__________ TestDependenciesIntegration.test_dependency_chain_success ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:723: in test_dependency_chain_success
    result = await permission_checker()
src/integration/auth/dependencies.py:154: in permission_checker
    if not user.has_permission(permission):
E   AttributeError: 'Depends' object has no attribute 'has_permission'
___ TestDependenciesIntegration.test_dependency_chain_multiple_requirements ____
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:746: in test_dependency_chain_multiple_requirements
    assert await permission_checker() == mock_user
src/integration/auth/dependencies.py:253: in permissions_checker
    user_permissions = set(user.permissions)
E   AttributeError: 'Depends' object has no attribute 'permissions'
_______ TestDependenciesIntegration.test_dependency_failure_propagation ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:770: in test_dependency_failure_propagation
    await admin_checker()
src/integration/auth/dependencies.py:218: in admin_checker
    if not user.is_admin:
E   AttributeError: 'Depends' object has no attribute 'is_admin'
_____ TestDependenciesErrorHandling.test_jwt_manager_initialization_error ______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:786: in test_jwt_manager_initialization_error
    with pytest.raises(Exception, match="Config error"):
E   Failed: DID NOT RAISE <class 'Exception'>
_____________ TestDependenciesPerformance.test_jwt_manager_caching _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:856: in test_jwt_manager_caching
    assert call_count == 1  # Should only be constructed once
E   assert 0 == 1
______ TestDependenciesPerformance.test_permission_validation_efficiency _______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_dependencies.py:874: in test_permission_validation_efficiency
    result = await permission_checker()
src/integration/auth/dependencies.py:154: in permission_checker
    if not user.has_permission(permission):
E   AttributeError: 'Depends' object has no attribute 'has_permission'
________ TestAuthenticationRouter.test_auth_router_routes_registration _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:190: in test_auth_router_routes_registration
    assert expected_path in route_paths
E   AssertionError: assert '/users/{user_id}' in {'/auth/change-password', '/auth/login', '/auth/logout', '/auth/me', '/auth/refresh', '/auth/token/info', ...}
________________ TestAuthRouterApp.test_login_endpoint_success _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:274: in test_login_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:14,835 [INFO] src.integration.auth.endpoints: login:170 - Successful login: user=admin, ip=testclient
Error: -23 07:58:14,835 [ERROR] src.integration.auth.endpoints: login:191 - Login error: unsupported operand type(s) for *: 'Mock' and 'int'
2025-08-23 07:58:14,837 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/login "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
INFO     src.integration.auth.endpoints:endpoints.py:170 Successful login: user=admin, ip=testclient
ERROR    src.integration.auth.endpoints:endpoints.py:191 Login error: unsupported operand type(s) for *: 'Mock' and 'int'
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/login "HTTP/1.1 500 Internal Server Error"
____________ TestAuthRouterApp.test_login_endpoint_invalid_username ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:295: in test_login_endpoint_invalid_username
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
E   assert 422 == 401
E    +  where 422 = <Response [422 Unprocessable Entity]>.status_code
E    +  and   401 = status.HTTP_401_UNAUTHORIZED
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:14,907 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/login "HTTP/1.1 422 Unprocessable Entity"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/login "HTTP/1.1 422 Unprocessable Entity"
____________ TestAuthRouterApp.test_login_endpoint_invalid_password ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:309: in test_login_endpoint_invalid_password
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
E   assert 422 == 401
E    +  where 422 = <Response [422 Unprocessable Entity]>.status_code
E    +  and   401 = status.HTTP_401_UNAUTHORIZED
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:14,977 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/login "HTTP/1.1 422 Unprocessable Entity"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/login "HTTP/1.1 422 Unprocessable Entity"
______________ TestAuthRouterApp.test_login_endpoint_remember_me _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:349: in test_login_endpoint_remember_me
    assert response.status_code == status.HTTP_200_OK
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,082 [INFO] src.integration.auth.endpoints: login:170 - Successful login: user=admin, ip=testclient
Error: -23 07:58:15,082 [ERROR] src.integration.auth.endpoints: login:191 - Login error: unsupported operand type(s) for *: 'Mock' and 'int'
2025-08-23 07:58:15,083 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/login "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
INFO     src.integration.auth.endpoints:endpoints.py:170 Successful login: user=admin, ip=testclient
ERROR    src.integration.auth.endpoints:endpoints.py:191 Login error: unsupported operand type(s) for *: 'Mock' and 'int'
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/login "HTTP/1.1 500 Internal Server Error"
____________ TestAuthRouterApp.test_refresh_token_endpoint_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:383: in test_refresh_token_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:15,207 [ERROR] src.integration.auth.endpoints: refresh_token:226 - Token refresh error: cannot unpack non-iterable Mock object
2025-08-23 07:58:15,209 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/refresh "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:226 Token refresh error: cannot unpack non-iterable Mock object
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/refresh "HTTP/1.1 500 Internal Server Error"
_________ TestAuthRouterApp.test_refresh_token_endpoint_invalid_token __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:406: in test_refresh_token_endpoint_invalid_token
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
E   assert 500 == 401
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   401 = status.HTTP_401_UNAUTHORIZED
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:15,277 [ERROR] src.integration.auth.endpoints: refresh_token:226 - Token refresh error: cannot unpack non-iterable Mock object
2025-08-23 07:58:15,279 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/refresh "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:226 Token refresh error: cannot unpack non-iterable Mock object
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/refresh "HTTP/1.1 500 Internal Server Error"
________________ TestAuthRouterApp.test_logout_endpoint_success ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:432: in test_logout_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,384 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
_______________ TestAuthRouterApp.test_logout_endpoint_no_token ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:453: in test_logout_endpoint_no_token
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,454 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
__________________ TestAuthRouterApp.test_me_endpoint_success __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:472: in test_me_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,527 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/auth/me "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/auth/me "HTTP/1.1 401 Unauthorized"
___________ TestAuthRouterApp.test_change_password_endpoint_success ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:497: in test_change_password_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,596 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
________ TestAuthRouterApp.test_change_password_endpoint_wrong_current _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:518: in test_change_password_endpoint_wrong_current
    assert response.status_code == status.HTTP_400_BAD_REQUEST
E   assert 401 == 400
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   400 = status.HTTP_400_BAD_REQUEST
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,664 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
________ TestAuthRouterApp.test_change_password_endpoint_user_not_found ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:536: in test_change_password_endpoint_user_not_found
    assert response.status_code == status.HTTP_404_NOT_FOUND
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   404 = status.HTTP_404_NOT_FOUND
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:15,732 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
______________ TestAuthRouterApp.test_token_info_endpoint_success ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:549: in test_token_info_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:15,800 [ERROR] src.integration.auth.endpoints: get_token_info:365 - Token info error: argument of type 'Mock' is not iterable
2025-08-23 07:58:15,802 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/token/info "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:365 Token info error: argument of type 'Mock' is not iterable
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/token/info "HTTP/1.1 500 Internal Server Error"
___________ TestAuthRouterApp.test_token_info_endpoint_invalid_token ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:577: in test_token_info_endpoint_invalid_token
    assert response.status_code == status.HTTP_400_BAD_REQUEST
E   assert 500 == 400
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
E    +  and   400 = status.HTTP_400_BAD_REQUEST
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:15,927 [ERROR] src.integration.auth.endpoints: get_token_info:365 - Token info error: argument of type 'Mock' is not iterable
2025-08-23 07:58:15,929 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/token/info "HTTP/1.1 500 Internal Server Error"
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:365 Token info error: argument of type 'Mock' is not iterable
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/token/info "HTTP/1.1 500 Internal Server Error"
_____________________ TestLoginEndpoint.test_login_success _____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:621: in test_login_success
    ), patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_______________ TestLoginEndpoint.test_login_invalid_credentials _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:648: in test_login_invalid_credentials
    login_request = LoginRequest(username="testuser", ***)
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for LoginRequest
E   password
E     Value error, Password must contain at least 3 of: uppercase, lowercase, digit, special character [type=value_error, input_value='wrongpassword', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
__________________ TestLoginEndpoint.test_login_inactive_user __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:676: in test_login_inactive_user
    ), patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_________ TestLoginEndpoint.test_login_remember_me_extended_expiration _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:702: in test_login_remember_me_extended_expiration
    ), patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
__________________ TestLoginEndpoint.test_login_service_error __________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:726: in test_login_service_error
    ), patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
____________________ TestLogoutEndpoint.test_logout_success ____________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/integration/auth/endpoints.py:249: in logout
    if jwt_manager.revoke_token(logout_request.refresh_token):
E   AttributeError: 'Depends' object has no attribute 'revoke_token'

During handling of the above exception, another exception occurred:
tests/unit/test_integration/test_auth_endpoints.py:764: in test_logout_success
    response = await logout(logout_request, test_user)
src/integration/auth/endpoints.py:263: in logout
    raise HTTPException(
E   fastapi.exceptions.HTTPException: 500: Logout service unavailable
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:16,637 [ERROR] src.integration.auth.endpoints: logout:262 - Logout error: 'Depends' object has no attribute 'revoke_token'
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:262 Logout error: 'Depends' object has no attribute 'revoke_token'
_______________ TestLogoutEndpoint.test_logout_revoke_all_tokens _______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:782: in test_logout_revoke_all_tokens
    ), patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_______________ TestLogoutEndpoint.test_logout_no_refresh_token ________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:812: in test_logout_no_refresh_token
    assert response["message"] == "Successfully logged out"
E   AssertionError: assert 'Logout successful' == 'Successfully logged out'
E     - Successfully logged out
E     + Logout successful
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:16,852 [INFO] src.integration.auth.endpoints: logout:253 - User logged out: testuser
------------------------------ Captured log call -------------------------------
INFO     src.integration.auth.endpoints:endpoints.py:253 User logged out: testuser
________________ TestLogoutEndpoint.test_logout_revoke_failure _________________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/integration/auth/endpoints.py:249: in logout
    if jwt_manager.revoke_token(logout_request.refresh_token):
E   AttributeError: 'Depends' object has no attribute 'revoke_token'

During handling of the above exception, another exception occurred:
tests/unit/test_integration/test_auth_endpoints.py:830: in test_logout_revoke_failure
    response = await logout(logout_request, test_user)
src/integration/auth/endpoints.py:263: in logout
    raise HTTPException(
E   fastapi.exceptions.HTTPException: 500: Logout service unavailable
----------------------------- Captured stdout call -----------------------------
Error: -23 07:58:16,891 [ERROR] src.integration.auth.endpoints: logout:262 - Logout error: 'Depends' object has no attribute 'revoke_token'
------------------------------ Captured log call -------------------------------
ERROR    src.integration.auth.endpoints:endpoints.py:262 Logout error: 'Depends' object has no attribute 'revoke_token'
_____________ TestRefreshTokenEndpoint.test_refresh_token_success ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:853: in test_refresh_token_success
    refresh_request = RefreshRequest(refresh_token="valid_refresh_token")
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
E   refresh_token
E     Value error, Invalid JWT token format [type=value_error, input_value='valid_refresh_token', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
__________ TestRefreshTokenEndpoint.test_refresh_token_invalid_token ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:879: in test_refresh_token_invalid_token
    refresh_request = RefreshRequest(refresh_token="invalid_token")
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
E   refresh_token
E     Value error, Invalid JWT token format [type=value_error, input_value='invalid_token', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
__________ TestRefreshTokenEndpoint.test_refresh_token_service_error ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:894: in test_refresh_token_service_error
    refresh_request = RefreshRequest(refresh_token="token")
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
E   refresh_token
E     Value error, Invalid refresh token format [type=value_error, input_value='token', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
___________ TestPasswordChangeEndpoint.test_change_password_success ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:931: in test_change_password_success
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_______ TestPasswordChangeEndpoint.test_change_password_invalid_current ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:957: in test_change_password_invalid_current
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
________ TestPasswordChangeEndpoint.test_change_password_service_error _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:978: in test_change_password_service_error
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_____________ TestUserManagementEndpoints.test_create_user_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1056: in test_create_user_success
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_______ TestUserManagementEndpoints.test_create_user_duplicate_username ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1075: in test_create_user_duplicate_username
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_____________ TestUserManagementEndpoints.test_list_users_success ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1092: in test_list_users_success
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
__________ TestUserManagementEndpoints.test_list_users_with_inactive ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1107: in test_list_users_with_inactive
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_____________ TestUserManagementEndpoints.test_delete_user_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1123: in test_delete_user_success
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
____________ TestUserManagementEndpoints.test_delete_user_not_found ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1139: in test_delete_user_not_found
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
___________ TestUserManagementEndpoints.test_delete_self_prevention ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1152: in test_delete_self_prevention
    with patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_____________ TestAdminEndpoints.test_list_users_endpoint_success ______________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1197: in test_list_users_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,642 [INFO] httpx: _send_single_request:1025 - HTTP Request: GET http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
_____________ TestAdminEndpoints.test_create_user_endpoint_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1231: in test_create_user_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,697 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
_______ TestAdminEndpoints.test_create_user_endpoint_duplicate_username ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1269: in test_create_user_endpoint_duplicate_username
    assert response.status_code == status.HTTP_400_BAD_REQUEST
E   assert 401 == 400
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   400 = status.HTTP_400_BAD_REQUEST
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,752 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
________ TestAdminEndpoints.test_create_user_endpoint_validation_errors ________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1295: in test_create_user_endpoint_validation_errors
    assert response.status_code == 422  # Validation error
E   assert 401 == 422
E    +  where 401 = <Response [401 Unauthorized]>.status_code
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,809 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
_____________ TestAdminEndpoints.test_delete_user_endpoint_success _____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1323: in test_delete_user_endpoint_success
    assert response.status_code == status.HTTP_200_OK
E   assert 401 == 200
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   200 = status.HTTP_200_OK
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,864 [INFO] httpx: _send_single_request:1025 - HTTP Request: DELETE http://testserver/auth/users/test_delete_user "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: DELETE http://testserver/auth/users/test_delete_user "HTTP/1.1 401 Unauthorized"
__________ TestAdminEndpoints.test_delete_user_endpoint_self_deletion __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1350: in test_delete_user_endpoint_self_deletion
    assert response.status_code == status.HTTP_400_BAD_REQUEST
E   assert 401 == 400
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   400 = status.HTTP_400_BAD_REQUEST
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,919 [INFO] httpx: _send_single_request:1025 - HTTP Request: DELETE http://testserver/auth/users/admin "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: DELETE http://testserver/auth/users/admin "HTTP/1.1 401 Unauthorized"
_________ TestAdminEndpoints.test_delete_user_endpoint_user_not_found __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1366: in test_delete_user_endpoint_user_not_found
    assert response.status_code == status.HTTP_404_NOT_FOUND
E   assert 401 == 404
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   404 = status.HTTP_404_NOT_FOUND
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:18,973 [INFO] httpx: _send_single_request:1025 - HTTP Request: DELETE http://testserver/auth/users/non_existent_user "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: DELETE http://testserver/auth/users/non_existent_user "HTTP/1.1 401 Unauthorized"
__________ TestAuthEndpointIntegration.test_full_authentication_flow ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1385: in test_full_authentication_flow
    with patch("src.integration.auth.endpoints.get_jwt_manager") as mock_jwt, patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
__________ TestAuthEndpointErrorHandling.test_login_unexpected_error ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1425: in test_login_unexpected_error
    with patch("src.integration.auth.endpoints.get_jwt_manager") as mock_jwt, patch(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_________ TestAuthEndpointErrorHandling.test_refresh_unexpected_error __________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1441: in test_refresh_unexpected_error
    refresh_request = RefreshRequest(refresh_token="token")
E   pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
E   refresh_token
E     Value error, Invalid refresh token format [type=value_error, input_value='token', input_type=str]
E       For further information visit https://errors.pydantic.dev/2.8/v/value_error
_____ TestAuthEndpointErrorHandling.test_password_change_unexpected_error ______
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1463: in test_password_change_unexpected_error
    with patch("src.integration.auth.endpoints.get_user_service") as mock_service:
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1497: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1467: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
_________ TestEndpointErrorHandling.test_logout_endpoint_service_error _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1565: in test_logout_endpoint_service_error
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:19,805 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/logout "HTTP/1.1 401 Unauthorized"
_________ TestEndpointErrorHandling.test_change_password_service_error _________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1583: in test_change_password_service_error
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:19,860 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/change-password "HTTP/1.1 401 Unauthorized"
___________ TestEndpointErrorHandling.test_create_user_service_error ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1625: in test_create_user_service_error
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:19,937 [INFO] httpx: _send_single_request:1025 - HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/auth/users "HTTP/1.1 401 Unauthorized"
___________ TestEndpointErrorHandling.test_delete_user_service_error ___________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1648: in test_delete_user_service_error
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
E   assert 401 == 500
E    +  where 401 = <Response [401 Unauthorized]>.status_code
E    +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
----------------------------- Captured stdout call -----------------------------
2025-08-23 07:58:19,991 [INFO] httpx: _send_single_request:1025 - HTTP Request: DELETE http://testserver/auth/users/someuser "HTTP/1.1 401 Unauthorized"
------------------------------ Captured log call -------------------------------
INFO     httpx:_client.py:1025 HTTP Request: DELETE http://testserver/auth/users/someuser "HTTP/1.1 401 Unauthorized"
___________ TestAuthEndpointSecurity.test_sensitive_data_not_logged ____________
[gw1] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_integration/test_auth_endpoints.py:1674: in test_sensitive_data_not_logged
    assert "TestPass123!" not in request_str
E   assert 'TestPass123!' not in "username='t...ber_me=False"
E     'TestPass123!' is contained here:
E       username='testuser' password='TestPass123!' remember_me=False
E     ?                               ++++++++++++
_ TestFeaturePerformanceBenchmarks.test_feature_engineering_batch_performance __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:158: in test_feature_engineering_batch_performance
    results = engine.extract_batch_features(
E   TypeError: FeatureEngineeringEngine.extract_batch_features() got an unexpected keyword argument 'parallel'
__ TestFeaturePerformanceBenchmarks.test_memory_efficiency_with_cache_cleanup __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:193: in test_memory_efficiency_with_cache_cleanup
    cache.put(key, large_feature_dict)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
____ TestFeaturePerformanceBenchmarks.test_concurrent_feature_store_access _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:212: in test_concurrent_feature_store_access
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_____ TestFeaturePerformanceBenchmarks.test_feature_extraction_scalability _____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:257: in test_feature_extraction_scalability
    features = extractor.extract_features(events, target_time)
src/features/temporal.py:105: in extract_features
    generic_features = self._extract_generic_sensor_features(sorted_events)
src/features/temporal.py:285: in _extract_generic_sensor_features
    for key, value in event.attributes.items():
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1169: in __call__
    return self._mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1173: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1248: in _execute_mock_call
    return self.return_value
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:577: in __get_return_value
    ret = self._get_child_mock(
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1090: in _get_child_mock
    return klass(**kw)
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:458: in __new__
    new = type(cls.__name__, bases, {'__doc__': cls.__doc__})
E   Failed: Timeout (>30.0s) from pytest-timeout.
----------------------------- Captured stdout call -----------------------------
~~~~~~~~~~~~~~~~~~~~~ Stack of <unknown> (140665572816576) ~~~~~~~~~~~~~~~~~~~~~
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/execnet/gateway_base.py", line 411, in _perform_spawn
    reply.run()
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/execnet/gateway_base.py", line 341, in run
    self._result = func(*args, **kwargs)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/execnet/gateway_base.py", line 1160, in _thread_receiver
    msg = Message.from_io(io)
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/execnet/gateway_base.py", line 567, in from_io
    header = io.read(9)  # type 1, channel 4, payload 4
  File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/execnet/gateway_base.py", line 534, in read
    data = self._read(numbytes - len(buf))
____ TestFeatureMissingDataHandling.test_temporal_features_with_sensor_gaps ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:317: in test_temporal_features_with_sensor_gaps
    assert (
E   AssertionError: Should calculate time since last change
E   assert 'time_since_last_change' in {'activity_variance': 0.24360000000000004, 'avg_off_duration': 1800.0, 'avg_on_duration': 757.5903614457832, 'avg_transition_interval': 385.32663316582915, ...}
_ TestFeatureMissingDataHandling.test_sequential_features_with_incomplete_sequences _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/sequential.py:85: in extract_features
    cutoff_time = target_time - timedelta(hours=lookback_hours)
E   TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_performance.py:340: in test_sequential_features_with_incomplete_sequences
    features = extractor.extract_features(events, room_configs)
src/features/sequential.py:139: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room living_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
_ TestFeatureMissingDataHandling.test_contextual_features_with_missing_environmental_data _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:375: in test_contextual_features_with_missing_environmental_data
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestFeatureMissingDataHandling.test_feature_store_with_database_unavailable __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:391: in test_feature_store_with_database_unavailable
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_ TestFeatureMissingDataHandling.test_feature_extraction_with_corrupted_events _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
src/features/temporal.py:74: in extract_features
    sorted_events = sorted(events, key=lambda e: e.timestamp)
E   TypeError: '<' not supported between instances of 'NoneType' and 'datetime.datetime'

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_performance.py:430: in test_feature_extraction_with_corrupted_events
    features = extractor.extract_features(events, datetime(2024, 1, 7, 12, 0, 0))
src/features/temporal.py:114: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140664671959568'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140664671959568'> | Caused by: TypeError: '<' not supported between instances of 'NoneType' and 'datetime.datetime'
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: '<' not supported between instances of 'NoneType' and 'datetime.datetime'
_ TestFeatureTimezoneTransitions.test_temporal_features_across_dst_spring_forward _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:463: in test_temporal_features_across_dst_spring_forward
    assert "cyclical_hour_sin" in features, "Should include cyclical time features"
E   AssertionError: Should include cyclical time features
E   assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 300.0, 'avg_transition_interval': 60.0, ...}
___ TestFeatureTimezoneTransitions.test_feature_extraction_timezone_changes ____
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:518: in test_feature_extraction_timezone_changes
    assert "cyclical_hour_sin" in features, "Should recalculate cyclical features"
E   AssertionError: Should recalculate cyclical features
E   assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 180.0, 'avg_transition_interval': 60.0, ...}
_____ TestFeatureTimezoneTransitions.test_cross_timezone_room_correlation ______
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:544: in test_cross_timezone_room_correlation
    features = extractor.extract_features(events, room_states)
E   TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
_ TestFeatureCacheInvalidationScenarios.test_feature_cache_with_memory_pressure _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:568: in test_feature_cache_with_memory_pressure
    cache.put(key, features)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
_ TestFeatureCacheInvalidationScenarios.test_cache_invalidation_on_time_expiry _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:588: in test_cache_invalidation_on_time_expiry
    cache.put("old_key", old_features, max_age_seconds=1)
E   TypeError: FeatureCache.put() got an unexpected keyword argument 'max_age_seconds'
_ TestFeatureCacheInvalidationScenarios.test_concurrent_cache_access_and_invalidation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:621: in test_concurrent_cache_access_and_invalidation
    future.result()  # Wait for completion
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/_base.py:449: in result
    return self.__get_result()
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/_base.py:401: in __get_result
    raise self._exception
/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/concurrent/futures/thread.py:59: in run
    result = self.fn(*self.args, **self.kwargs)
tests/unit/test_features/test_performance.py:609: in cache_worker
    cache.put(key, features)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
___ TestFeatureCacheInvalidationScenarios.test_feature_store_cache_coherence ___
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:633: in test_feature_store_cache_coherence
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
__ TestFeatureCacheInvalidationScenarios.test_memory_leak_prevention_in_cache __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:665: in test_memory_leak_prevention_in_cache
    cache.put(key, large_feature_dict)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
_ TestFeatureErrorRecoveryAndResilience.test_temporal_feature_extraction_with_partial_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:693: in _extract_cyclical_features
    raise Exception("Cyclical feature extraction failed")
E   Exception: Cyclical feature extraction failed

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_performance.py:703: in test_temporal_feature_extraction_with_partial_failures
    features = extractor.extract_features(events, datetime(2024, 1, 7, 12, 0, 0))
src/features/temporal.py:114: in extract_features
    raise FeatureExtractionError(
E   src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140665156762800'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140665156762800'> | Caused by: Exception: Cyclical feature extraction failed
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: Cyclical feature extraction failed
_ TestFeatureErrorRecoveryAndResilience.test_feature_engineering_graceful_degradation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:746: in test_feature_engineering_graceful_degradation
    assert len(features) > 0, "Successful extractions should yield features"
E   TypeError: object of type 'coroutine' has no len()

During handling of the above exception, another exception occurred:
tests/unit/test_features/test_performance.py:749: in test_feature_engineering_graceful_degradation
    assert "temporarily unavailable" in str(
E   AssertionError: Should get expected error message
E   assert 'temporarily unavailable' in "object of type 'coroutine' has no len()"
E    +  where "object of type 'coroutine' has no len()" = str(TypeError("object of type 'coroutine' has no len()"))
_ TestFeatureErrorRecoveryAndResilience.test_feature_store_database_reconnection _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:789: in test_feature_store_database_reconnection
    store = FeatureStore(db_manager=mock_db)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
_ TestFeatureErrorRecoveryAndResilience.test_concurrent_feature_extraction_with_failures _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_performance.py:836: in test_concurrent_feature_extraction_with_failures
    assert failed_tasks == expected_failures, "Expected tasks should fail"
E   AssertionError: Expected tasks should fail
E   assert [8, 12, 4, 0, 16] == [0, 4, 8, 12, 16]
E     At index 0 diff: 8 != 0
E     Full diff:
E     - [0, 4, 8, 12, 16]
E     + [8, 12, 4, 0, 16]
_ TestSequentialFeatureExtractorComprehensive.test_room_dwell_time_calculation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_sequential_comprehensive.py:367: in test_room_dwell_time_calculation
    assert (
E   assert 225.0 < 60
E    +  where 225.0 = abs((675.0 - 900.0))
_ TestSequentialFeatureExtractorComprehensive.test_max_sequence_length_detection _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_sequential_comprehensive.py:403: in test_max_sequence_length_detection
    assert features["max_room_sequence_length"] == 4.0
E   assert 1 == 4.0
_ TestSequentialFeatureExtractorComprehensive.test_room_transition_edge_cases __
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_sequential_comprehensive.py:482: in test_room_transition_edge_cases
    assert features_same["max_room_sequence_length"] == len(same_room_events)
E   AssertionError: assert 1 == 5
E    +  where 5 = len([<Mock spec='SensorEvent' id='140664667802320'>, <Mock spec='SensorEvent' id='140664667813072'>, <Mock spec='SensorEvent' id='140664674032944'>, <Mock spec='SensorEvent' id='140664674038656'>, <Mock spec='SensorEvent' id='140664674041008'>])
_ TestSequentialFeatureExtractorComprehensive.test_velocity_acceleration_calculation _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_sequential_comprehensive.py:647: in test_velocity_acceleration_calculation
    assert features["velocity_acceleration"] > 0  # Some variation in acceleration
E   assert 0.0 > 0
_ TestSequentialFeatureExtractorComprehensive.test_error_handling_comprehensive _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_sequential_comprehensive.py:1964: in test_error_handling_comprehensive
    with pytest.raises(FeatureExtractionError):
E   Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
_ TestTemporalFeatureExtractorTimeSinceFeatures.test_extract_time_since_features_basic _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal.py:357: in test_extract_time_since_features_basic
    assert features["time_since_last_off"] == 1200.0  # 20 minutes to last "off"
E   assert 3600.0 == 1200.0
_ TestTemporalFeatureExtractorTimeSinceFeatures.test_extract_time_since_features_motion_sensor _
[gw0] linux -- Python 3.13.7 /opt/hostedtoolcache/Python/3.13.7/x64/bin/python
tests/unit/test_features/test_temporal.py:372: in test_extract_time_since_features_motion_sensor
    assert (
E   assert 1200.0 == 1800.0
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_fields.py:161
../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_fields.py:161
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_fields.py:161: UserWarning: Field "model_info" has conflict with protected namespace "model_".
  
  You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    warnings.warn(

../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:291
../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:291
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:291: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.8/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:341
../../../../../opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:341
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/_internal/_config.py:341: UserWarning: Valid config keys have changed in V2:
  * 'schema_extra' has been renamed to 'json_schema_extra'
    warnings.warn(message, UserWarning)

tests/unit/test_data/test_dialect_compatibility.py:25
tests/unit/test_data/test_dialect_compatibility.py:25
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_dialect_compatibility.py:25: MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0) (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
    Base = declarative_base()

tests/unit/test_data/test_dialect_compatibility.py:320
tests/unit/test_data/test_dialect_compatibility.py:320
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_dialect_compatibility.py:320: PytestUnknownMarkWarning: Unknown pytest.mark.database_models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.database_models

tests/unit/test_data/test_event_processor_comprehensive.py:2225
tests/unit/test_data/test_event_processor_comprehensive.py:2225
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_event_processor_comprehensive.py:2225: PytestUnknownMarkWarning: Unknown pytest.mark.performance - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.performance

tests/unit/test_data/test_models_advanced.py:1524
tests/unit/test_data/test_models_advanced.py:1524
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_data/test_models_advanced.py:1524: PytestUnknownMarkWarning: Unknown pytest.mark.database_models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.database_models

tests/unit/test_models/test_base_predictor.py:256
tests/unit/test_models/test_base_predictor.py:256
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:256: PytestCollectionWarning: cannot collect test class 'TestConcretePredictor' because it has a __init__ constructor (from: tests/unit/test_models/test_base_predictor.py)
    class TestConcretePredictor(BasePredictor):

tests/unit/test_models/test_ensemble_comprehensive.py:2489
tests/unit/test_models/test_ensemble_comprehensive.py:2489
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_ensemble_comprehensive.py:2489: PytestUnknownMarkWarning: Unknown pytest.mark.models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytestmark = pytest.mark.models

tests/unit/test_models/test_gp_predictor_comprehensive.py:1660
tests/unit/test_models/test_gp_predictor_comprehensive.py:1660
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_gp_predictor_comprehensive.py:1660: PytestUnknownMarkWarning: Unknown pytest.mark.models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytestmark = pytest.mark.models

tests/unit/test_models/test_hmm_predictor_comprehensive.py:1618
tests/unit/test_models/test_hmm_predictor_comprehensive.py:1618
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_hmm_predictor_comprehensive.py:1618: PytestUnknownMarkWarning: Unknown pytest.mark.models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytestmark = pytest.mark.models

tests/unit/test_models/test_lstm_predictor_comprehensive.py:1505
tests/unit/test_models/test_lstm_predictor_comprehensive.py:1505
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_lstm_predictor_comprehensive.py:1505: PytestUnknownMarkWarning: Unknown pytest.mark.models - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytestmark = pytest.mark.models

tests/unit/test_utils/test_time_utils.py:803
tests/unit/test_utils/test_time_utils.py:803
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_utils/test_time_utils.py:803: PytestUnknownMarkWarning: Unknown pytest.mark.performance - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.performance

tests/unit/test_adaptation_consolidated.py: 4 warnings
tests/unit/test_adaptation/test_tracker.py: 6 warnings
  <string>:24: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

tests/unit/test_adaptation/test_drift_detector_comprehensive.py: 25 warnings
tests/unit/test_adaptation/test_tracker.py: 15 warnings
tests/unit/test_features/test_cache_invalidation_advanced.py: 1 warning
tests/unit/test_core/test_exception_propagation_advanced.py: 3 warnings
tests/unit/test_integration/test_api_server.py: 11 warnings
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/_pytest/python.py:183: PytestUnhandledCoroutineWarning: async def functions are not natively supported and have been skipped.
  You need to install a suitable plugin for your async framework, for example:
    - anyio
    - pytest-asyncio
    - pytest-tornasync
    - pytest-trio
    - pytest-twisted
    warnings.warn(PytestUnhandledCoroutineWarning(msg.format(nodeid)))

tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_track_model_training_context
tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_track_model_training_default_type
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/adaptation/monitoring_enhanced_tracking.py:250: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    async with self.monitoring_integration.track_training_operation(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_empty_prediction_result_attributes
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/adaptation/monitoring_enhanced_tracking.py:62: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    async with self.monitoring_integration.track_prediction_operation(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_adaptation/test_retrainer.py::TestRetrainingNeedEvaluation::test_cooldown_period_enforcement
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1151: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    _safe_super(CallableMixin, self).__init__(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_adaptation/test_retrainer.py: 10 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_retrainer.py:223: FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead.
    "timestamp": pd.date_range(

tests/unit/test_adaptation/test_retrainer.py::TestDataManagement::test_data_validation_before_training
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_retrainer.py:1207: RuntimeWarning: coroutine 'AdaptiveRetrainer._validate_training_data' was never awaited
    is_valid = await adaptive_retrainer._validate_training_data(empty_X, empty_y)
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_analyze_sensor_behavior_normal_data
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_high_frequency
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_low_frequency
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_unstable_timing
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestRealTimeQualityMonitor::test_calculate_quality_metrics_perfect_data
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestEdgeCasesAndErrorHandling::test_memory_usage_with_large_datasets
tests/unit/test_data/test_pattern_detector_comprehensive.py::TestEdgeCasesAndErrorHandling::test_performance_with_high_frequency_data
tests/unit/test_data/test_validation.py::TestStatisticalPatternAnalyzer::test_sensor_behavior_analysis
tests/unit/test_data/test_validation.py::TestStatisticalPatternAnalyzer::test_sensor_malfunction_detection
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:579: UserWarning: scipy.stats.shapiro: Input data has range zero. The results may not be accurate.
    res = hypotest_fun_out(*samples, **kwds)

tests/unit/test_data/test_pattern_detector_comprehensive.py::TestEdgeCasesAndErrorHandling::test_performance_with_high_frequency_data
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:579: UserWarning: scipy.stats.shapiro: For N > 5000, computed p-value may not be accurate. Current N is 9999.
    res = hypotest_fun_out(*samples, **kwds)

tests/unit/test_data/test_schema_validator_comprehensive.py::TestDatabaseSchemaValidator::test_validate_database_schema_missing_tables
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/validation/schema_validator.py:774: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    timescale_installed = result.fetchone() is not None
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_data/test_schema_validator_comprehensive.py::TestDatabaseSchemaValidator::test_validate_database_schema_missing_tables
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/validation/schema_validator.py:795: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    is_hypertable = result.fetchone() is not None
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_core/test_config_validator.py::TestDatabaseConfigValidator::test_connection_test_success
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_core/test_config_validator.py:441: DeprecationWarning: There is no current event loop
    return asyncio.get_event_loop().run_until_complete(coro)

tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheConcurrencyAndCoherence::test_concurrent_cache_read_write_coherence
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/_pytest/threadexception.py:73: PytestUnhandledThreadExceptionWarning: Exception in thread Thread-73 (reader_thread)
  
  Traceback (most recent call last):
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 1043, in _bootstrap_inner
      self.run()
      ~~~~~~~~^^
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 994, in run
      self._target(*self._args, **self._kwargs)
      ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_cache_invalidation_advanced.py", line 515, in reader_thread
      features = cache.get(key)
  TypeError: FeatureCache.get() missing 3 required positional arguments: 'target_time', 'lookback_hours', and 'feature_types'
  
    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))

tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheConcurrencyAndCoherence::test_cache_invalidation_race_conditions
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/_pytest/threadexception.py:73: PytestUnhandledThreadExceptionWarning: Exception in thread Thread-79 (aggressive_reader)
  
  Traceback (most recent call last):
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 1043, in _bootstrap_inner
      self.run()
      ~~~~~~~~^^
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 994, in run
      self._target(*self._args, **self._kwargs)
      ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_cache_invalidation_advanced.py", line 562, in aggressive_reader
      features = cache.get(key)
  TypeError: FeatureCache.get() missing 3 required positional arguments: 'target_time', 'lookback_hours', and 'feature_types'
  
    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))

tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheConcurrencyAndCoherence::test_cache_eviction_under_concurrent_load
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/_pytest/threadexception.py:73: PytestUnhandledThreadExceptionWarning: Exception in thread Thread-86 (cache_accessor)
  
  Traceback (most recent call last):
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 1043, in _bootstrap_inner
      self.run()
      ~~~~~~~~^^
    File "/opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/threading.py", line 994, in run
      self._target(*self._args, **self._kwargs)
      ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_cache_invalidation_advanced.py", line 616, in cache_accessor
      features = cache.get(key)
  TypeError: FeatureCache.get() missing 3 required positional arguments: 'target_time', 'lookback_hours', and 'feature_types'
  
    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))

tests/unit/test_features/test_contextual.py: 3 warnings
tests/unit/test_features/test_temporal.py: 3 warnings
tests/unit/test_features/test_contextual_comprehensive.py: 4 warnings
tests/unit/test_features/test_engineering_comprehensive.py: 4 warnings
tests/unit/test_features/test_temporal_comprehensive.py: 2 warnings
tests/unit/test_features/test_sequential.py: 1 warning
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide
    c /= stddev[:, None]

tests/unit/test_features/test_contextual.py: 1 warning
tests/unit/test_features/test_temporal.py: 3 warnings
tests/unit/test_features/test_contextual_comprehensive.py: 1 warning
tests/unit/test_features/test_engineering_comprehensive.py: 4 warnings
tests/unit/test_features/test_temporal_comprehensive.py: 2 warnings
tests/unit/test_features/test_sequential.py: 1 warning
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/numpy/lib/function_base.py:2898: RuntimeWarning: invalid value encountered in divide
    c /= stddev[None, :]

tests/unit/test_data/test_database.py::TestDatabaseManager::test_create_engine_postgresql
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/coverage/collector.py:248: RuntimeWarning: coroutine 'DatabaseManager._health_check_loop' was never awaited
    def lock_data(self) -> None:
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_data/test_database.py::TestDatabaseCompatibilityLayer::test_create_database_specific_models
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/unittest/mock.py:1151: RuntimeWarning: coroutine 'DatabaseManager._health_check_loop' was never awaited
    _safe_super(CallableMixin, self).__init__(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_features/test_engineering_comprehensive.py::TestFeatureEngineeringEngineComprehensive::test_validate_configuration_initialization_errors
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/_pytest/unraisableexception.py:78: PytestUnraisableExceptionWarning: Exception ignored in: <function FeatureEngineeringEngine.__del__ at 0x7fef0e10b9c0>
  
  Traceback (most recent call last):
    File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/features/engineering.py", line 841, in __del__
      if self.executor:
         ^^^^^^^^^^^^^
  AttributeError: 'FeatureEngineeringEngine' object has no attribute 'executor'
  
    warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))

tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestComponentFailureRecovery::test_feature_engineering_engine_extractor_orchestration_failures
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_error_recovery_fault_tolerance.py:267: RuntimeWarning: coroutine 'FeatureEngineeringEngine.extract_features' was never awaited
    features = engine.extract_features(
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestResourceExhaustionRecovery::test_memory_exhaustion_graceful_degradation
tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_with_rapid_state_changes
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/sqlalchemy/orm/attributes.py:484: RuntimeWarning: coroutine 'FeatureEngineeringEngine.extract_features' was never awaited
    key,
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_ingestion/test_bulk_importer.py: 74 warnings
  <string>:8: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

tests/unit/test_ingestion/test_bulk_importer.py: 74 warnings
  <string>:9: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

tests/unit/test_ingestion/test_bulk_importer.py::TestImportProgress::test_import_progress_properties
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:68: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    progress.start_time = datetime.utcnow() - timedelta(seconds=60)

tests/unit/test_ingestion/test_bulk_importer.py: 53 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:56: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    return (datetime.utcnow() - self.start_time).total_seconds()

tests/unit/test_ingestion/test_bulk_importer.py::TestImportProgress::test_import_progress_edge_cases
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:89: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    progress.start_time = datetime.utcnow()

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_save_resume_data
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_save_resume_data_file_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:268: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp": datetime.utcnow().isoformat(),

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_process_entities_batch
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_process_entities_batch
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_process_entities_batch
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_process_entities_batch_with_completed
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_update_progress
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_update_progress_async_callback
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_update_progress_callback_error
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_update_progress_with_periodic_logging
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:636: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.progress.last_update = datetime.utcnow()

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_convert_ha_events_to_sensor_events
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:554: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    created_at=datetime.utcnow(),

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_optimize_import_performance
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:1011: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_verify_import_integrity
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_verify_import_integrity_empty_database
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_verify_import_integrity_perfect_score
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_verify_import_integrity_database_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:877: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "verification_timestamp": datetime.utcnow().isoformat(),

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_create_import_checkpoint
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_create_import_checkpoint_file_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:968: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp": datetime.utcnow().isoformat(),

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_create_import_checkpoint
tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_create_import_checkpoint_file_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:980: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    checkpoint_file = f"checkpoint_{checkpoint_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporter::test_import_historical_data_with_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:172: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    end_date = datetime.utcnow()

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_optimize_import_performance_missing_psutil
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:1541: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_optimize_import_performance_suggestions
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:1558: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(seconds=1)

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_optimize_import_performance_suggestions
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:1579: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer2.progress.start_time = datetime.utcnow() - timedelta(seconds=1)

tests/unit/test_ingestion/test_bulk_importer.py::TestBulkImporterIntegration::test_update_progress_with_periodic_logging
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_ingestion/test_bulk_importer.py:1918: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClient::test_get_entity_state_success
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/yaml/error.py:6: RuntimeWarning: coroutine 'HomeAssistantClient._handle_websocket_messages' was never awaited
    def __init__(self, name, index, line, column, buffer, pointer):
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_handle_websocket_messages_json_error
tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_handle_websocket_messages_connection_closed
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/ha_client.py:260: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    async for message in self.websocket:
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_health_component_endpoint
tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_health_component_endpoint_not_found
tests/unit/test_integration/test_api_server.py::TestResponseModels::test_error_response_serialization
tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_api_with_no_tracking_manager
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/main.py:1087: PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.8/migration/
    warnings.warn('The `dict` method is deprecated; use `model_dump` instead.', category=PydanticDeprecatedSince20)

tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_start_health_monitoring
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/utils/health_monitor.py:610: DeprecationWarning: Callback API version 1 is deprecated, update to latest version
    client = mqtt.Client()

tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_malformed_json_request
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/httpx/_models.py:408: DeprecationWarning: Use 'content=<...>' to upload raw bytes/text content.
    headers, stream = encode_request(

tests/unit/test_integration/test_auth_dependencies.py: 21 warnings
tests/unit/test_integration/test_auth_endpoints.py: 36 warnings
tests/unit/test_integration/test_auth_models.py: 10 warnings
  /opt/hostedtoolcache/Python/3.13.7/x64/lib/python3.13/site-packages/pydantic/main.py:193: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.__pydantic_validator__.validate_python(data, self_instance=self)

tests/unit/test_features/test_performance.py::TestFeatureErrorRecoveryAndResilience::test_feature_extraction_memory_exhaustion_recovery
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_features/test_performance.py:862: RuntimeWarning: coroutine 'FeatureEngineeringEngine.extract_features' was never awaited
    gc.collect()
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- generated xml file: /home/runner/work/ha-ml-predictor/ha-ml-predictor/junit-unit.xml -

---------- coverage: platform linux, python 3.13.7-final-0 -----------
Name                                              Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------------
src/__init__.py                                       0      0   100%
src/adaptation/__init__.py                            7      0   100%
src/adaptation/drift_detector.py                    544    152    72%   391, 436-438, 488, 495, 500-501, 552, 580-581, 597, 620-622, 725-728, 751, 757, 769-771, 815-816, 822-823, 827-829, 864-867, 873-876, 880-882, 904-926, 940-969, 1000, 1012-1014, 1024-1033, 1042-1075, 1087-1112, 1120-1137, 1146-1171, 1221-1222, 1272-1274, 1321, 1438-1440, 1453-1485, 1489-1495, 1519-1521, 1533
src/adaptation/monitoring_enhanced_tracking.py       95      4    96%   66-83
src/adaptation/optimizer.py                         567    268    53%   30-36, 99-100, 104-121, 127, 131-133, 137-143, 149-155, 169-180, 184, 230, 375-378, 382-383, 411-420, 442-443, 458-461, 477-479, 483-500, 537-575, 586-589, 595-605, 631-633, 644, 668-671, 704-707, 717-720, 732-734, 752, 772-773, 776-779, 793-799, 821-822, 864-866, 882-883, 885-886, 917, 941-943, 980-981, 997, 1021-1023, 1044-1086, 1231-1268, 1274-1302, 1317-1361, 1386-1387, 1391-1432, 1439
src/adaptation/retrainer.py                         936    349    63%   134, 379-380, 401-408, 413, 428, 449-451, 587-589, 604-605, 626-748, 820-828, 848-872, 881-882, 906, 915-917, 927-929, 933-935, 968-989, 1009, 1057-1059, 1074-1075, 1079-1090, 1105-1107, 1125, 1135, 1139, 1144-1146, 1161-1163, 1183, 1187, 1195-1206, 1222-1233, 1278-1280, 1291, 1293, 1297-1298, 1358-1403, 1409-1418, 1444-1446, 1526-1534, 1557, 1574-1576, 1578-1581, 1595-1598, 1602-1604, 1607-1610, 1627-1629, 1631-1634, 1658-1670, 1681-1683, 1691-1708, 1719-1723, 1762, 1786-1787, 1826-1827, 1847, 1850-1854, 1860-1879, 1890-1941, 1945-1963, 1967-1985, 2008-2009, 2067-2092, 2144, 2147-2151, 2193, 2196-2200, 2206-2220, 2258-2261, 2281, 2286-2288, 2307-2324
src/adaptation/tracker.py                           617    442    28%   138, 142, 264, 269-284, 288-292, 296-299, 305-316, 320, 436-454, 458-475, 492-533, 550-585, 598-608, 623-646, 667-712, 718-720, 726-728, 732-766, 772-798, 802-828, 832-874, 883-960, 964-975, 992-1030, 1063-1064, 1073-1093, 1103-1136, 1144-1153, 1159-1184, 1190-1227, 1231-1246, 1252-1411, 1415-1423, 1427-1463, 1467-1539, 1545-1556
src/adaptation/tracking_manager.py                  835    694    17%   22-26, 59-64, 208-269, 273-358, 362-399, 403-447, 459-578, 600-627, 632-708, 712-723, 731-741, 747-761, 765-775, 781-787, 793-799, 814-850, 862-897, 903-926, 930-981, 987-995, 999-1021, 1025-1063, 1067-1089, 1093-1154, 1158-1174, 1180-1295, 1299-1339, 1347-1384, 1388-1435, 1440-1446, 1467-1498, 1504-1512, 1516-1524, 1537-1544, 1554-1561, 1565-1574, 1583-1599, 1603-1609, 1613-1632, 1636-1711, 1715-1764, 1781-1807, 1822-1858, 1889-1937, 1953-2023, 2034-2070, 2075-2082, 2086-2152, 2158-2170, 2176-2188, 2192-2244, 2249-2256, 2260-2302, 2309-2310
src/adaptation/validator.py                         866    345    60%   109, 149, 280, 287, 298, 434-447, 451-463, 492-500, 555-556, 681, 717-720, 742-743, 760-764, 815-843, 859, 871-873, 877-885, 888, 915-950, 968, 1007-1009, 1035-1044, 1059, 1063-1065, 1079-1095, 1103-1105, 1121-1128, 1155, 1173-1182, 1225-1227, 1253-1258, 1281-1283, 1314-1316, 1341, 1375-1390, 1402-1404, 1428-1433, 1459-1460, 1468, 1502-1504, 1510-1522, 1571-1572, 1594, 1602, 1663, 1685, 1703-1711, 1738-1752, 1764-1874, 1882-1883, 1893-1897, 1905-1906, 1909, 1915-1930, 1951, 1953-1954, 1959-1960, 1964-1997, 2001-2123, 2133-2243, 2254-2308, 2314-2321
src/core/__init__.py                                  0      0   100%
src/core/backup_manager.py                          351     71    80%   104-109, 116-121, 193, 223-228, 235-242, 335-336, 376, 390, 501, 575-581, 590-591, 605-632, 635, 644-654, 662-667, 673, 677, 719-724
src/core/config.py                                  317     13    96%   40, 320, 355-357, 414-417, 572-577
src/core/config_validator.py                        426     55    87%   116, 121, 132-133, 170, 178-179, 218, 223, 250, 256-273, 276, 280, 282, 289, 407-412, 479, 531, 533, 559, 572-573, 588, 591-596, 642-643, 652-653, 658-659, 664-665, 678, 703, 728-729, 736-738
src/core/constants.py                                48      0   100%
src/core/environment.py                             291     20    93%   204-205, 352, 354, 429, 436, 447-448, 454, 496, 550-556, 578-581, 586-588
src/core/exceptions.py                              410      2    99%   117, 121
src/data/__init__.py                                  0      0   100%
src/data/ingestion/__init__.py                        0      0   100%
src/data/ingestion/bulk_importer.py                 419     19    95%   76, 425-428, 435, 460, 479, 575, 696-700, 794-796, 941, 1006
src/data/ingestion/event_processor.py               558     66    88%   234, 290, 297, 348, 373, 388, 465, 558, 567, 613, 633, 719-740, 757, 759, 912-919, 959, 969, 993, 1031-1032, 1034-1037, 1047-1064, 1068-1081, 1154-1159, 1182-1185, 1217-1218
src/data/ingestion/ha_client.py                     377     41    89%   261-267, 269-271, 282, 301, 348-349, 384-399, 451-453, 590, 609, 636, 658-667
src/data/storage/__init__.py                          3      0   100%
src/data/storage/database.py                        389     69    82%   133-135, 150, 155-173, 178, 183, 188-192, 269-276, 307-308, 311-312, 320-321, 368-370, 376-379, 449, 453-455, 499-500, 513-521, 528-529, 556, 576-577, 599, 603, 622-625, 731-733, 769, 772-775, 788-789
src/data/storage/database_compatibility.py           75      3    96%   139, 194-196
src/data/storage/dialect_utils.py                   136     17    88%   109, 254, 292-308, 324-328, 343-347
src/data/storage/models.py                          447     10    98%   54-57, 76, 422-444, 972, 1010
src/data/validation/__init__.py                       4      0   100%
src/data/validation/event_validator.py              300     32    89%   81, 86, 91, 175, 256, 349, 449, 495, 543, 575-576, 750-755, 765, 768-780, 866-868, 887-891, 905-915
src/data/validation/pattern_detector.py             336     36    89%   31-50, 130-131, 165, 184, 211-212, 223, 240, 257, 279, 517-518, 617, 648-649, 671, 703-704, 721, 748, 750, 756-757
src/data/validation/schema_validator.py             332     56    83%   32-44, 346-377, 413-445, 631-632, 760-761
src/features/__init__.py                              0      0   100%
src/features/contextual.py                          449     35    92%   177, 179, 181, 384, 676-680, 711-719, 737, 816, 859-861, 872-877, 879, 896, 898, 902, 904, 1037-1038, 1052-1054, 1088-1089
src/features/engineering.py                         303     15    95%   140, 393-394, 425-426, 593-596, 611-614, 704, 724, 745, 764, 831
src/features/sequential.py                          350     12    97%   209-210, 236-238, 464-467, 485, 760, 767, 788, 831
src/features/store.py                               220     63    71%   65-67, 75, 256-262, 314-325, 482-499, 508-551, 588-591, 599-600, 604-612, 616-649, 663-664, 669
src/features/temporal.py                            323      6    98%   222, 298-301, 654
src/integration/__init__.py                          34     18    47%   85-92, 215-229, 248-258, 263
src/integration/api_server.py                       675    356    47%   139, 147, 156, 183, 191, 199, 279-296, 308-309, 316-327, 350-369, 374-436, 504-506, 531-573, 584-604, 687-704, 718-730, 766, 783-785, 815-824, 836, 847-1020, 1042, 1065-1069, 1101-1103, 1113-1115, 1131-1133, 1153-1155, 1169, 1184-1186, 1200, 1214-1216, 1226-1241, 1251-1264, 1274-1294, 1304-1312, 1322-1344, 1356-1376, 1386-1399, 1409-1422, 1436-1475, 1487-1504, 1519-1544, 1559-1587, 1599-1613, 1625-1659, 1684-1702, 1706-1711, 1730-1733
src/integration/auth/__init__.py                      6      0   100%
src/integration/auth/auth_models.py                 156     10    94%   96, 106, 220, 228, 238, 271, 296-298, 302
src/integration/auth/dependencies.py                 93     20    78%   35, 155-162, 189-196, 219-224, 254-278
src/integration/auth/endpoints.py                   148     59    60%   113-116, 123-124, 213-215, 223-224, 250, 288-321, 346-350, 382-397, 409-455, 468-496
src/integration/auth/exceptions.py                   65     43    34%   22-26, 44-50, 67-71, 83-87, 99-103, 115-117, 131-139, 152, 165-171, 188-194
src/integration/auth/jwt_manager.py                 181    140    23%   35-38, 42-50, 54-60, 86, 89, 117-150, 162-183, 199-245, 261-279, 291-308, 320-340, 353-367, 385-416, 420-425, 429-436, 440-448, 460-480
src/integration/auth/middleware.py                  146     49    66%   86, 131-140, 164-165, 177-178, 192-215, 228-241, 269-322, 339, 353-361, 369, 373, 389, 406, 458
src/integration/dashboard.py                        668    558    16%   34-46, 156, 188-193, 212-236, 240-250, 265-285, 297-320, 324-330, 368-398, 406-426, 432-617, 621-658, 662-694, 698-734, 738-820, 829-927, 936-999, 1009-1074, 1083-1181, 1193-1275, 1287-1323, 1333-1372, 1376-1396, 1404-1420, 1429-1462, 1466-1482, 1486-1511, 1515-1556, 1560-1570, 1574-1582, 1594, 1615-1627, 1641-1647
src/integration/discovery_publisher.py              392    263    33%   202-262, 271-326, 341-379, 388-431, 443-469, 484-487, 491, 536-584, 599-661, 680-709, 723-747, 760-777, 791-847, 859, 876, 889, 903, 915, 928, 942, 957, 970, 983, 996, 1008, 1024-1159, 1172
src/integration/enhanced_integration_manager.py     383    313    18%   98-128, 132-176, 180-199, 218-258, 270-316, 332-374, 383-411, 415-425, 437-458, 462-483, 487-517, 521-533, 537-573, 577-598, 602-613, 617-634, 640-658, 664-682, 688-697, 703-723, 729-746, 752-794, 798, 809-811, 817-826, 832-851, 857-868, 872-883, 887-898, 902-915, 922
src/integration/enhanced_mqtt_manager.py            224    180    20%   85-129, 136-157, 163-181, 203-235, 258-305, 309, 315, 319, 323, 327-366, 370-393, 399-401, 405-406, 414-421, 427-439, 445-482, 486-529, 535-551, 555-576, 580-611, 618
src/integration/ha_entity_definitions.py            531    328    38%   151-152, 167-168, 184-185, 205-206, 224-225, 240-241, 258-259, 272-273, 287-288, 339-362, 371-404, 413-444, 453-501, 510-534, 538, 544, 548, 578-745, 749-822, 826-979, 983-1143, 1147-1243, 1247-1371, 1375-1411, 1415-1437, 1443, 1465-1557, 1569-1588, 1594-1603, 1609-1619, 1625-1643, 1649-1660, 1666-1673, 1679-1689, 1695-1699, 1705-1711, 1718
src/integration/ha_tracking_bridge.py               249    249     0%   16-605
src/integration/monitoring_api.py                   131    131     0%   6-285
src/integration/mqtt_integration_manager.py         310    264    15%   70-96, 100-167, 171-190, 194-215, 237-267, 290-317, 326-349, 353-450, 454, 462-464, 468-470, 476-503, 507-522, 526-543, 547, 561-585, 600-650, 662-676, 687-703, 719-733, 740
src/integration/mqtt_publisher.py                   318    268    16%   83-115, 119-163, 167-189, 193-226, 247-339, 366, 370-375, 379, 404-453, 457-463, 467-492, 496-519, 523-562, 568-597, 601-628, 632-641, 645-660, 664-678, 685
src/integration/prediction_publisher.py             163    106    35%   116-131, 150-243, 272-362, 385-397, 401, 420-457, 461-479, 483-488, 498-503
src/integration/realtime_api_endpoints.py           292    230    21%   50, 55-60, 97-118, 128-231, 241-257, 270-294, 308-354, 367-405, 418-431, 444-486, 499-531, 550-613, 633-648, 652-656, 660-676, 680-695, 699, 703, 707, 711-722, 739-777
src/integration/realtime_publisher.py               482    387    20%   65, 85-86, 101, 113, 124-126, 132-147, 151-157, 161-164, 168-171, 177-199, 203-224, 228, 258-260, 264-281, 285-291, 295-298, 304-326, 330-351, 355, 406-438, 445-462, 466-482, 503-608, 622-663, 669-707, 714-747, 760-764, 768-770, 775-780, 802-810, 839-901, 905-923, 927-962, 966-983, 987-999, 1006, 1023-1031, 1037-1090, 1099-1114
src/integration/tracking_integration.py             186    145    22%   76-90, 94-132, 136-153, 157-159, 163-165, 169-202, 206-207, 211-212, 218-232, 236-250, 254-286, 290-350, 354-375, 396-411, 434-457, 464
src/integration/websocket_api.py                    628    469    25%   99, 114-115, 138-140, 187-206, 210, 214-215, 219-227, 231-232, 236, 242-244, 248, 291-304, 315-336, 340-354, 360-386, 392-418, 424-436, 440-465, 474-505, 509-518, 522-544, 549-563, 567-574, 578-594, 607-624, 628-652, 658-682, 686-711, 720-765, 771-790, 796-817, 821-823, 839-888, 892-914, 924-955, 965-996, 1006-1037, 1047-1048, 1054-1056, 1064-1077, 1081-1086, 1095-1103, 1130-1148, 1152-1175, 1179-1215, 1219-1251, 1255-1263, 1267-1277, 1281-1287, 1301-1357, 1363, 1376-1402, 1412-1426, 1434-1440
src/main_system.py                                   78      6    92%   146-157
src/models/__init__.py                                0      0   100%
src/models/base/__init__.py                           6      0   100%
src/models/base/gp_predictor.py                     411    371    10%   34-42, 74-113, 125-238, 259-427, 448-541, 552-595, 605-624, 637-663, 678-713, 727-752, 773-789, 801-821, 825-834, 838-850, 854-871, 893-1041, 1055-1084, 1096-1127, 1131-1175
src/models/base/hmm_predictor.py                    372    338     9%   43-99, 120-277, 298-378, 389-440, 444-461, 472-527, 543-550, 554-575, 588-612, 619-630, 634-646, 655-668, 678-692, 696, 717-752, 764-802, 824-943, 949-972, 982-1003
src/models/base/lstm_predictor.py                   306    280     8%   44-95, 116-299, 318-412, 423-467, 483-566, 581-615, 619-639, 651-680, 692-723, 745-859
src/models/base/predictor.py                        160    104    35%   41-68, 87, 122-138, 163, 186, 196, 216-222, 234-257, 269-294, 298, 322, 334-346, 350, 362-381, 385-389, 393-404, 408-410, 414
src/models/base/xgboost_predictor.py                230    203    12%   42-73, 94-256, 277-349, 360-363, 372-373, 385-402, 421-455, 468-523, 535-559, 568-571, 575-578, 602-630, 642-672
src/models/ensemble.py                              562    520     7%   35-37, 42-44, 64-105, 128-254, 275-341, 352-377, 400-537, 545-625, 634-692, 702-732, 736-764, 775-881, 887-901, 906-936, 957-1072, 1081-1143, 1147-1159, 1170-1255, 1262, 1284-1328, 1340-1391
src/models/training_config.py                       263    174    34%   39-64, 69-72, 96-110, 126-143, 212-225, 238-250, 255-362, 387-411, 418-442, 456-503, 507-515, 519-523, 527, 533-540, 546-553, 559-560, 564-576, 580-628, 632-653, 657-670, 681-684, 691-692
src/models/training_integration.py                  369    327    11%   50-74, 78-91, 95-107, 111-128, 132-152, 158-198, 202-238, 244-258, 269-304, 308-320, 324-330, 334-340, 344-367, 371-420, 424-476, 480-487, 493-522, 526-549, 555-586, 590-612, 616-639, 643-657, 661-670, 674-681, 685, 689-704, 708-715, 727-741, 745-768, 772-790, 794-800, 804-808, 827-841
src/models/training_pipeline.py                     725    575    21%   171-192, 220, 259-290, 309-367, 386-398, 421-449, 472-648, 654-688, 694-723, 729-850, 870, 880-931, 944-977, 999-1034, 1051-1080, 1097-1131, 1148-1174, 1189-1266, 1276-1336, 1345-1406, 1410-1418, 1428-1486, 1490-1491, 1501-1583, 1589-1602, 1608-1632, 1636-1647, 1651-1683, 1689, 1693, 1697, 1701, 1705, 1711-1747, 1753-1771, 1780-1819, 1823-1829, 1835-1876, 1880-1885
src/utils/__init__.py                                 0      0   100%
src/utils/alerts.py                                 252    130    48%   97-111, 115-118, 130-147, 151-161, 167-205, 209-215, 227-264, 277-305, 311, 326, 330-372, 477-478, 482-484, 496-545, 549-562, 575-584, 595-596, 602-620, 633-639, 650
src/utils/health_monitor.py                         533    243    54%   90, 94, 134-145, 224-225, 229, 234-235, 258, 267-268, 285-307, 311-333, 347-351, 369, 389-415, 418-454, 458-462, 479-480, 482-484, 487-488, 490-492, 496-497, 499-501, 532-533, 557-565, 581-582, 603-606, 614, 618-627, 635-636, 656-657, 691-700, 705, 709-745, 766, 789-790, 792-793, 817-818, 837-838, 840-841, 864-865, 900-901, 906-928, 947, 966-982, 988-991, 999-1004, 1027-1028, 1058-1111, 1130-1136, 1141-1176, 1189-1239, 1243, 1287-1291
src/utils/incident_response.py                      392    289    26%   70-77, 81-84, 88-89, 129-135, 139-143, 149-155, 166-178, 188-196, 200, 238-270, 276-318, 333-337, 349-363, 367-381, 385-403, 407-437, 441-520, 524-624, 633-649, 654-662, 670-677, 688-712, 724-740, 744-785, 791-803, 807-821, 825-847, 851-867, 873, 877, 881-882, 886-898, 916-926, 930-945, 955-957
src/utils/logger.py                                 124     37    70%   105, 125, 140, 154, 179-193, 205-211, 217-222, 228-233, 250, 266, 286, 302, 337, 352-364, 380, 385-416, 456
src/utils/metrics.py                                307    206    33%   29-117, 325-340, 353-367, 381-391, 399-411, 423, 431, 439, 447, 453, 459, 475-496, 500, 506, 510, 514-515, 535-541, 547-552, 567-586, 590-592, 596-600, 626, 632-668, 681-690, 694, 698, 707-738, 747-754, 758-766, 776-778, 787-796, 809-825, 838-845
src/utils/monitoring.py                             273    210    23%   60-72, 116, 126-142, 152-176, 180-203, 209-229, 233-241, 247-275, 299-305, 311, 315, 326-359, 369-399, 409-447, 457-488, 498-524, 528-567, 577-583, 591-597, 603-614, 620-653, 657, 661, 665-669, 693-695
src/utils/monitoring_integration.py                 111     83    25%   21-29, 34-38, 42-59, 63-70, 74-85, 92-183, 190-244, 256-274, 290-307, 326-331, 342-347, 361, 367, 373-379, 391-418, 428-430
src/utils/time_utils.py                             233    159    32%   39-43, 47-53, 58, 62-63, 67, 71-76, 90, 97-98, 103-105, 110, 115-121, 126-132, 154-182, 196-224, 240-249, 265-274, 292-312, 329-336, 357-360, 374-399, 412-416, 431-443, 462-479, 492-495, 499-500, 504-506, 511, 516-521, 526-539, 545, 550, 555, 560
-------------------------------------------------------------------------------
TOTAL                                             23499  11736    50%
Coverage HTML written to dir htmlcov-unit
Coverage XML written to file coverage-unit.xml

- Generated html report: file:///home/runner/work/ha-ml-predictor/ha-ml-predictor/report-unit.html -
=========================== short test summary info ============================
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_detect_accuracy_drift_no_drift - AttributeError: 'ConceptDriftDetector' object has no attribute 'detect_accuracy_drift'
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_detect_accuracy_drift_significant_drift - AttributeError: 'ConceptDriftDetector' object has no attribute 'detect_accuracy_drift'
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_calculate_population_stability_index_stable - AttributeError: 'ConceptDriftDetector' object has no attribute '_calculate_population_stability_index'
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_calculate_population_stability_index_drift - AttributeError: 'ConceptDriftDetector' object has no attribute '_calculate_population_stability_index'
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_perform_page_hinkley_test_no_drift - AttributeError: 'ConceptDriftDetector' object has no attribute '_perform_page_hinkley_test'. Did you mean: '_run_page_hinkley_test'?
FAILED tests/unit/test_adaptation_consolidated.py::TestConceptDriftDetector::test_perform_page_hinkley_test_with_drift - AttributeError: 'ConceptDriftDetector' object has no attribute '_perform_page_hinkley_test'. Did you mean: '_run_page_hinkley_test'?
FAILED tests/unit/test_adaptation_consolidated.py::TestFeatureDriftDetector::test_detect_feature_drift_no_drift - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/unit/test_adaptation_consolidated.py::TestFeatureDriftDetector::test_detect_feature_drift_with_drift - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/unit/test_adaptation_consolidated.py::TestFeatureDriftDetector::test_statistical_tests - AttributeError: 'FeatureDriftDetector' object has no attribute '_perform_kolmogorov_smirnov_test'
FAILED tests/unit/test_adaptation_consolidated.py::TestRealTimeMetrics::test_is_healthy_property - AssertionError: assert False
 +  where False = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=0.0, window_24h_accuracy=85.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=0, window_24h_predictions=50, accuracy_trend=<TrendDirection.STABLE: 'stable'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 7, 187902), measurement_start=None).is_healthy
FAILED tests/unit/test_adaptation_consolidated.py::TestAccuracyAlert::test_alert_initialization - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation_consolidated.py::TestAccuracyAlert::test_alert_resolution - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation_consolidated.py::TestPredictionValidator::test_validator_initialization - AttributeError: 'PredictionValidator' object has no attribute 'pending_validations'. Did you mean: 'get_pending_validations'?
FAILED tests/unit/test_adaptation_consolidated.py::TestPredictionValidator::test_record_prediction - TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
FAILED tests/unit/test_adaptation_consolidated.py::TestPredictionValidator::test_validate_prediction_accurate - TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
FAILED tests/unit/test_adaptation_consolidated.py::TestPredictionValidator::test_validate_prediction_inaccurate - TypeError: PredictionValidator.record_prediction() got an unexpected keyword argument 'features'
FAILED tests/unit/test_adaptation_consolidated.py::TestPredictionValidator::test_get_accuracy_metrics - src.adaptation.validator.ValidationError: Failed to calculate accuracy metrics | Error Code: VALIDATION_ERROR | Caused by: TypeError: unsupported type for timedelta hours component: datetime.datetime
FAILED tests/unit/test_core/test_config.py::TestConfigurationCorruptionAndRecovery::test_yaml_with_circular_references_deep - TypeError: HomeAssistantConfig.__init__() got an unexpected keyword argument 'reference'
FAILED tests/unit/test_core/test_config.py::TestConfigurationCorruptionAndRecovery::test_configuration_with_malformed_data_types - Failed: DID NOT RAISE <class 'TypeError'>
FAILED tests/unit/test_core/test_config.py::TestConfigurationRecoveryMechanisms::test_configuration_fallback_to_defaults - KeyError: 'prediction'
FAILED tests/unit/test_core/test_config.py::TestConfigurationRecoveryMechanisms::test_configuration_validation_recovery - Failed: DID NOT RAISE <class 'TypeError'>
FAILED tests/unit/test_adaptation_consolidated.py::TestAdaptationPerformanceEdgeCases::test_large_validation_history_performance - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/test_adaptation_consolidated.py::TestAdaptationPerformanceEdgeCases::test_memory_management_with_history_limit - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/test_adaptation_consolidated.py::TestAdaptationPerformanceEdgeCases::test_edge_case_empty_data - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/test_adaptation_consolidated.py::TestAdaptationPerformanceEdgeCases::test_extreme_accuracy_values - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/test_adaptation_consolidated.py::TestAdaptationPerformanceEdgeCases::test_concurrent_validation_recording - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/test_main_system.py::TestOccupancyPredictionSystem::test_run_without_initialization - TypeError: object MagicMock can't be used in 'await' expression
FAILED tests/unit/test_main_system.py::TestOccupancyPredictionSystem::test_shutdown_handles_tracking_manager_exception - assert True is False
 +  where True = <src.main_system.OccupancyPredictionSystem object at 0x7fef0cf9e5b0>.running
FAILED tests/unit/test_main_system.py::TestOccupancyPredictionSystem::test_system_passes_correct_config_to_components - TypeError: 'coroutine' object is not subscriptable
FAILED tests/unit/test_main_system.py::TestMainSystemErrorScenarios::test_api_server_status_check_failure - AssertionError: Regex pattern did not match.
 Regex: 'API status check failed'
 Input: "'coroutine' object is not subscriptable"
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_without_confidence - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_type'. Did you mean 'predicted_time'?
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_initialization - AssertionError: assert <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> != <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>>
 +  where <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> = <MagicMock spec='TrackingManager' id='140664671116256'>.record_prediction
 +    where <MagicMock spec='TrackingManager' id='140664671116256'> = <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>.tracking_manager
 +  and   <bound method MonitoringEnhancedTrackingManager._monitored_record_prediction of <src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fef0bbf7590>> = <MagicMock spec='TrackingManager' id='140664671116256'>.record_prediction
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_method_wrapping - AttributeError: module 'src.adaptation' has no attribute 'monitoring_integration'
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_no_confidence - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_track_model_training_context - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_track_model_training_default_type - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitoring_integration_none_error - Failed: DID NOT RAISE <class 'AttributeError'>
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_empty_prediction_result_attributes - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestFactoryFunctions::test_get_enhanced_tracking_manager - AssertionError: expected call not found.
Expected: create_monitoring_enhanced_tracking_manager(config=<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')
  Actual: create_monitoring_enhanced_tracking_manager(<MagicMock spec='TrackingConfig' id='140664673065616'>, test_param='value')

pytest introspection follows:

Args:
assert (<MagicMock s...4673065616'>,) == ()
  Left contains one more item: <MagicMock spec='TrackingConfig' id='140664673065616'>
  Full diff:
  - ()
  + (<MagicMock spec='TrackingConfig' id='140664673065616'>,)
Kwargs:
assert {'test_param': 'value'} == {'config': <M...ram': 'value'}
  Omitting 1 identical items, use -vv to show
  Right contains 1 more item:
  {'config': <MagicMock spec='TrackingConfig' id='140664673065616'>}
  Full diff:
    {
  -  'config': <MagicMock spec='TrackingConfig' id='140664673065616'>,
     'test_param': 'value',
    }
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestIntegrationScenarios::test_complete_prediction_lifecycle - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
FAILED tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestIntegrationScenarios::test_error_recovery_scenarios - AttributeError: 'method' object has no attribute 'side_effect' and no __dict__ for setting new attributes
FAILED tests/unit/test_core/test_config.py::TestConfigurationStabilityAndResilience::test_configuration_loading_with_system_stress - assert 17 > 100
FAILED tests/unit/test_adaptation/test_tracker.py::TestRealTimeMetrics::test_overall_health_score_calculation - AssertionError: assert 69.2 > 80.0
 +  where 69.2 = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=92.0, window_24h_accuracy=88.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=15, window_24h_predictions=120, accuracy_trend=<TrendDirection.IMPROVING: 'improving'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 17, 45678), measurement_start=None).overall_health_score
FAILED tests/unit/test_adaptation/test_tracker.py::TestRealTimeMetrics::test_is_healthy_criteria - AssertionError: assert False
 +  where False = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=70.0, window_24h_accuracy=70.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=5, window_24h_predictions=40, accuracy_trend=<TrendDirection.UNKNOWN: 'unknown'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 23, 7, 57, 17, 70568), measurement_start=None).is_healthy
FAILED tests/unit/test_adaptation/test_tracker.py::TestRealTimeMetrics::test_to_dict_serialization - AssertionError: assert 'ModelType.LSTM' == 'LSTM'
  - LSTM
  + ModelType.LSTM
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_accuracy_alert_initialization - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_age_calculation - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_escalation_requirements - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_acknowledge_alert - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_resolve_alert - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_escalate_alert - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyAlert::test_alert_to_dict - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
FAILED tests/unit/test_adaptation/test_tracker.py::TestAccuracyTracker::test_tracker_initialization - AttributeError: 'AccuracyTracker' object has no attribute '_prediction_validator'
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_analyze_trend_insufficient_data - ValueError: too many values to unpack (expected 2)
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_analyze_trend_improving - ValueError: too many values to unpack (expected 2)
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_analyze_trend_degrading - ValueError: too many values to unpack (expected 2)
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_analyze_trend_stable - ValueError: too many values to unpack (expected 2)
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_calculate_global_trend - AssertionError: assert 'overall_direction' in {'average_slope': 0.0, 'confidence': 0.0, 'direction': <TrendDirection.UNKNOWN: 'unknown'>}
FAILED tests/unit/test_adaptation/test_tracker.py::TestTrendAnalysis::test_calculate_global_trend_empty - KeyError: 'overall_direction'
FAILED tests/unit/test_core/test_backup_manager.py::TestModelBackupManager::test_create_backup_uncompressed_success - AssertionError: assert 'test_models.tar' in '-C'
FAILED tests/unit/test_core/test_backup_manager.py::TestModelBackupManager::test_create_backup_nonexistent_models_dir - AssertionError: assert False
 +  where False = <bound method PathBase.exists of PosixPath('/path/that/does/not/exist')>()
 +    where <bound method PathBase.exists of PosixPath('/path/that/does/not/exist')> = PosixPath('/path/that/does/not/exist').exists
 +      where PosixPath('/path/that/does/not/exist') = Path('/path/that/does/not/exist')
FAILED tests/unit/test_core/test_backup_manager.py::TestBackupManager::test_list_backups_with_data - FileExistsError: [Errno 17] File exists: '/tmp/tmpzsxzshs8/database'
FAILED tests/unit/test_core/test_backup_manager.py::TestBackupManager::test_list_backups_filtered_by_type - FileExistsError: [Errno 17] File exists: '/tmp/tmpekluij2_/database'
FAILED tests/unit/test_core/test_backup_manager.py::TestBackupManager::test_get_backup_info_found - FileExistsError: [Errno 17] File exists: '/tmp/tmpoovz5ajt/database'
FAILED tests/unit/test_core/test_backup_manager.py::TestBackupManager::test_cleanup_expired_backups - FileExistsError: [Errno 17] File exists: '/tmp/tmpcv_cmbhh/database'
FAILED tests/unit/test_core/test_backup_manager.py::TestBackupManager::test_run_scheduled_backups_single_cycle - AssertionError: Expected 'create_backup' to have been called once. Called 0 times.
FAILED tests/unit/test_data/test_event_processor_comprehensive.py::test_event_processor_comprehensive_test_suite_completion - AssertionError: Expected 65+ test methods, found 54
assert 54 >= 65
FAILED tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityValidator::test_path_traversal_detection - AssertionError: Failed to detect path traversal: ..%252f..%252f..%252fetc%252fpasswd
assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityValidator::test_input_sanitization_aggressive - AssertionError: assert '=' not in '* FROM user... id=1 OR 1=1'
  '=' is contained here:
    * FROM users WHERE id=1 OR 1=1
  ?                      +
FAILED tests/unit/test_data/test_event_validator_comprehensive.py::TestIntegrityValidator::test_duplicate_detection - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_event_validator_comprehensive.py::TestIntegrityValidator::test_cross_system_consistency_validation - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_event_validator_comprehensive.py::TestPerformanceValidator::test_performance_stats_tracking - AssertionError: assert 'batches_processed' in {}
FAILED tests/unit/test_data/test_models_advanced.py::TestSensorEventAdvancedFeatures::test_get_sensor_efficiency_metrics - sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) misuse of window function lag()
[SQL: SELECT sensor_events.sensor_id, sensor_events.sensor_type, count(*) AS total_events, count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state) AS state_changes, avg(sensor_events.confidence_score) AS avg_confidence, min(sensor_events.confidence_score) AS min_confidence, max(sensor_events.confidence_score) AS max_confidence, (count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state)) / (CAST(count(*) AS FLOAT) + 0.0) AS state_change_ratio, CAST(STRFTIME('%s', avg(sensor_events.timestamp - lag(sensor_events.timestamp) OVER (PARTITION BY sensor_events.sensor_id ORDER BY sensor_events.timestamp))) AS INTEGER) AS avg_interval_seconds 
FROM sensor_events 
WHERE sensor_events.room_id = ? AND sensor_events.timestamp >= ? GROUP BY sensor_events.sensor_id, sensor_events.sensor_type ORDER BY count(*) DESC]
[parameters: ('efficiency_room', '2025-08-22 07:57:24.004992')]
(Background on this error at: https://sqlalche.me/e/20/e3q8)
FAILED tests/unit/test_data/test_models_advanced.py::TestSensorEventAdvancedFeatures::test_efficiency_score_calculation - AssertionError: Score 0.6200000000000001 not in range 0.2-0.6
assert 0.6200000000000001 <= 0.6
FAILED tests/unit/test_data/test_models_advanced.py::TestTimescaleDBFunctions::test_create_timescale_hypertables - AssertionError: assert 5 >= 6
 +  where 5 = <AsyncMock name='execute' id='140664672170208'>.call_count
FAILED tests/unit/test_data/test_models_advanced.py::TestDatabaseCompatibilityHelpers::test_get_json_column_type_postgresql - AssertionError: assert <class 'sqlalchemy.sql.sqltypes.JSON'> == <class 'sqlalchemy.dialects.postgresql.json.JSONB'>
FAILED tests/unit/test_data/test_models_advanced.py::TestModelsIntegration::test_model_relationships_cascade - sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: prediction_audits.prediction_id
[SQL: UPDATE prediction_audits SET prediction_id=? WHERE prediction_audits.id = ?]
[parameters: (None, 1)]
(Background on this error at: https://sqlalche.me/e/20/gkpj)
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_statistical_anomaly_detection_with_outliers - assert 2 == 3
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_high_frequency - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_low_frequency - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestStatisticalPatternAnalyzer::test_detect_sensor_malfunction_unstable_timing - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestCorruptionDetector::test_detect_timestamp_corruption - AssertionError: assert 'COR003' in {'COR001', 'COR002'}
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestRealTimeQualityMonitor::test_accuracy_score_calculation - assert 0.8666666666666667 <= 0.8
 +  where 0.8666666666666667 = DataQualityMetrics(completeness_score=0.0, consistency_score=0.5, accuracy_score=0.8666666666666667, timeliness_score=1.0, anomaly_count=0, corruption_indicators=[], quality_trends={}).accuracy_score
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestEdgeCasesAndErrorHandling::test_analyzer_with_malformed_events - TypeError: argument of type 'NoneType' is not iterable
FAILED tests/unit/test_data/test_pattern_detector_comprehensive.py::TestEdgeCasesAndErrorHandling::test_quality_monitor_with_inconsistent_data_types - TypeError: unhashable type: 'list'
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_iso_datetime_format_validation - AssertionError: Invalid datetime accepted: 2024-01-15 10:30:00
assert not True
 +  where True = <bound method JSONSchemaValidator._validate_iso_datetime_format of <src.data.validation.schema_validator.JSONSchemaValidator object at 0x7fef2892f1d0>>('2024-01-15 10:30:00')
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_duration_format_validation - AssertionError: Invalid duration accepted: P
assert not True
 +  where True = <bound method JSONSchemaValidator._validate_duration_format of <src.data.validation.schema_validator.JSONSchemaValidator object at 0x7fef28990e50>>('P')
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validate_sensor_event_schema_valid_data - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='0f2dbbda-14a1-4102-8cfc-7a54bb65adad').is_valid
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validate_sensor_event_schema_missing_required_fields - assert 1 >= 3
 +  where 1 = len([ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})])
 +    where [ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})] = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='ee51bca3-3ce4-4964-8e22-bbb7c1a997c6').errors
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validate_sensor_event_schema_invalid_types - assert False
 +  where False = any(<generator object TestJSONSchemaValidator.test_validate_sensor_event_schema_invalid_types.<locals>.<genexpr> at 0x7fef0d04e0c0>)
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validate_room_config_schema_valid_data - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='390b42a5-6725-4607-80fb-f274d4db49c4').is_valid
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validate_room_config_schema_invalid_structure - AssertionError: assert 'format' in set()
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validation_context_strict_mode - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_validation_context_allow_additional - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='658447e8-39ac-458e-8641-de59b86035a0').is_valid
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_register_custom_schema - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='ff56fd2c-87fc-493c-88e7-a546ca4c4263').is_valid
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestJSONSchemaValidator::test_custom_validator_integration - assert 0 == 1
 +  where 0 = len([])
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestDatabaseSchemaValidator::test_validate_database_schema_complete_setup - AssertionError: assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.model_type', message='Missing required column: predictions.model_type', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the model_type column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.model_version', message='Missing required column: predictions.model_version', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the model_version column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.predicted_time', message='Missing required column: predictions.predicted_time', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the predicted_time column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.confidence', message='Missing required column: predictions.confidence', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the confidence column to predictions table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='predictions.features_hash', message='Missin...message='Missing required column: room_states.confidence', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the confidence column to room_states table', context={}), ValidationError(rule_id='DB_SCH_003', field='column', value='room_states.occupancy_state', message='Missing required column: room_states.occupancy_state', severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Add the occupancy_state column to room_states table', context={}), ValidationError(rule_id='DB_SCH_009', field='index', value='room_states.idx_room_timestamp', message='Missing critical index: idx_room_timestamp on room_states', severity=<ErrorSeverity.MEDIUM: 'medium'>, suggestion='Create index idx_room_timestamp for better query performance', context={}), ValidationError(rule_id='DB_SCH_009', field='index', value='predictions.idx_room_predicted_time', message='Missing critical index: idx_room_predicted_time on predictions', severity=<ErrorSeverity.MEDIUM: 'medium'>, suggestion='Create index idx_room_predicted_time for better query performance', context={})], warnings=[], confidence_score=0.0, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='15f4a84c-3e7e-4bc9-985e-25f5f20ba117').is_valid
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestAPISchemaValidator::test_validate_headers_format_errors - TypeError: object of type 'int' has no len()
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestEdgeCasesAndErrorHandling::test_database_validator_with_connection_issues - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestEdgeCasesAndErrorHandling::test_schema_validator_memory_efficiency - assert False
 +  where False = all(<generator object TestEdgeCasesAndErrorHandling.test_schema_validator_memory_efficiency.<locals>.<genexpr> at 0x7fef2891d540>)
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestEdgeCasesAndErrorHandling::test_concurrent_schema_validation - assert False
 +  where False = all(<generator object TestEdgeCasesAndErrorHandling.test_concurrent_schema_validation.<locals>.<genexpr> at 0x7fef2891db40>)
FAILED tests/unit/test_data/test_schema_validator_comprehensive.py::TestIntegrationScenarios::test_complete_sensor_event_validation_pipeline - AssertionError: Completely valid data: expected True, got False
assert False == True
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='a9ba47be-fe76-4df1-ad07-de000396c7f3').is_valid
FAILED tests/unit/test_core/test_config_validator.py::TestDatabaseConfigValidator::test_validate_pool_size_warnings - assert False
 +  where False = any(<generator object TestDatabaseConfigValidator.test_validate_pool_size_warnings.<locals>.<genexpr> at 0x7fec866f1700>)
FAILED tests/unit/test_core/test_config_validator.py::TestDatabaseConfigValidator::test_connection_test_success - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["❌ Connection failed: 'coroutine' object has no attribute 'fetchval'"], warnings=[], info=[]).is_valid
FAILED tests/unit/test_core/test_config_validator.py::TestDatabaseConfigValidator::test_connection_test_auth_failure - assert False
 +  where False = any(<generator object TestDatabaseConfigValidator.test_connection_test_auth_failure.<locals>.<genexpr> at 0x7fec856536b0>)
FAILED tests/unit/test_data/test_validation.py::TestStatisticalPatternAnalyzer::test_statistical_anomaly_detection - AssertionError: assert 'anomaly_count' in {'event_count': 11, 'mean_interval': 300.1, 'median_interval': 300.0, 'state_distribution': {'on': 1.0}, ...}
FAILED tests/unit/test_core/test_config_validator.py::TestDatabaseConfigValidator::test_connection_test_database_not_found - assert False
 +  where False = any(<generator object TestDatabaseConfigValidator.test_connection_test_database_not_found.<locals>.<genexpr> at 0x7fec85650c70>)
FAILED tests/unit/test_data/test_validation.py::TestStatisticalPatternAnalyzer::test_sensor_malfunction_detection - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_data/test_validation.py::TestJSONSchemaValidator::test_sensor_event_schema_validation - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='68707c54-6b16-4b82-bd6c-b4b534f89f54').is_valid
FAILED tests/unit/test_data/test_validation.py::TestJSONSchemaValidator::test_invalid_sensor_event_schema - assert 0 > 0
 +  where 0 = len([])
FAILED tests/unit/test_core/test_config_validator.py::TestSystemRequirementsValidator::test_validate_sufficient_system - AttributeError: 'tuple' object has no attribute 'major'
FAILED tests/unit/test_data/test_validation.py::TestJSONSchemaValidator::test_schema_registration - assert False
 +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='e9924eb5-be19-40ac-8c7c-e0187d54162b').is_valid
FAILED tests/unit/test_core/test_config_validator.py::TestSystemRequirementsValidator::test_validate_old_python_version - AttributeError: 'tuple' object has no attribute 'major'
FAILED tests/unit/test_data/test_validation.py::TestComprehensiveEventValidator::test_event_sanitization - AssertionError: assert 'DROP TABLE' not in '&#x27;; DRO...LE users; --'
  'DROP TABLE' is contained here:
    &#x27;; DROP TABLE users; --
  ?         ++++++++++
FAILED tests/unit/test_core/test_config_validator.py::TestSystemRequirementsValidator::test_validate_old_python_version_warning - AttributeError: 'tuple' object has no attribute 'major'
FAILED tests/unit/test_core/test_config_validator.py::TestSystemRequirementsValidator::test_validate_low_disk_space_warning - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['Low disk space (< 5 GB available)'], info=['Python version: 3.13.7', 'Available disk space: 3.0 GB', 'Total memory: 15.6 GB, Available: 14.2 GB']).is_valid
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestAdvancedCacheInvalidation::test_cache_invalidation_on_data_freshness - AttributeError: 'MemoryAwareCache' object has no attribute 'size'
FAILED tests/unit/test_core/test_config_validator.py::TestConfigurationValidator::test_validate_configuration_all_valid - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', 'Very few sensors configured, predictions may be less accurate'], info=['Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room living_room: 3 sensor types, 3 entities', 'Rooms configured: 1', 'Total sensors: 3', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestAdvancedCacheInvalidation::test_cascading_cache_invalidation - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestAdvancedCacheInvalidation::test_selective_cache_invalidation_by_pattern - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_core/test_config_validator.py::TestConfigurationValidator::test_validate_configuration_with_connection_tests - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', 'Very few sensors configured, predictions may be less accurate'], info=['Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room living_room: 3 sensor types, 3 entities', 'Rooms configured: 1', 'Total sensors: 3', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', 'Testing external connections...', 'Connection successful', 'Connection successful', 'Connection successful', '💥 Configuration validation failed with 1 errors']).is_valid
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestAdvancedCacheInvalidation::test_cache_invalidation_on_config_changes - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_core/test_config_validator.py::TestConfigurationValidator::test_validate_config_files_success - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', "Room test_room missing essential sensor types: ['occupancy', 'door']", 'Very few sensors configured, predictions may be less accurate'], info=['Loaded configuration from: /tmp/tmpeki6adew/config.yaml', 'Loaded rooms configuration from: /tmp/tmpeki6adew/rooms.yaml', 'Home Assistant URL: http://localhost:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: localhost', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room test_room: 1 sensor types, 1 entities', 'Rooms configured: 1', 'Total sensors: 1', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestAdvancedCacheInvalidation::test_time_based_cache_invalidation_with_sliding_window - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheMemoryEfficiency::test_memory_efficient_large_feature_sets - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_core/test_config_validator.py::TestConfigurationValidator::test_validate_config_files_environment_specific - assert False is True
 +  where False = ValidationResult(is_valid=False, errors=["Missing required packages: ['scikit-learn']"], warnings=['TimescaleDB extension is recommended for time-series data', 'Topic prefix is empty, messages will be published to root topic', "Room test_room missing essential sensor types: ['occupancy', 'door']", 'Very few sensors configured, predictions may be less accurate'], info=['Loaded configuration from: /tmp/tmph2ejcvs8/config.staging.yaml', 'Loaded rooms configuration from: /tmp/tmph2ejcvs8/rooms.yaml', 'Home Assistant URL: http://staging:8123', 'Home Assistant token is configured', '✅ Home Assistant configuration is valid', 'Database connection string configured', 'Pool settings: size=10, max_overflow=20', '✅ Database configuration is valid', 'MQTT broker: staging-mqtt', 'MQTT discovery enabled with prefix: homeassistant', '✅ MQTT configuration is valid', 'Room test_room: 1 sensor types, 1 entities', 'Rooms configured: 1', 'Total sensors: 1', '✅ Rooms configuration is valid', 'Python version: 3.13.7', 'Available disk space: 23.6 GB', 'Total memory: 15.6 GB, Available: 14.2 GB', '❌ System Requirements configuration has errors', '💥 Configuration validation failed with 1 errors']).is_valid
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheMemoryEfficiency::test_weak_reference_cache_cleanup - TypeError: cannot create weak reference to 'dict' object
FAILED tests/unit/test_features/test_cache_invalidation_advanced.py::TestCacheConcurrencyAndCoherence::test_cache_eviction_under_concurrent_load - AttributeError: 'FeatureCache' object has no attribute 'size'
FAILED tests/unit/test_core/test_environment.py::TestSecretsManager::test_get_or_create_key_creates_new_key - AssertionError: assert 44 == 32
 +  where 44 = len(b'68F-TlOEc9wu65W_ZVSei9YWn6OfsIhgBsgpolkAj8c=')
FAILED tests/unit/test_core/test_environment.py::TestEnvironmentManager::test_inject_secrets - AssertionError: assert 'user:test_db_pass@' in 'postgresql://user@localhost/db'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestExceptionContextPreservation::test_exception_context_filtering_sensitive_data - AssertionError: assert {'name': 'John', 'ssn': '123-...'} == '[FILTERED]'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestExceptionHierarchyValidation::test_all_exceptions_inherit_from_base - TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestSystemLayerErrorPropagation::test_external_service_to_internal_error_propagation - TypeError: MissingFeatureError.__init__() got an unexpected keyword argument 'cause'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestValidationFunctionEdgeCases::test_validate_room_id_comprehensive - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestValidationFunctionEdgeCases::test_validate_entity_id_comprehensive - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestExceptionLoggingIntegration::test_error_alerting_classification - AssertionError: Classification mismatch for ModelPredictionError: alert_level
assert 'info' == 'warning'
  - warning
  + info
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestProductionErrorScenarios::test_memory_pressure_error_handling - AttributeError: 'ResourceExhaustionError' object has no attribute 'resource_type'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestProductionErrorScenarios::test_cascading_failure_scenario - KeyError: 'model_predictions'
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestProductionErrorScenarios::test_data_corruption_detection_and_handling - assert ('corruption' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" or 'invalid' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}")
 +  where "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fec85391c30>()
 +    where <built-in method lower of str object at 0x7fec85391c30> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
 +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
 +  and   "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fec8527df30>()
 +    where <built-in method lower of str object at 0x7fec8527df30> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
 +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
FAILED tests/unit/test_core/test_exception_propagation_advanced.py::TestProductionErrorScenarios::test_graceful_degradation_error_patterns - assert False
 +  where False = any(<generator object TestProductionErrorScenarios.test_graceful_degradation_error_patterns.<locals>.<genexpr> at 0x7fec851be880>)
FAILED tests/unit/test_core/test_exceptions.py::TestHomeAssistantErrors::test_home_assistant_authentication_error_with_string_token - AssertionError: assert 'very_long_token...' in 'very_long_...'
FAILED tests/unit/test_core/test_exceptions.py::TestHomeAssistantErrors::test_home_assistant_authentication_error_with_short_token - AssertionError: assert 'short_toke...' == 'short_token'
  - short_token
  ?           ^
  + short_toke...
  ?           ^^^
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_system_error_basic - AssertionError: assert 'Something we... SYSTEM_ERROR' == 'Something went wrong'
  - Something went wrong
  + Something went wrong | Error Code: SYSTEM_ERROR
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_system_error_with_operation - AssertionError: assert 'System error...ta_processing' == 'System error...ed to process'
  - System error during data_processing: Failed to process
  + System error during data_processing: Failed to process | Error Code: SYSTEM_ERROR | Context: operation=data_processing
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_system_error_with_component_and_operation - AssertionError: assert 'System error...Invalid input' == 'System error...dation failed'
  - System error in input_validator during user_input_validation: Validation failed
  + System error in input_validator during user_input_validation: Validation failed | Error Code: SYSTEM_ERROR | Context: component=input_validator, operation=user_input_validation | Caused by: ValueError: Invalid input
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_resource_exhaustion_error - AssertionError: assert 'System error...=90.0, unit=%' == 'Resource exh...limit: 90.0%)'
  - Resource exhaustion: memory at 95.5% (limit: 90.0%)
  + System error in memory during resource_monitoring: Resource exhaustion: memory at 95.5% (limit: 90.0%) | Error Code: RESOURCE_EXHAUSTION_ERROR | Context: component=memory, operation=resource_monitoring, resource_type=memory, current_usage=95.5, limit=90.0, unit=%
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_service_unavailable_error_basic - KeyError: 'operation'
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_maintenance_mode_error_basic - AssertionError: assert 'System error...CE_MODE_ERROR' == 'System in maintenance mode'
  - System in maintenance mode
  + System error in system during maintenance: System in maintenance mode | Error Code: MAINTENANCE_MODE_ERROR
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_maintenance_mode_error_with_component - AssertionError: assert 'System error...nent=database' == 'System compo...ode: database'
  - System component in maintenance mode: database
  + System error in database during maintenance: System component in maintenance mode: database | Error Code: MAINTENANCE_MODE_ERROR | Context: component=database
FAILED tests/unit/test_core/test_exceptions.py::TestSystemErrors::test_maintenance_mode_error_with_end_time - AssertionError: assert 'System error...-15 14:00 UTC' == 'System compo...15 14:00 UTC)'
  - System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC)
  + System error in search_engine during maintenance: System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC) | Error Code: MAINTENANCE_MODE_ERROR | Context: component=search_engine, estimated_end_time=2024-01-15 14:00 UTC
FAILED tests/unit/test_core/test_exceptions.py::TestAPIErrors::test_rate_limit_exceeded_error_basic - AssertionError: assert 'Rate limit e..._seconds=3600' == 'Rate limit e...sts per 3600s'
  - Rate limit exceeded for api: 100 requests per 3600s
  + Rate limit exceeded for api: 100 requests per 3600s | Error Code: RATE_LIMIT_EXCEEDED_ERROR | Context: service=api, limit=100, window_seconds=3600
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_room_id_invalid_empty - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_room_id_invalid_none - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_room_id_invalid_type - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_room_id_invalid_characters - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_entity_id_invalid_empty - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_entity_id_invalid_none - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_entity_id_invalid_type - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestValidationFunctions::test_validate_entity_id_invalid_format - TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
FAILED tests/unit/test_core/test_exceptions.py::TestExceptionInheritanceAndCompatibility::test_all_exceptions_are_exceptions - TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
FAILED tests/unit/test_core/test_exceptions.py::TestExceptionInheritanceAndCompatibility::test_error_context_preservation - TypeError: ConfigValidationError.__init__() got an unexpected keyword argument 'context'
FAILED tests/unit/test_core/test_jwt_configuration.py::TestJWTConfigurationSecurityValidation::test_jwt_secret_key_minimum_length_validation - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/unit/test_core/test_jwt_configuration.py::TestJWTConfigurationSecurityValidation::test_jwt_secret_key_acceptable_lengths - ValueError: JWT secret key must be at least 32 characters long
FAILED tests/unit/test_core/test_jwt_configuration.py::TestJWTConfigurationSecurityValidation::test_jwt_secret_key_missing_in_production - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/unit/test_core/test_jwt_configuration.py::TestJWTEnvironmentHandling::test_jwt_test_environment_fallback_behavior - ValueError: JWT is enabled but JWT_SECRET_KEY environment variable is not set
FAILED tests/unit/test_data/test_database_comprehensive.py::TestDatabaseConfig::test_database_config_creation - TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorDurationFeatures::test_extract_duration_features_basic - assert 1800.0 == 600.0
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_multiple_doors_handling - assert 1.25 <= 1.0
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_single_room_handling - assert 0.0 == 1.0
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_multi_sensor_event_ratio_accuracy - assert 0.6666666666666666 < 0.6
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_sensor_correlation_empty_data - assert 0.0 == 1.0
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_feature_extraction_error_handling - Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
FAILED tests/unit/test_features/test_contextual_comprehensive.py::TestContextualFeatureExtractorComprehensive::test_get_feature_names_method - AssertionError: assert 50 > 50
 +  where 50 = len(['current_temperature', 'avg_temperature', 'temperature_trend', 'temperature_variance', 'temperature_change_rate', 'temperature_stability', ...])
FAILED tests/unit/test_features/test_coverage_validation.py::TestTemporalCoverageGaps::test_temporal_extract_room_state_features_edge_cases - AssertionError: assert 'room_state_confidence_avg' in {'avg_occupancy_confidence': 0.5, 'recent_occupancy_ratio': 0.5, 'state_stability': 0.5}
FAILED tests/unit/test_features/test_coverage_validation.py::TestTemporalCoverageGaps::test_temporal_extract_historical_patterns_insufficient_data - AssertionError: assert 'historical_trend' in {'activity_variance': 0.0, 'day_activity_rate': 1.0, 'hour_activity_rate': 1.0, 'overall_activity_rate': 1.0, ...}
FAILED tests/unit/test_features/test_coverage_validation.py::TestTemporalCoverageGaps::test_temporal_extract_cyclical_features_edge_times - AssertionError: assert 'cyclical_hour_sin' in {'day_of_month_cos': -0.994869323391895, 'day_of_month_sin': 0.10116832198743272, 'day_of_week_cos': 1.0, 'day_of_week_sin': 0.0, ...}
FAILED tests/unit/test_features/test_coverage_validation.py::TestContextualCoverageGaps::test_contextual_environmental_features_with_invalid_values - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_coverage_validation.py::TestContextualCoverageGaps::test_contextual_multi_room_correlation_single_room - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_coverage_validation.py::TestEngineeringCoverageGaps::test_engineering_get_default_features_all_types - AttributeError: 'FeatureEngineeringEngine' object has no attribute 'get_default_features'
FAILED tests/unit/test_features/test_coverage_validation.py::TestEngineeringCoverageGaps::test_engineering_validate_configuration_invalid - TypeError: FeatureEngineeringEngine.validate_configuration() takes 1 positional argument but 2 were given
FAILED tests/unit/test_features/test_coverage_validation.py::TestEngineeringCoverageGaps::test_engineering_parallel_vs_sequential_extraction - TypeError: FeatureEngineeringEngine.extract_features() got an unexpected keyword argument 'parallel'
FAILED tests/unit/test_features/test_coverage_validation.py::TestStoreCoverageGaps::test_feature_cache_expired_record_cleanup - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_coverage_validation.py::TestStoreCoverageGaps::test_feature_store_compute_training_data_edge_cases - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_coverage_validation.py::TestStoreCoverageGaps::test_feature_store_cache_key_generation_complex - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_engineering_comprehensive.py::TestFeatureEngineeringEngineComprehensive::test_create_feature_dataframe - AssertionError: assert 'sequential_room_count' in Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138)
 +  where Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138) =    temporal_time_since_last_event  ...  meta_data_quality_score\n0                             0.0  ...                      0.0\n1                             0.0  ...                      0.0\n\n[2 rows x 138 columns].columns
FAILED tests/unit/test_features/test_engineering_comprehensive.py::TestFeatureEngineeringEngineComprehensive::test_end_to_end_integration - AssertionError: assert 138 == 157
 +  where 138 = len(Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138))
 +    where Index(['temporal_time_since_last_event', 'temporal_time_since_last_on',\n       'temporal_time_since_last_off', 'temporal_time_since_last_motion',\n       'temporal_current_state_duration', 'temporal_avg_on_duration',\n       'temporal_avg_off_duration', 'temporal_max_on_duration',\n       'temporal_max_off_duration', 'temporal_on_duration_std',\n       ...\n       'contextual_max_room_activity', 'contextual_room_activity_variance',\n       'contextual_presence_sensor_ratio', 'contextual_door_sensor_ratio',\n       'contextual_climate_sensor_ratio', 'meta_event_count',\n       'meta_room_state_count', 'meta_extraction_hour',\n       'meta_extraction_day_of_week', 'meta_data_quality_score'],\n      dtype='object', length=138) =    temporal_time_since_last_event  ...  meta_data_quality_score\n0                        -14400.0  ...                     0.05\n\n[1 rows x 138 columns].columns
 +  and   157 = len({'contextual_active_rooms_count': 1, 'contextual_avg_door_open_duration': 0.0, 'contextual_avg_humidity': 50.0, 'contextual_avg_light': 500.0, ...})
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestComponentFailureRecovery::test_temporal_extractor_partial_method_failures - src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140664671286816'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140664671286816'> | Caused by: RuntimeError: Cyclical feature extraction failed
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestComponentFailureRecovery::test_sequential_extractor_classifier_failure_fallback - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room room1 | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=room1 | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestComponentFailureRecovery::test_contextual_extractor_environmental_sensor_degradation - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestComponentFailureRecovery::test_feature_engineering_engine_extractor_orchestration_failures - AssertionError: Should have some successful extractions
assert 0 > 5
 +  where 0 = len([])
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestResourceExhaustionRecovery::test_cpu_exhaustion_timeout_handling - RuntimeError: super(): no arguments
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestResourceExhaustionRecovery::test_disk_space_exhaustion_fallback - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestNetworkAndDatabaseFailureRecovery::test_database_connection_failure_with_exponential_backoff - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_error_recovery_fault_tolerance.py::TestNetworkAndDatabaseFailureRecovery::test_database_corruption_recovery - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_temporal_features_with_significant_data_gaps - AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 3540.0, 'avg_transition_interval': 1006.1538461538462, ...}
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_sequential_features_with_incomplete_room_sequences - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room bedroom | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=bedroom | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_contextual_features_with_missing_environmental_sensors - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_feature_extraction_with_corrupted_timestamps - src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room office | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=office | Caused by: TypeError: '<' not supported between instances of 'str' and 'NoneType'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_feature_extraction_with_malformed_attributes - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_feature_store_with_database_connection_failures - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestMissingSensorDataScenarios::test_feature_engineering_with_partial_extractor_failures - TypeError: argument of type 'coroutine' is not iterable
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_with_single_sensor_type - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room single_sensor_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=single_sensor_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_with_rapid_state_changes - AssertionError: assert 'transition_rate' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 1.0, 'avg_transition_interval': 1.0, ...}
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_with_extreme_temporal_ranges - AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.2245359891353553, 'avg_off_duration': 1800.0, 'avg_on_duration': 604800.0, 'avg_transition_interval': 604800.0, ...}
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_memory_constraints - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_missing_data_scenarios.py::TestFeatureValidationEdgeCases::test_feature_extraction_with_duplicate_events - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room duplicate_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=duplicate_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_time_since_calculation_accuracy - assert 2700.0 < 1.0
 +  where 2700.0 = abs((3600.0 - 900.0))
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_duration_mathematical_accuracy - assert 1200.0 < 1.0
 +  where 1200.0 = abs((1800.0 - 600.0))
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_duration_statistical_features_numpy - assert 600.0 < 1.0
 +  where 600.0 = abs((1800.0 - 1200.0))
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_duration_ratio_calculation - assert 1.6666666666666665 < 0.1
 +  where 1.6666666666666665 = abs((0.8333333333333334 - 2.5))
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_duration_percentile_calculations - assert 150.0 < 60.0
 +  where 150.0 = abs((750.0 - 600.0))
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_transition_timing_comprehensive - assert -43.51648351648352 >= 0.0
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_transition_timing_edge_cases - KeyError: 'transition_regularity'
FAILED tests/unit/test_features/test_temporal_comprehensive.py::TestTemporalFeatureExtractorComprehensive::test_error_handling_comprehensive - Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestDaylightSavingTimeTransitions::test_temporal_features_spring_forward_transition - AssertionError: assert 'time_since_last_change' in {'activity_variance': 0.24987654320987648, 'avg_off_duration': 1800.0, 'avg_on_duration': 337.77777777777777, 'avg_transition_interval': 80.11173184357541, ...}
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestDaylightSavingTimeTransitions::test_temporal_features_fall_back_transition - AssertionError: assert 'duration_in_current_state' in {'activity_variance': 0.23040000000000002, 'avg_off_duration': 1800.0, 'avg_on_duration': 47.64705882352941, 'avg_transition_interval': 35.83892617449664, ...}
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestDaylightSavingTimeTransitions::test_sequential_features_across_dst_boundaries - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room living_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestDaylightSavingTimeTransitions::test_contextual_features_dst_environmental_correlation - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestCrossTimezoneScenarios::test_temporal_features_cross_timezone_correlation - AssertionError: assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 360.0, 'avg_transition_interval': 652.7272727272727, ...}
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestCrossTimezoneScenarios::test_sequential_features_global_room_transitions - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room pst_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=pst_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_timezone_dst_handling.py::TestTimezoneEdgeCasesAndErrorHandling::test_mixed_timezone_aware_naive_events - src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room mixed_tz_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=mixed_tz_room | Caused by: TypeError: can't compare offset-naive and offset-aware datetimes
FAILED tests/unit/test_ingestion/test_event_processor.py::TestEventProcessor::test_check_room_state_change_presence_sensor - AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.
FAILED tests/unit/test_ingestion/test_event_processor.py::TestEventProcessor::test_check_room_state_change_motion_sensor - AssertionError: Expected 'handle_room_state_change' to be called once. Called 0 times.
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_get_bulk_history_with_rate_limit_handling - AttributeError: 'RateLimitExceededError' object has no attribute 'reset_time'
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_validate_and_normalize_state_edge_cases - AssertionError: assert 'on' == 'off'
  - off
  + on
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_handle_websocket_messages_json_error - AssertionError: expected call not found.
Expected: warning('Received invalid JSON: invalid json')
  Actual: not called.
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_reconnect_with_exponential_backoff - AssertionError: expected call not found.
Expected: sleep(4)
  Actual: sleep(8)

pytest introspection follows:

Args:
assert (8,) == (4,)
  At index 0 diff: 8 != 4
  Full diff:
  - (4,)
  ?  ^
  + (8,)
  ?  ^
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_reconnect_delay_capped - AssertionError: expected call not found.
Expected: sleep(300)
  Actual: not called.
FAILED tests/unit/test_ingestion/test_ha_client.py::TestHomeAssistantClientIntegration::test_reconnect_failure_triggers_retry - AssertionError: Expected 'create_task' to have been called once. Called 0 times.
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_root_endpoint - KeyError: 'status'
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_health_component_endpoint - assert 400 == 200
 +  where 400 = <Response [400 Bad Request]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_health_component_endpoint_not_found - assert 400 == 404
 +  where 400 = <Response [400 Bad Request]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_predictions_room - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_predictions_room_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_predictions_all - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_accuracy_metrics - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_accuracy_metrics_with_params - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_trigger_manual_retrain - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_refresh_mqtt_discovery - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEndpoints::test_get_system_stats - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_active_incidents - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_incident_details - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_incident_details_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_incident_history - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_incident_history_invalid_hours - assert 401 == 400
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_get_incident_statistics - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_acknowledge_incident - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_acknowledge_incident_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_resolve_incident - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_resolve_incident_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_start_incident_response - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestIncidentEndpoints::test_stop_incident_response - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestErrorHandling::test_api_error_handler - assert 401 == 400
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestErrorHandling::test_system_error_handler - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestErrorHandling::test_general_exception_handler - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestErrorHandling::test_health_endpoint_exception - assert 200 == 500
 +  where 200 = <Response [200 OK]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestErrorHandling::test_validation_error - assert 401 == 422
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestResponseModels::test_manual_retrain_request_validation - pydantic_core._pydantic_core.ValidationError: 1 validation error for ManualRetrainRequest
room_id
  Value error, Room 'living_room' not found in configuration [type=value_error, input_value='living_room', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_api_server.py::TestAPIPerformance::test_multiple_prediction_requests - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIPerformance::test_large_stats_response - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_malformed_json_request - assert 401 == 422
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_missing_required_fields - assert 401 == 422
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_invalid_query_parameters - assert 401 == 422
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_oversized_request - assert 401 in [200, 413, 422]
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_special_characters_in_room_id - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_long_incident_id - assert 401 in [404, 414]
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_concurrent_retrain_requests - assert False
 +  where False = all(<generator object TestAPIEdgeCases.test_concurrent_retrain_requests.<locals>.<genexpr> at 0x7fec86295630>)
FAILED tests/unit/test_integration/test_api_server.py::TestAPIEdgeCases::test_api_with_no_tracking_manager - assert 429 in [404, 500]
 +  where 429 = <Response [429 Too Many Requests]>.status_code
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestGetJWTManager::test_get_jwt_manager_jwt_disabled - Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestGetJWTManager::test_get_jwt_manager_singleton_behavior - AssertionError: Expected 'JWTManager' to have been called once. Called 0 times.
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermission::test_require_permission_success - AttributeError: 'Depends' object has no attribute 'has_permission'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermission::test_require_permission_admin_bypass - AttributeError: 'Depends' object has no attribute 'has_permission'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermission::test_require_permission_failure - AttributeError: 'Depends' object has no attribute 'has_permission'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequireRole::test_require_role_success - AttributeError: 'Depends' object has no attribute 'has_role'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequireRole::test_require_role_failure - AttributeError: 'Depends' object has no attribute 'has_role'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequireAdmin::test_require_admin_success - AttributeError: 'Depends' object has no attribute 'is_admin'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequireAdmin::test_require_admin_failure - AttributeError: 'Depends' object has no attribute 'is_admin'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermissions::test_require_permissions_all_success - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermissions::test_require_permissions_all_failure - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermissions::test_require_permissions_any_success - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermissions::test_require_permissions_any_failure - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestRequirePermissions::test_require_permissions_admin_bypass - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestGetRequestContext::test_get_request_context_without_user - AssertionError: assert None == 'req_456'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesIntegration::test_dependency_chain_success - AttributeError: 'Depends' object has no attribute 'has_permission'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesIntegration::test_dependency_chain_multiple_requirements - AttributeError: 'Depends' object has no attribute 'permissions'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesIntegration::test_dependency_failure_propagation - AttributeError: 'Depends' object has no attribute 'is_admin'
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesErrorHandling::test_jwt_manager_initialization_error - Failed: DID NOT RAISE <class 'Exception'>
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesPerformance::test_jwt_manager_caching - assert 0 == 1
FAILED tests/unit/test_integration/test_auth_dependencies.py::TestDependenciesPerformance::test_permission_validation_efficiency - AttributeError: 'Depends' object has no attribute 'has_permission'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthenticationRouter::test_auth_router_routes_registration - AssertionError: assert '/users/{user_id}' in {'/auth/change-password', '/auth/login', '/auth/logout', '/auth/me', '/auth/refresh', '/auth/token/info', ...}
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_login_endpoint_success - assert 500 == 200
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_login_endpoint_invalid_username - assert 422 == 401
 +  where 422 = <Response [422 Unprocessable Entity]>.status_code
 +  and   401 = status.HTTP_401_UNAUTHORIZED
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_login_endpoint_invalid_password - assert 422 == 401
 +  where 422 = <Response [422 Unprocessable Entity]>.status_code
 +  and   401 = status.HTTP_401_UNAUTHORIZED
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_login_endpoint_remember_me - assert 500 == 200
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_refresh_token_endpoint_success - assert 500 == 200
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_refresh_token_endpoint_invalid_token - assert 500 == 401
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   401 = status.HTTP_401_UNAUTHORIZED
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_logout_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_logout_endpoint_no_token - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_me_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_change_password_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_change_password_endpoint_wrong_current - assert 401 == 400
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   400 = status.HTTP_400_BAD_REQUEST
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_change_password_endpoint_user_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   404 = status.HTTP_404_NOT_FOUND
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_token_info_endpoint_success - assert 500 == 200
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthRouterApp::test_token_info_endpoint_invalid_token - assert 500 == 400
 +  where 500 = <Response [500 Internal Server Error]>.status_code
 +  and   400 = status.HTTP_400_BAD_REQUEST
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLoginEndpoint::test_login_success - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLoginEndpoint::test_login_invalid_credentials - pydantic_core._pydantic_core.ValidationError: 1 validation error for LoginRequest
password
  Value error, Password must contain at least 3 of: uppercase, lowercase, digit, special character [type=value_error, input_value='wrongpassword', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLoginEndpoint::test_login_inactive_user - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLoginEndpoint::test_login_remember_me_extended_expiration - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLoginEndpoint::test_login_service_error - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLogoutEndpoint::test_logout_success - fastapi.exceptions.HTTPException: 500: Logout service unavailable
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLogoutEndpoint::test_logout_revoke_all_tokens - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLogoutEndpoint::test_logout_no_refresh_token - AssertionError: assert 'Logout successful' == 'Successfully logged out'
  - Successfully logged out
  + Logout successful
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestLogoutEndpoint::test_logout_revoke_failure - fastapi.exceptions.HTTPException: 500: Logout service unavailable
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestRefreshTokenEndpoint::test_refresh_token_success - pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
refresh_token
  Value error, Invalid JWT token format [type=value_error, input_value='valid_refresh_token', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestRefreshTokenEndpoint::test_refresh_token_invalid_token - pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
refresh_token
  Value error, Invalid JWT token format [type=value_error, input_value='invalid_token', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestRefreshTokenEndpoint::test_refresh_token_service_error - pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
refresh_token
  Value error, Invalid refresh token format [type=value_error, input_value='token', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestPasswordChangeEndpoint::test_change_password_success - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestPasswordChangeEndpoint::test_change_password_invalid_current - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestPasswordChangeEndpoint::test_change_password_service_error - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_create_user_success - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_create_user_duplicate_username - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_list_users_success - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_list_users_with_inactive - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_delete_user_success - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_delete_user_not_found - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestUserManagementEndpoints::test_delete_self_prevention - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_list_users_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_create_user_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_create_user_endpoint_duplicate_username - assert 401 == 400
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   400 = status.HTTP_400_BAD_REQUEST
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_create_user_endpoint_validation_errors - assert 401 == 422
 +  where 401 = <Response [401 Unauthorized]>.status_code
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_delete_user_endpoint_success - assert 401 == 200
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   200 = status.HTTP_200_OK
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_delete_user_endpoint_self_deletion - assert 401 == 400
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   400 = status.HTTP_400_BAD_REQUEST
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAdminEndpoints::test_delete_user_endpoint_user_not_found - assert 401 == 404
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   404 = status.HTTP_404_NOT_FOUND
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthEndpointIntegration::test_full_authentication_flow - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthEndpointErrorHandling::test_login_unexpected_error - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthEndpointErrorHandling::test_refresh_unexpected_error - pydantic_core._pydantic_core.ValidationError: 1 validation error for RefreshRequest
refresh_token
  Value error, Invalid refresh token format [type=value_error, input_value='token', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthEndpointErrorHandling::test_password_change_unexpected_error - AttributeError: <module 'src.integration.auth.endpoints' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/auth/endpoints.py'> does not have the attribute 'get_user_service'
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestEndpointErrorHandling::test_logout_endpoint_service_error - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestEndpointErrorHandling::test_change_password_service_error - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestEndpointErrorHandling::test_create_user_service_error - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestEndpointErrorHandling::test_delete_user_service_error - assert 401 == 500
 +  where 401 = <Response [401 Unauthorized]>.status_code
 +  and   500 = status.HTTP_500_INTERNAL_SERVER_ERROR
FAILED tests/unit/test_integration/test_auth_endpoints.py::TestAuthEndpointSecurity::test_sensitive_data_not_logged - assert 'TestPass123!' not in "username='t...ber_me=False"
  'TestPass123!' is contained here:
    username='testuser' password='TestPass123!' remember_me=False
  ?                               ++++++++++++
FAILED tests/unit/test_features/test_performance.py::TestFeaturePerformanceBenchmarks::test_feature_engineering_batch_performance - TypeError: FeatureEngineeringEngine.extract_batch_features() got an unexpected keyword argument 'parallel'
FAILED tests/unit/test_features/test_performance.py::TestFeaturePerformanceBenchmarks::test_memory_efficiency_with_cache_cleanup - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_performance.py::TestFeaturePerformanceBenchmarks::test_concurrent_feature_store_access - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_performance.py::TestFeaturePerformanceBenchmarks::test_feature_extraction_scalability - Failed: Timeout (>30.0s) from pytest-timeout.
FAILED tests/unit/test_features/test_performance.py::TestFeatureMissingDataHandling::test_temporal_features_with_sensor_gaps - AssertionError: Should calculate time since last change
assert 'time_since_last_change' in {'activity_variance': 0.24360000000000004, 'avg_off_duration': 1800.0, 'avg_on_duration': 757.5903614457832, 'avg_transition_interval': 385.32663316582915, ...}
FAILED tests/unit/test_features/test_performance.py::TestFeatureMissingDataHandling::test_sequential_features_with_incomplete_sequences - src.core.exceptions.FeatureExtractionError: Feature extraction failed: sequential for room living_room | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=sequential, room_id=living_room | Caused by: TypeError: unsupported operand type(s) for -: 'dict' and 'datetime.timedelta'
FAILED tests/unit/test_features/test_performance.py::TestFeatureMissingDataHandling::test_contextual_features_with_missing_environmental_data - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_performance.py::TestFeatureMissingDataHandling::test_feature_store_with_database_unavailable - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_performance.py::TestFeatureMissingDataHandling::test_feature_extraction_with_corrupted_events - src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140664671959568'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140664671959568'> | Caused by: TypeError: '<' not supported between instances of 'NoneType' and 'datetime.datetime'
FAILED tests/unit/test_features/test_performance.py::TestFeatureTimezoneTransitions::test_temporal_features_across_dst_spring_forward - AssertionError: Should include cyclical time features
assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 300.0, 'avg_transition_interval': 60.0, ...}
FAILED tests/unit/test_features/test_performance.py::TestFeatureTimezoneTransitions::test_feature_extraction_timezone_changes - AssertionError: Should recalculate cyclical features
assert 'cyclical_hour_sin' in {'activity_variance': 0.25, 'avg_off_duration': 1800.0, 'avg_on_duration': 180.0, 'avg_transition_interval': 60.0, ...}
FAILED tests/unit/test_features/test_performance.py::TestFeatureTimezoneTransitions::test_cross_timezone_room_correlation - TypeError: ContextualFeatureExtractor.extract_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/test_features/test_performance.py::TestFeatureCacheInvalidationScenarios::test_feature_cache_with_memory_pressure - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_performance.py::TestFeatureCacheInvalidationScenarios::test_cache_invalidation_on_time_expiry - TypeError: FeatureCache.put() got an unexpected keyword argument 'max_age_seconds'
FAILED tests/unit/test_features/test_performance.py::TestFeatureCacheInvalidationScenarios::test_concurrent_cache_access_and_invalidation - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_performance.py::TestFeatureCacheInvalidationScenarios::test_feature_store_cache_coherence - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_performance.py::TestFeatureCacheInvalidationScenarios::test_memory_leak_prevention_in_cache - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/test_features/test_performance.py::TestFeatureErrorRecoveryAndResilience::test_temporal_feature_extraction_with_partial_failures - src.core.exceptions.FeatureExtractionError: Feature extraction failed: temporal for room <Mock name='mock.room_id' id='140665156762800'> | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=temporal, room_id=<Mock name='mock.room_id' id='140665156762800'> | Caused by: Exception: Cyclical feature extraction failed
FAILED tests/unit/test_features/test_performance.py::TestFeatureErrorRecoveryAndResilience::test_feature_engineering_graceful_degradation - AssertionError: Should get expected error message
assert 'temporarily unavailable' in "object of type 'coroutine' has no len()"
 +  where "object of type 'coroutine' has no len()" = str(TypeError("object of type 'coroutine' has no len()"))
FAILED tests/unit/test_features/test_performance.py::TestFeatureErrorRecoveryAndResilience::test_feature_store_database_reconnection - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'db_manager'
FAILED tests/unit/test_features/test_performance.py::TestFeatureErrorRecoveryAndResilience::test_concurrent_feature_extraction_with_failures - AssertionError: Expected tasks should fail
assert [8, 12, 4, 0, 16] == [0, 4, 8, 12, 16]
  At index 0 diff: 8 != 0
  Full diff:
  - [0, 4, 8, 12, 16]
  + [8, 12, 4, 0, 16]
FAILED tests/unit/test_features/test_sequential_comprehensive.py::TestSequentialFeatureExtractorComprehensive::test_room_dwell_time_calculation - assert 225.0 < 60
 +  where 225.0 = abs((675.0 - 900.0))
FAILED tests/unit/test_features/test_sequential_comprehensive.py::TestSequentialFeatureExtractorComprehensive::test_max_sequence_length_detection - assert 1 == 4.0
FAILED tests/unit/test_features/test_sequential_comprehensive.py::TestSequentialFeatureExtractorComprehensive::test_room_transition_edge_cases - AssertionError: assert 1 == 5
 +  where 5 = len([<Mock spec='SensorEvent' id='140664667802320'>, <Mock spec='SensorEvent' id='140664667813072'>, <Mock spec='SensorEvent' id='140664674032944'>, <Mock spec='SensorEvent' id='140664674038656'>, <Mock spec='SensorEvent' id='140664674041008'>])
FAILED tests/unit/test_features/test_sequential_comprehensive.py::TestSequentialFeatureExtractorComprehensive::test_velocity_acceleration_calculation - assert 0.0 > 0
FAILED tests/unit/test_features/test_sequential_comprehensive.py::TestSequentialFeatureExtractorComprehensive::test_error_handling_comprehensive - Failed: DID NOT RAISE <class 'src.core.exceptions.FeatureExtractionError'>
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorTimeSinceFeatures::test_extract_time_since_features_basic - assert 3600.0 == 1200.0
FAILED tests/unit/test_features/test_temporal.py::TestTemporalFeatureExtractorTimeSinceFeatures::test_extract_time_since_features_motion_sensor - assert 1200.0 == 1800.0
ERROR tests/unit/test_adaptation/test_tracking_manager.py
ERROR tests/unit/test_core/test_config_validator_comprehensive.py
ERROR tests/unit/test_utils/test_metrics_comprehensive.py
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_tracker_initialization - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_record_validation_basic - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_calculate_window_accuracy - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_detect_trend - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_alert_generation - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAccuracyTracker::test_get_real_time_metrics - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'room_id'
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptiveRetrainer::test_retrainer_initialization - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_accuracy_degradation - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_insufficient_samples - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptiveRetrainer::test_request_retraining - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptiveRetrainer::test_perform_retraining - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold'
ERROR tests/unit/test_adaptation_consolidated.py::TestModelOptimizer::test_optimizer_initialization - TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
ERROR tests/unit/test_adaptation_consolidated.py::TestModelOptimizer::test_optimize_hyperparameters - TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
ERROR tests/unit/test_adaptation_consolidated.py::TestModelOptimizer::test_generate_hyperparameter_combinations - TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'max_trials'
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_tracking_manager_initialization - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_start_monitoring - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_stop_monitoring - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_record_prediction_integration - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_validate_prediction_integration - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_monitoring_loop_integration - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_get_system_status - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_error_handling - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestTrackingManager::test_resource_cleanup - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptationIntegrationScenarios::test_full_adaptation_workflow - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptationIntegrationScenarios::test_concurrent_monitoring_operations - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptationIntegrationScenarios::test_error_propagation - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation_consolidated.py::TestAdaptationIntegrationScenarios::test_recovery_from_component_failures - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'auto_retraining_enabled'. Did you mean 'adaptive_retraining_enabled'?
ERROR tests/unit/test_adaptation/test_drift_detector_comprehensive.py::TestConceptDriftDetectorStatistical::test_detect_drift_complete_analysis - TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'prediction_count'
ERROR tests/unit/test_adaptation/test_drift_detector_comprehensive.py::TestConceptDriftDetectorStatistical::test_prediction_drift_analysis_mathematical_accuracy - TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'prediction_count'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_success - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_type'. Did you mean 'predicted_time'?
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::test_full_prediction_workflow
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::test_system_lifecycle_with_monitoring
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::test_error_scenarios_with_alerts
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestIntegrationScenarios::test_monitoring_integration_coverage
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::test_prediction_with_string_model_type
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::test_validation_with_non_dict_result
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::test_concept_drift_without_tracking_manager_support
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestEdgeCases::test_context_manager_exception_handling
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestPerformanceAndStress::test_high_volume_predictions
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking.py::TestPerformanceAndStress::test_rapid_metric_recording
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_success - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_with_kwargs - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitored_record_prediction_string_model_type - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_prediction_monitoring_timeout_handling - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_concurrent_prediction_recording - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_monitoring_enhanced_tracking_comprehensive.py::TestMonitoringEnhancedTrackingManager::test_monitoring_context_manager_error - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_initialization - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_initialize_and_shutdown - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_accuracy_trigger - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_error_trigger - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_drift_trigger - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_confidence_trigger - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_no_triggers - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_evaluate_retraining_need_cooldown - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_select_retraining_strategy_incremental - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_select_retraining_strategy_full_retrain - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_select_retraining_strategy_ensemble_rebalance - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_request_manual_retraining - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_get_retraining_status_specific_request - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_get_retraining_status_all_requests - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_cancel_pending_retraining - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_get_retrainer_stats - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_queue_retraining_request_priority_ordering - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestAdaptiveRetrainer::test_queue_retraining_request_duplicate_handling - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_incremental_retrain_success - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_incremental_retrain_fallback_to_full - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_full_retrain_with_optimization - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_ensemble_rebalance - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_ensemble_rebalance_fallback - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_prepare_retraining_data - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_extract_features_for_retraining - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_retrain_model_with_optimization - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_retrain_model_missing_from_registry - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingExecution::test_retrain_model_insufficient_data - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingProgress::test_progress_tracking_during_retraining - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingProgress::test_get_retraining_progress - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestRetrainingProgress::test_execute_retraining_with_progress_tracking - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestIntegrationAndValidation::test_drift_detector_integration - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestIntegrationAndValidation::test_prediction_validator_integration - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestIntegrationAndValidation::test_model_optimizer_integration_status - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestIntegrationAndValidation::test_drift_detector_status - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestIntegrationAndValidation::test_prediction_validator_status - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestConcurrencyAndResourceManagement::test_concurrent_request_limit - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestConcurrencyAndResourceManagement::test_multiple_concurrent_requests - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestConcurrencyAndResourceManagement::test_resource_tracking_accuracy - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestConcurrencyAndResourceManagement::test_cooldown_tracking_thread_safety - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_retraining_execution_failure - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_model_training_failure - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_validation_failure_handling - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_malformed_retraining_request - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_invalid_model_type_handling - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_optimizer_failure_graceful_handling - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestErrorHandlingAndEdgeCases::test_memory_cleanup_on_shutdown - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestPerformanceAndScalability::test_high_volume_request_handling - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestPerformanceAndScalability::test_memory_efficiency_large_history - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_adaptation/test_retrainer_comprehensive.py::TestPerformanceAndScalability::test_concurrent_status_queries - TypeError: OptimizationResult.__init__() missing 2 required positional arguments: 'total_evaluations' and 'convergence_achieved'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_event_validator_comprehensive_initialization - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_all_sensor_states - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_invalid_states_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_state_transition_matrix - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_timestamp_edge_cases - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_room_sensor_configuration_matching - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_sensor_state_enum_compliance - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_confidence_score_calculation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventValidatorAdvanced::test_validate_event_performance_timing - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_classifier_initialization_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_movement_metrics_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_max_velocity_complex_patterns - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_door_interactions_multiple_doors - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_presence_sensor_ratio_complex_config - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_sensor_revisits_complex_patterns - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_avg_dwell_time_mathematical_precision - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_timing_variance_statistical_accuracy - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_movement_entropy_information_theory - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_calculate_spatial_dispersion_advanced_geometry - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_score_human_pattern_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_score_cat_pattern_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_classify_movement_comprehensive_human - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_classify_movement_comprehensive_cat - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_analyze_sequence_patterns_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_get_sequence_time_analysis_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_extract_movement_signature_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestMovementPatternClassifierAdvanced::test_compare_movement_patterns_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_event_processor_comprehensive_initialization - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_process_event_comprehensive_flow - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_process_event_with_sequence_classification - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_process_event_batch_performance - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_process_event_duplicate_detection_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_check_room_state_change_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_check_room_state_change_motion_patterns - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_validate_event_sequence_integrity_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_validate_event_sequence_integrity_with_anomalies - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_validate_room_configuration_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_validate_room_configuration_edge_cases - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_processing_stats_management - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_determine_sensor_type_comprehensive - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorAdvanced::test_movement_sequence_creation_edge_cases - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorIntegrationAdvanced::test_end_to_end_event_processing_workflow - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorIntegrationAdvanced::test_concurrent_event_processing - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorIntegrationAdvanced::test_high_volume_processing_stress - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorPerformance::test_validator_performance_benchmark - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_processor_comprehensive.py::TestEventProcessorPerformance::test_classifier_performance_benchmark - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_valid_sensor_event_schema - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_missing_required_fields - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_null_required_fields - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_invalid_field_types - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_valid_sensor_types_and_states - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_timestamp_format_validation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_timezone_awareness - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSchemaValidator::test_room_configuration_validation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestPerformanceValidator::test_bulk_validation_small_dataset - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestPerformanceValidator::test_bulk_validation_large_dataset - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestPerformanceValidator::test_single_event_validation_performance - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestPerformanceValidator::test_validation_with_errors_and_warnings - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_initialization - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_validation_rules_initialization - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_single_event_validation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_single_event_validation_with_security_issues - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_bulk_events_validation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_bulk_validation_with_duplicates - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_room_specific_validation - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_event_data_sanitization - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_validation_summary - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_validation_error_context_and_suggestions - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_concurrent_validation_safety - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestComprehensiveEventValidator::test_large_scale_performance - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_comprehensive_sql_injection_prevention - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_comprehensive_xss_prevention - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_path_traversal_prevention - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_sanitization_effectiveness - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_bypass_attempt_prevention - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_event_validator_comprehensive.py::TestSecurityTestingIntegration::test_false_positive_prevention - TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::test_initialize_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::test_initialize_already_initialized
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerInitialization::test_initialize_error_handling
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::test_create_engine_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::test_create_engine_error
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::test_create_engine_connection_pooling_config
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerEngineCreation::test_create_engine_event_listeners
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionFactory::test_setup_session_factory
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionFactory::test_setup_session_factory_no_engine
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::test_verify_connection_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::test_verify_connection_failure
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionVerification::test_verify_connection_timeout
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_failure
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_no_engine
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_stats_update
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_loop
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerHealthCheck::test_health_check_loop_error_recovery
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::test_get_session_context_manager
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::test_get_session_no_factory
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::test_session_error_handling
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSessionManagement::test_concurrent_sessions
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::test_calculate_retry_delay
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::test_retry_with_exponential_backoff
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::test_retry_max_retries_exceeded
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerRetryLogic::test_retry_non_retryable_error
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::test_stats_initialization
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::test_stats_update_on_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::test_stats_update_on_failure
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::test_success_rate_calculation
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerConnectionStatistics::test_stats_reset
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::test_cleanup_success
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::test_cleanup_no_active_components
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::test_cleanup_task_cancellation_error
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerCleanup::test_cleanup_engine_disposal_error
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::test_connection_string_sanitization
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::test_query_timeout_enforcement
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::test_connection_pool_security
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerSecurityAndTimeout::test_sql_injection_prevention
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::test_connection_recovery_after_disconnect
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::test_engine_recreation_on_failure
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerErrorRecovery::test_graceful_degradation
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::test_connection_pool_metrics
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::test_query_performance_tracking
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerPerformanceMonitoring::test_health_check_response_time
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerIntegrationScenarios::test_full_lifecycle - TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerIntegrationScenarios::test_concurrent_operations - TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
ERROR tests/unit/test_data/test_database_comprehensive.py::TestDatabaseManagerIntegrationScenarios::test_error_recovery_scenario - TypeError: DatabaseConfig.__init__() got an unexpected keyword argument 'query_timeout'
!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 593 failures !!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! xdist.dsession.Interrupted: stopping after 400 failures !!!!!!!!!!!!
= 370 failed, 1615 passed, 55 skipped, 462 warnings, 223 errors in 255.08s (0:04:15) =
Error: Process completed with exit code 2.