==================================== ERRORS ====================================
_________ ERROR collecting tests/unit/test_adaptation_consolidated.py __________
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation_consolidated.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_adaptation_consolidated.py:43: in <module>
    from src.adaptation.optimizer import (
E   ImportError: cannot import name 'HyperparameterSpace' from 'src.adaptation.optimizer' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/adaptation/optimizer.py)
_____ ERROR collecting tests/unit/test_adaptation/test_tracking_manager.py _____
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracking_manager.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_adaptation/test_tracking_manager.py:21: in <module>
    from src.adaptation.retrainer import AdaptiveRetrainer, RetrainerError
E   ImportError: cannot import name 'RetrainerError' from 'src.adaptation.retrainer' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/adaptation/retrainer.py)
_ ERROR at setup of TestErrorHandling.test_get_real_time_metrics_error_handling _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracker.py, line 770
      async def test_get_real_time_metrics_error_handling(self, mock_validator):
          """Test error handling in get_real_time_metrics."""
          mock_validator.get_accuracy_metrics = AsyncMock(
              side_effect=Exception("Database error")
          )
          tracker = AccuracyTracker(prediction_validator=mock_validator)
          with pytest.raises(AccuracyTrackingError):
              await tracker.get_real_time_metrics(room_id="test_room")
E       fixture 'mock_validator' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_tracker.py::<event_loop>, tests/unit/test_adaptation/test_tracker.py::TestErrorHandling::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracker.py:770
__ ERROR at setup of TestErrorHandling.test_get_active_alerts_error_handling ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracker.py, line 781
      async def test_get_active_alerts_error_handling(self, mock_validator):
          """Test error handling in get_active_alerts."""
          tracker = AccuracyTracker(prediction_validator=mock_validator)
          # Simulate internal error by corrupting alerts list
          tracker._active_alerts = None
          with pytest.raises(AccuracyTrackingError):
              await tracker.get_active_alerts()
E       fixture 'mock_validator' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_adaptation/test_tracker.py::<event_loop>, tests/unit/test_adaptation/test_tracker.py::TestErrorHandling::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_adaptation/test_tracker.py:781
_____ ERROR at setup of TestSchemaValidator.test_valid_sensor_event_schema _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
______ ERROR at setup of TestSchemaValidator.test_missing_required_fields ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
________ ERROR at setup of TestSchemaValidator.test_invalid_field_types ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
________ ERROR at setup of TestSchemaValidator.test_invalid_sensor_type ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_______ ERROR at setup of TestSchemaValidator.test_invalid_sensor_state ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_____ ERROR at setup of TestSchemaValidator.test_invalid_timestamp_format ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
___ ERROR at setup of TestSchemaValidator.test_room_configuration_validation ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:206: in setup_method
    self.config = SystemConfig(
E   TypeError: SystemConfig.__init__() missing 2 required positional arguments: 'tracking' and 'api'
_ ERROR at setup of TestEnhancedIntegrationManagerIntegration.test_full_integration_workflow _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_enhanced_integration_manager.py, line 1070
      @pytest.mark.asyncio
      async def test_full_integration_workflow(self, mock_config):
          """Test complete integration workflow."""
          # Create mock managers
          mock_mqtt_manager = AsyncMock()
          mock_mqtt_manager.stats = MagicMock()
          mock_mqtt_manager.stats.initialized = True
          mock_mqtt_manager.discovery_publisher = AsyncMock()
          mock_tracking_manager = AsyncMock()
          with patch(
              "src.integration.enhanced_integration_manager.get_config",
              return_value=mock_config,
          ), patch(
              "src.integration.enhanced_integration_manager.HAEntityDefinitions"
          ) as mock_ha_definitions_class:
              mock_ha_definitions = AsyncMock()
              mock_ha_definitions.define_all_entities.return_value = {
                  "entity1": MagicMock()
              }
              mock_ha_definitions.define_all_services.return_value = {
                  "service1": MagicMock()
              }
              mock_ha_definitions.publish_all_entities.return_value = {
                  "entity1": MagicMock(success=True)
              }
              mock_ha_definitions.publish_all_services.return_value = {
                  "service1": MagicMock(success=True)
              }
              mock_ha_definitions.get_entity_stats.return_value = {}
              mock_ha_definitions_class.return_value = mock_ha_definitions
              # Create and initialize manager
              manager = EnhancedIntegrationManager(
                  mqtt_integration_manager=mock_mqtt_manager,
                  tracking_manager=mock_tracking_manager,
              )
              # Test full lifecycle
              await manager.initialize()
              assert manager._enhanced_integration_active is True
              # Test entity state update
              mock_entity_config = MagicMock()
              mock_entity_config.state_topic = "test/state"
              mock_ha_definitions.get_entity_definition.return_value = mock_entity_config
              mock_publish_result = MagicMock()
              mock_publish_result.success = True
              mock_mqtt_manager.mqtt_publisher.publish_json.return_value = (
                  mock_publish_result
              )
              result = await manager.update_entity_state("test_entity", "test_state")
              assert result is True
              # Test command processing
              manager.command_handlers["test_cmd"] = AsyncMock(
                  return_value={"success": True}
              )
              request = CommandRequest(
                  command="test_cmd", parameters={}, timestamp=datetime.utcnow()
              )
              response = await manager.process_command(request)
              assert response.success is True
              # Test shutdown
              await manager.shutdown()
              assert manager._enhanced_integration_active is False
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_enhanced_integration_manager.py::<event_loop>, tests/unit/test_integration/test_enhanced_integration_manager.py::TestEnhancedIntegrationManagerIntegrati
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_enhanced_integration_manager.py:1070
_ ERROR at setup of TestEnhancedIntegrationManagerIntegration.test_background_task_lifecycle _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_enhanced_integration_manager.py, line 1142
      @pytest.mark.asyncio
      async def test_background_task_lifecycle(self, mock_config):
          """Test background task lifecycle management."""
          mock_mqtt_manager = AsyncMock()
          mock_mqtt_manager.stats = MagicMock()
          mock_mqtt_manager.stats.initialized = True
          with patch(
              "src.integration.enhanced_integration_manager.get_config",
              return_value=mock_config,
          ), patch("asyncio.create_task") as mock_create_task:
              mock_task1 = AsyncMock()
              mock_task2 = AsyncMock()
              mock_task1.done.return_value = False
              mock_task2.done.return_value = False
              mock_create_task.side_effect = [mock_task1, mock_task2]
              manager = EnhancedIntegrationManager(
                  mqtt_integration_manager=mock_mqtt_manager
              )
              # Initialize should start background tasks
              await manager.initialize()
              assert len(manager._background_tasks) == 2
              # Shutdown should clean up tasks
              await manager.shutdown()
              mock_task1.cancel.assert_called_once()
              mock_task2.cancel.assert_called_once()
E       fixture 'mock_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_enhanced_integration_manager.py::<event_loop>, tests/unit/test_integration/test_enhanced_integration_manager.py::TestEnhancedIntegrationManagerIntegrati
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_enhanced_integration_manager.py:1142
__ ERROR at setup of TestPredictionPublishing.test_publish_prediction_success __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_mqtt_integration_manager.py:208: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'room_id'
__ ERROR at setup of TestPredictionPublishing.test_publish_prediction_failure __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_mqtt_integration_manager.py:208: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'room_id'
_ ERROR at setup of TestPredictionPublishing.test_publish_prediction_not_active _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_mqtt_integration_manager.py:208: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'room_id'
_ ERROR at setup of TestPredictionPublishing.test_publish_prediction_no_publisher _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_mqtt_integration_manager.py:208: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'room_id'
_ ERROR at setup of TestPredictionPublishing.test_publish_prediction_exception _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_mqtt_integration_manager.py:208: in prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'room_id'
__ ERROR at setup of TestRealtimePublishingSystem.test_system_initialization ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
____ ERROR at setup of TestRealtimePublishingSystem.test_initialize_system _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_____ ERROR at setup of TestRealtimePublishingSystem.test_shutdown_system ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_ ERROR at setup of TestRealtimePublishingSystem.test_publish_prediction_all_channels _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_ ERROR at setup of TestRealtimePublishingSystem.test_publish_prediction_mqtt_only _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_ ERROR at setup of TestRealtimePublishingSystem.test_publish_prediction_mqtt_failure _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
__ ERROR at setup of TestRealtimePublishingSystem.test_publish_system_status ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_ ERROR at setup of TestRealtimePublishingSystem.test_handle_websocket_connection _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
__ ERROR at setup of TestRealtimePublishingSystem.test_format_prediction_data __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_____ ERROR at setup of TestRealtimePublishingSystem.test_get_system_stats _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_integration/test_realtime_publisher.py:493: in mock_prediction_publisher
    return_value=MQTTPublishResult(success=True, error_message=None)
E   TypeError: MQTTPublishResult.__init__() missing 3 required positional arguments: 'topic', 'payload_size', and 'publish_time'
_____ ERROR at setup of TestBroadcastCallbacks.test_add_broadcast_callback _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 758
      async def test_add_broadcast_callback(self, system_with_callbacks):
          """Test adding broadcast callbacks."""
          callback_called = False
          received_event = None
          received_results = None
          def test_callback(event, results):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 749
      @pytest.fixture
      def system_with_callbacks(self, mock_mqtt_config, mock_rooms):
E       fixture 'mock_mqtt_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, system_with_callbacks, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_realtime_publisher.py::<event_loop>, tests/unit/test_integration/test_realtime_publisher.py::TestBroadcastCallbacks::<event_loop>
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py:749
____ ERROR at setup of TestBroadcastCallbacks.test_async_broadcast_callback ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 793
      async def test_async_broadcast_callback(self, system_with_callbacks):
          """Test async broadcast callbacks."""
          callback_called = False
          async def async_callback(event, results):
              nonlocal callback_called
              await asyncio.sleep(0.01)  # Simulate async work
              callback_called = True
          # Add async callback
          system_with_callbacks.add_broadcast_callback(async_callback)
          # Create and publish prediction
          prediction_result = PredictionResult(
              predicted_time=datetime.now(timezone.utc) + timedelta(minutes=5),
              confidence=0.70,
              model_type="test",
              features_used=[],
              prediction_horizon_minutes=5,
              raw_predictions={},
          )
          await system_with_callbacks.publish_prediction(prediction_result, "kitchen")
          # Check callback was called
          assert callback_called
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 749
      @pytest.fixture
      def system_with_callbacks(self, mock_mqtt_config, mock_rooms):
E       fixture 'mock_mqtt_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, system_with_callbacks, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_realtime_publisher.py::<event_loop>, tests/unit/test_integration/test_realtime_publisher.py::TestBroadcastCallbacks::<event_loop>
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py:749
____ ERROR at setup of TestBroadcastCallbacks.test_callback_error_handling _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 820
      async def test_callback_error_handling(self, system_with_callbacks):
          """Test error handling in callbacks."""
          def failing_callback(event, results):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 749
      @pytest.fixture
      def system_with_callbacks(self, mock_mqtt_config, mock_rooms):
E       fixture 'mock_mqtt_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, system_with_callbacks, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_realtime_publisher.py::<event_loop>, tests/unit/test_integration/test_realtime_publisher.py::TestBroadcastCallbacks::<event_loop>
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py:749
_______ ERROR at setup of TestErrorHandling.test_broadcast_error_metrics _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py, line 855
      async def test_broadcast_error_metrics(self, mock_mqtt_config, mock_rooms):
          """Test error metrics tracking."""
          system = RealtimePublishingSystem(
              mqtt_config=mock_mqtt_config,
              rooms=mock_rooms,
              enabled_channels=[PublishingChannel.WEBSOCKET],
          )
          # Mock websocket manager to raise exception
          system.websocket_manager.broadcast_to_room = AsyncMock(
              side_effect=Exception("Broadcast failed")
          )
          prediction_result = PredictionResult(
              predicted_time=datetime.now(timezone.utc) + timedelta(minutes=12),
              confidence=0.60,
              model_type="test",
              features_used=[],
              prediction_horizon_minutes=12,
              raw_predictions={},
          )
          # Should handle error gracefully
          results = await system.publish_prediction(prediction_result, "living_room")
          # Check error handling
          assert "websocket" in results
          assert results["websocket"]["success"] is False
          assert system.metrics.channel_errors["websocket"] == 1
E       fixture 'mock_mqtt_config' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_integration/test_realtime_publisher.py::<event_loop>, tests/unit/test_integration/test_realtime_publisher.py::TestErrorHandling::<event_loop>, tmp_path, tmp_path_factory
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_integration/test_realtime_publisher.py:855
_ ERROR at setup of TestBasePredictorPrediction.test_base_predictor_prediction_success _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 490
      @pytest.mark.asyncio
      async def test_base_predictor_prediction_success(
          self, trained_predictor, sample_prediction_features
      ):
          """Test successful prediction."""
          prediction_time = datetime.now(timezone.utc)
          results = await trained_predictor.predict(
              sample_prediction_features, prediction_time, "occupied"
          )
          assert len(results) == 2
          for i, result in enumerate(results):
              assert isinstance(result, PredictionResult)
              assert result.predicted_time > prediction_time
              assert result.transition_type == "occupied_to_vacant"
              assert result.confidence_score == 0.8
              assert result.model_type == ModelType.LSTM.value
              assert result.features_used == ["feature1", "feature2", "feature3"]
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 470
      @pytest.fixture
      def trained_predictor(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_features, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py:
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:470
_ ERROR at setup of TestBasePredictorPrediction.test_base_predictor_prediction_invalid_features _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 522
      @pytest.mark.asyncio
      async def test_base_predictor_prediction_invalid_features(self, trained_predictor):
          """Test prediction with invalid features."""
          invalid_features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
          prediction_time = datetime.now(timezone.utc)
          with pytest.raises(ModelPredictionError):
              await trained_predictor.predict(invalid_features, prediction_time)
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 470
      @pytest.fixture
      def trained_predictor(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_features, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py:
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:470
_ ERROR at setup of TestBasePredictorPrediction.test_base_predictor_predict_single _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 531
      @pytest.mark.asyncio
      async def test_base_predictor_predict_single(self, trained_predictor):
          """Test single prediction from feature dictionary."""
          prediction_time = datetime.now(timezone.utc)
          features_dict = {"feature1": 1.5, "feature2": 0.8, "feature3": -1.1}
          result = await trained_predictor.predict_single(
              features_dict, prediction_time, "vacant"
          )
          assert isinstance(result, PredictionResult)
          assert result.transition_type == "vacant_to_occupied"
          assert result.predicted_time > prediction_time
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 470
      @pytest.fixture
      def trained_predictor(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_prediction_features, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py:
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:470
_ ERROR at setup of TestBasePredictorFeatureValidation.test_validate_features_success _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 581
      def test_validate_features_success(self, trained_predictor_with_features):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 569
      @pytest.fixture
      def trained_predictor_with_features(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorFeatureVal
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:569
_ ERROR at setup of TestBasePredictorFeatureValidation.test_validate_features_missing _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 589
      def test_validate_features_missing(self, trained_predictor_with_features):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 569
      @pytest.fixture
      def trained_predictor_with_features(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorFeatureVal
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:569
_ ERROR at setup of TestBasePredictorFeatureValidation.test_validate_features_extra _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 603
      def test_validate_features_extra(self, trained_predictor_with_features):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 569
      @pytest.fixture
      def trained_predictor_with_features(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorFeatureVal
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:569
_ ERROR at setup of TestBasePredictorFeatureImportance.test_get_feature_importance_trained _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 652
      def test_get_feature_importance_trained(self, trained_predictor_with_importance):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 639
      @pytest.fixture
      def trained_predictor_with_importance(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorFeatureImp
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:639
____ ERROR at setup of TestBasePredictorModelManagement.test_get_model_info ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 699
      def test_get_model_info(self, trained_predictor_for_management):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 677
      @pytest.fixture
      def trained_predictor_for_management(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorModelManag
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:677
_ ERROR at setup of TestBasePredictorModelManagement.test_get_training_history _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 713
      def test_get_training_history(self, trained_predictor_for_management):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 677
      @pytest.fixture
      def trained_predictor_for_management(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorModelManag
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:677
_ ERROR at setup of TestBasePredictorModelManagement.test_get_prediction_accuracy_insufficient_data _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 723
      def test_get_prediction_accuracy_insufficient_data(
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 677
      @pytest.fixture
      def trained_predictor_for_management(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorModelManag
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:677
_ ERROR at setup of TestBasePredictorModelManagement.test_get_prediction_accuracy_sufficient_data _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 732
      def test_get_prediction_accuracy_sufficient_data(
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 677
      @pytest.fixture
      def trained_predictor_for_management(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorModelManag
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:677
_ ERROR at setup of TestBasePredictorModelManagement.test_clear_prediction_history _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 753
      def test_clear_prediction_history(self, trained_predictor_for_management):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 677
      @pytest.fixture
      def trained_predictor_for_management(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorModelManag
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:677
___ ERROR at setup of TestBasePredictorSerialization.test_save_model_success ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 784
      def test_save_model_success(self, trained_predictor_for_serialization):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 767
      @pytest.fixture
      def trained_predictor_for_serialization(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorSerializat
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:767
___ ERROR at setup of TestBasePredictorSerialization.test_save_model_failure ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 800
      def test_save_model_failure(self, trained_predictor_for_serialization):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 767
      @pytest.fixture
      def trained_predictor_for_serialization(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorSerializat
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:767
___ ERROR at setup of TestBasePredictorSerialization.test_load_model_success ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 809
      def test_load_model_success(self, trained_predictor_for_serialization):
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 767
      @pytest.fixture
      def trained_predictor_for_serialization(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorSerializat
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:767
_ ERROR at setup of TestBasePredictorUtilityMethods.test_generate_model_version_increment _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 909
      def test_generate_model_version_increment(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorUtilityMet
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:909
_ ERROR at setup of TestBasePredictorUtilityMethods.test_string_representations_trained _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
file /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py, line 941
      def test_string_representations_trained(self, sample_training_data):
E       fixture 'sample_training_data' not found
>       available fixtures: _session_event_loop, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_background_tasks, cov, doctest_namespace, event_loop, event_loop_policy, extra, extras, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, include_metadata_in_junit_xml, metadata, mock_aiohttp_session, mock_event_processor, mock_ha_client, mock_websocket, mocker, module_mocker, monkeypatch, no_cover, package_mocker, populated_test_db, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_ha_events, sample_sensor_events, session_mocker, test_config_dir, test_db_engine, test_db_manager, test_db_session, test_environment_variables, test_room_config, test_system_config, testrun_uid, tests/unit/test_models/__init__.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::<event_loop>, tests/unit/test_models/test_base_predictor.py::TestBasePredictorUtilityMet
>       use 'pytest --fixtures [testpath]' for help on them.
/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/test_models/test_base_predictor.py:941
=================================== FAILURES ===================================
_ TestConstantValidationInIntegrationScenarios.test_database_schema_integration _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants_integration_advanced.py:618: in test_database_schema_integration
    assert table_name in schema_definitions
E   AssertionError: assert 'room_states' in {'model_accuracy': {'columns': ['id', 'room_id', 'model_type', 'accuracy_score', 'timestamp'], 'primary_key': ['id']}, 'predictions': {'columns': ['id', 'room_id', 'prediction_type', 'predicted_time', 'confidence'], 'primary_key': ['id']}, 'sensor_events': {'columns': ['id', 'room_id', 'sensor_id', 'sensor_type', 'state', 'timestamp'], 'indexes': ['idx_sensor_events_room_time', 'idx_sensor_events_sensor_time'], 'primary_key': ['id', 'timestamp']}}
________ TestConstantValidationPerformance.test_enum_lookup_performance ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants_integration_advanced.py:767: in test_enum_lookup_performance
    assert len(valid_states) == 800  # 4 valid * 200 iterations
E   AssertionError: assert 1000 == 800
E    +  where 1000 = len([<SensorState.ON: 'on'>, <SensorState.OFF: 'off'>, <SensorState.OPEN: 'open'>, <SensorState.CLOSED: 'closed'>, <SensorState.UNKNOWN: 'unknown'>, <SensorState.ON: 'on'>, ...])
_ TestConstantValidationPerformance.test_dictionary_constant_access_performance _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_constants_integration_advanced.py:817: in test_dictionary_constant_access_performance
    assert access_time < 0.01
E   assert 0.011544227600097656 < 0.01
________ TestOccupancyPredictionSystem.test_run_without_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_main_system.py:265: in test_run_without_initialization
    await run_with_timeout()
tests/unit/test_main_system.py:261: in run_with_timeout
    await asyncio.wait_for(system.run(), timeout=0.1)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/tasks.py:520: in wait_for
    return await fut
src/main_system.py:114: in run
    await self.shutdown()
src/main_system.py:122: in shutdown
    await self.tracking_manager.stop_tracking()
E   TypeError: object MagicMock can't be used in 'await' expression
__________ TestSecretsManager.test_get_or_create_key_creates_new_key ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_environment.py:174: in test_get_or_create_key_creates_new_key
    assert len(decoded_key) == 32  # Fernet keys are 32 bytes
E   AssertionError: assert 44 == 32
E    +  where 44 = len(b'Vu4dAYkafI3EPLX9Rnc7-ae-V4RcxMp3RS7RTG3ieHw=')
_ TestOccupancyPredictionSystem.test_shutdown_handles_tracking_manager_exception _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_main_system.py:444: in test_shutdown_handles_tracking_manager_exception
    assert system.running is False
E   assert True is False
E    +  where True = <src.main_system.OccupancyPredictionSystem object at 0x7f3c6b0f5190>.running
__________________ TestEnvironmentManager.test_inject_secrets __________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_environment.py:488: in test_inject_secrets
    assert "user:test_db_pass@" in config["database"]["connection_string"]
E   AssertionError: assert 'user:test_db_pass@' in 'postgresql://user@localhost/db'
_ TestOccupancyPredictionSystem.test_system_passes_correct_config_to_components _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
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
_ TestExceptionContextPreservation.test_exception_context_filtering_sensitive_data _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:196: in test_exception_context_filtering_sensitive_data
    assert filtered_context["user_data"] == "[FILTERED]"
E   AssertionError: assert {'name': 'John', 'ssn': '123-...'} == '[FILTERED]'
______ TestMainSystemErrorScenarios.test_api_server_status_check_failure _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
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
____ TestExceptionHierarchyValidation.test_all_exceptions_inherit_from_base ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:228: in test_all_exceptions_inherit_from_base
    instance = exc_class("Test error")
E   TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
_ TestSystemLayerErrorPropagation.test_external_service_to_internal_error_propagation _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:325: in test_external_service_to_internal_error_propagation
    feature_error = MissingFeatureError(
E   TypeError: MissingFeatureError.__init__() got an unexpected keyword argument 'cause'
_____ TestValidationFunctionEdgeCases.test_validate_room_id_comprehensive ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:611: in test_validate_room_id_comprehensive
    validate_room_id(invalid_room_id)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
____ TestValidationFunctionEdgeCases.test_validate_entity_id_comprehensive _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:660: in test_validate_entity_id_comprehensive
    validate_entity_id(invalid_entity_id)
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
______ TestExceptionLoggingIntegration.test_error_alerting_classification ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:835: in test_error_alerting_classification
    assert (
E   AssertionError: Classification mismatch for ModelPredictionError: alert_level
E   assert 'info' == 'warning'
E     - warning
E     + info
_______ TestProductionErrorScenarios.test_memory_pressure_error_handling _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:863: in test_memory_pressure_error_handling
    assert system_error.resource_type == "memory"
E   AttributeError: 'ResourceExhaustionError' object has no attribute 'resource_type'
_________ TestProductionErrorScenarios.test_cascading_failure_scenario _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:904: in test_cascading_failure_scenario
    assert len(degraded_error.context["model_predictions"]) == 3
E   KeyError: 'model_predictions'
___ TestProductionErrorScenarios.test_data_corruption_detection_and_handling ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:982: in test_data_corruption_detection_and_handling
    assert "corruption" in str(error).lower() or "invalid" in str(error).lower()
E   assert ('corruption' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" or 'invalid' in "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}")
E    +  where "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fc2a535a930>()
E    +    where <built-in method lower of str object at 0x7fc2a535a930> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
E    +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
E    +  and   "database integrity error in table 'sensor_events': primary_key_violation | error code: db_integrity_error | context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = <built-in method lower of str object at 0x7fc2a535ab30>()
E    +    where <built-in method lower of str object at 0x7fc2a535ab30> = "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}".lower
E    +      where "Database integrity error in table 'sensor_events': primary_key_violation | Error Code: DB_INTEGRITY_ERROR | Context: constraint=primary_key_violation, table=sensor_events, values={'affected_records': 1500}" = str(DatabaseIntegrityError("Database integrity error in table 'sensor_events': primary_key_violation"))
____ TestProductionErrorScenarios.test_graceful_degradation_error_patterns _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exception_propagation_advanced.py:1020: in test_graceful_degradation_error_patterns
    assert any(
E   assert False
E    +  where False = any(<generator object TestProductionErrorScenarios.test_graceful_degradation_error_patterns.<locals>.<genexpr> at 0x7fc2a51e4900>)
_ TestHomeAssistantErrors.test_home_assistant_authentication_error_with_string_token _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:286: in test_home_assistant_authentication_error_with_string_token
    assert "very_long_token..." in error.context["token_hint"]
E   AssertionError: assert 'very_long_token...' in 'very_long_...'
_ TestHomeAssistantErrors.test_home_assistant_authentication_error_with_short_token _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:304: in test_home_assistant_authentication_error_with_short_token
    assert error.context["token_hint"] == "short_token"
E   AssertionError: assert 'short_toke...' == 'short_token'
E     - short_token
E     ?           ^
E     + short_toke...
E     ?           ^^^
___________________ TestSystemErrors.test_system_error_basic ___________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1040: in test_system_error_basic
    assert str(error) == "Something went wrong"
E   AssertionError: assert 'Something we... SYSTEM_ERROR' == 'Something went wrong'
E     - Something went wrong
E     + Something went wrong | Error Code: SYSTEM_ERROR
______________ TestSystemErrors.test_system_error_with_operation _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1049: in test_system_error_with_operation
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...ta_processing' == 'System error...ed to process'
E     - System error during data_processing: Failed to process
E     + System error during data_processing: Failed to process | Error Code: SYSTEM_ERROR | Context: operation=data_processing
_______ TestSystemErrors.test_system_error_with_component_and_operation ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1064: in test_system_error_with_component_and_operation
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...Invalid input' == 'System error...dation failed'
E     - System error in input_validator during user_input_validation: Validation failed
E     + System error in input_validator during user_input_validation: Validation failed | Error Code: SYSTEM_ERROR | Context: component=input_validator, operation=user_input_validation | Caused by: ValueError: Invalid input
_______________ TestSystemErrors.test_resource_exhaustion_error ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1074: in test_resource_exhaustion_error
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...=90.0, unit=%' == 'Resource exh...limit: 90.0%)'
E     - Resource exhaustion: memory at 95.5% (limit: 90.0%)
E     + System error in memory during resource_monitoring: Resource exhaustion: memory at 95.5% (limit: 90.0%) | Error Code: RESOURCE_EXHAUSTION_ERROR | Context: component=memory, operation=resource_monitoring, resource_type=memory, current_usage=95.5, limit=90.0, unit=%
____________ TestSystemErrors.test_service_unavailable_error_basic _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1090: in test_service_unavailable_error_basic
    assert error.context["operation"] == "service_access"
E   KeyError: 'operation'
______________ TestSystemErrors.test_maintenance_mode_error_basic ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1114: in test_maintenance_mode_error_basic
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...CE_MODE_ERROR' == 'System in maintenance mode'
E     - System in maintenance mode
E     + System error in system during maintenance: System in maintenance mode | Error Code: MAINTENANCE_MODE_ERROR
_________ TestSystemErrors.test_maintenance_mode_error_with_component __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1125: in test_maintenance_mode_error_with_component
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...nent=database' == 'System compo...ode: database'
E     - System component in maintenance mode: database
E     + System error in database during maintenance: System component in maintenance mode: database | Error Code: MAINTENANCE_MODE_ERROR | Context: component=database
__________ TestSystemErrors.test_maintenance_mode_error_with_end_time __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1133: in test_maintenance_mode_error_with_end_time
    assert str(error) == expected_msg
E   AssertionError: assert 'System error...-15 14:00 UTC' == 'System compo...15 14:00 UTC)'
E     - System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC)
E     + System error in search_engine during maintenance: System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC) | Error Code: MAINTENANCE_MODE_ERROR | Context: component=search_engine, estimated_end_time=2024-01-15 14:00 UTC
______________ TestAPIErrors.test_rate_limit_exceeded_error_basic ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1184: in test_rate_limit_exceeded_error_basic
    assert str(error) == expected_msg
E   AssertionError: assert 'Rate limit e..._seconds=3600' == 'Rate limit e...sts per 3600s'
E     - Rate limit exceeded for api: 100 requests per 3600s
E     + Rate limit exceeded for api: 100 requests per 3600s | Error Code: RATE_LIMIT_EXCEEDED_ERROR | Context: service=api, limit=100, window_seconds=3600
_________ TestValidationFunctions.test_validate_room_id_invalid_empty __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1307: in test_validate_room_id_invalid_empty
    validate_room_id("")
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
__________ TestValidationFunctions.test_validate_room_id_invalid_none __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1316: in test_validate_room_id_invalid_none
    validate_room_id(None)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
__________ TestValidationFunctions.test_validate_room_id_invalid_type __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1324: in test_validate_room_id_invalid_type
    validate_room_id(123)
src/core/exceptions.py:1271: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_______ TestValidationFunctions.test_validate_room_id_invalid_characters _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1345: in test_validate_room_id_invalid_characters
    validate_room_id(room_id)
src/core/exceptions.py:1278: in validate_room_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
________ TestValidationFunctions.test_validate_entity_id_invalid_empty _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1370: in test_validate_entity_id_invalid_empty
    validate_entity_id("")
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_________ TestValidationFunctions.test_validate_entity_id_invalid_none _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1378: in test_validate_entity_id_invalid_none
    validate_entity_id(None)
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_________ TestValidationFunctions.test_validate_entity_id_invalid_type _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1385: in test_validate_entity_id_invalid_type
    validate_entity_id(["sensor", "temperature"])
src/core/exceptions.py:1296: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
________ TestValidationFunctions.test_validate_entity_id_invalid_format ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1419: in test_validate_entity_id_invalid_format
    validate_entity_id(entity_id)
src/core/exceptions.py:1303: in validate_entity_id
    raise DataValidationError(
E   TypeError: DataValidationError.__init__() missing 2 required positional arguments: 'data_source' and 'validation_errors'
_ TestExceptionInheritanceAndCompatibility.test_all_exceptions_are_exceptions __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1485: in test_all_exceptions_are_exceptions
    instance = exc_class("test message")
E   TypeError: ConfigFileNotFoundError.__init__() missing 1 required positional argument: 'config_dir'
___ TestExceptionInheritanceAndCompatibility.test_error_context_preservation ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_exceptions.py:1493: in test_error_context_preservation
    config_error = ConfigValidationError("Test error", context=context)
E   TypeError: ConfigValidationError.__init__() got an unexpected keyword argument 'context'
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_minimum_length_validation _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:45: in test_jwt_secret_key_minimum_length_validation
    with pytest.raises(
E   Failed: DID NOT RAISE <class 'ValueError'>
----------------------------- Captured stdout call -----------------------------
Warning: Using default test JWT secret key in production environment
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_acceptable_lengths _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:67: in test_jwt_secret_key_acceptable_lengths
    config = JWTConfig()
<string>:13: in __init__
    ???
src/core/config.py:210: in __post_init__
    raise ValueError("JWT secret key must be at least 32 characters long")
E   ValueError: JWT secret key must be at least 32 characters long
_ TestJWTConfigurationSecurityValidation.test_jwt_secret_key_missing_in_production _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_jwt_configuration.py:108: in test_jwt_secret_key_missing_in_production
    with pytest.raises(
E   Failed: DID NOT RAISE <class 'ValueError'>
----------------------------- Captured stdout call -----------------------------
Warning: Using default test JWT secret key in production environment
____ TestJWTEnvironmentHandling.test_jwt_test_environment_fallback_behavior ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
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
_________ TestSensorEventAdvancedFeatures.test_get_advanced_analytics __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.OperationalError: near "(": syntax error
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:90: in test_get_advanced_analytics
    analytics = await SensorEvent.get_advanced_analytics(
src/data/storage/models.py:351: in get_advanced_analytics
    stats_result = await session.execute(stats_query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:306: in orm_execute_statement
    result = conn.execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near "(": syntax error
E   [SQL: SELECT percentile_cont(?) WITHIN GROUP (ORDER BY sensor_events.confidence_score DESC) AS median_confidence, stddev_samp(sensor_events.confidence_score) AS confidence_stddev, CAST(STRFTIME('%s', max(sensor_events.timestamp) - min(sensor_events.timestamp)) AS INTEGER) AS time_span_seconds
E   FROM sensor_events
E   WHERE sensor_events.room_id = ? AND sensor_events.timestamp >= ? AND sensor_events.confidence_score IS NOT NULL]
E   [parameters: (0.5, 'analytics_room', '2025-08-22 16:51:06.164272')]
E   (Background on this error at: https://sqlalche.me/e/20/e3q8)
__________ TestRealTimeMetrics.test_overall_health_score_calculation ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:70: in test_overall_health_score_calculation
    assert metrics.overall_health_score > 80.0
E   AssertionError: assert 69.2 > 80.0
E    +  where 69.2 = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=92.0, window_24h_accuracy=88.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=15, window_24h_predictions=120, accuracy_trend=<TrendDirection.IMPROVING: 'improving'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 22, 22, 51, 6, 918056), measurement_start=None).overall_health_score
_________________ TestRealTimeMetrics.test_is_healthy_criteria _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:108: in test_is_healthy_criteria
    assert metrics_healthy.is_healthy
E   AssertionError: assert False
E    +  where False = RealTimeMetrics(room_id='test_room', model_type=None, window_1h_accuracy=0.0, window_6h_accuracy=70.0, window_24h_accuracy=70.0, window_1h_mean_error=0.0, window_6h_mean_error=0.0, window_24h_mean_error=0.0, window_1h_predictions=0, window_6h_predictions=5, window_24h_predictions=40, accuracy_trend=<TrendDirection.UNKNOWN: 'unknown'>, trend_slope=0.0, trend_confidence=0.0, recent_predictions_rate=0.0, validation_lag_minutes=0.0, confidence_calibration=0.0, active_alerts=[], last_alert_time=None, dominant_accuracy_level=None, recent_validation_records=[], last_updated=datetime.datetime(2025, 8, 22, 22, 51, 6, 938511), measurement_start=None).is_healthy
________________ TestRealTimeMetrics.test_to_dict_serialization ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:131: in test_to_dict_serialization
    assert result["model_type"] == "LSTM"
E   AssertionError: assert 'ModelType.LSTM' == 'LSTM'
E     - LSTM
E     + ModelType.LSTM
_____________ TestAccuracyAlert.test_accuracy_alert_initialization _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:151: in test_accuracy_alert_initialization
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
____________________ TestAccuracyAlert.test_age_calculation ____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:180: in test_age_calculation
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
________________ TestAccuracyAlert.test_escalation_requirements ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:198: in test_escalation_requirements
    critical_alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
___________________ TestAccuracyAlert.test_acknowledge_alert ___________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:230: in test_acknowledge_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_____________________ TestAccuracyAlert.test_resolve_alert _____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:250: in test_resolve_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
____________________ TestAccuracyAlert.test_escalate_alert _____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:271: in test_escalate_alert
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_____________________ TestAccuracyAlert.test_alert_to_dict _____________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:292: in test_alert_to_dict
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'message'
_______________ TestAccuracyTracker.test_tracker_initialization ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:358: in test_tracker_initialization
    assert accuracy_tracker._prediction_validator == mock_validator
E   AttributeError: 'AccuracyTracker' object has no attribute '_prediction_validator'
____________ TestTrendAnalysis.test_analyze_trend_insufficient_data ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:684: in test_analyze_trend_insufficient_data
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
________________ TestTrendAnalysis.test_analyze_trend_improving ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:696: in test_analyze_trend_improving
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
________________ TestTrendAnalysis.test_analyze_trend_degrading ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:708: in test_analyze_trend_degrading
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
_________________ TestTrendAnalysis.test_analyze_trend_stable __________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:720: in test_analyze_trend_stable
    direction, slope = tracker._analyze_trend(datapoints)
E   ValueError: too many values to unpack (expected 2)
------------------------------ Captured log call -------------------------------
ERROR    src.adaptation.tracker:tracker.py:1039 Failed to analyze trend: argument of type 'float' is not iterable
________________ TestTrendAnalysis.test_calculate_global_trend _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:737: in test_calculate_global_trend
    assert "overall_direction" in global_trend
E   AssertionError: assert 'overall_direction' in {'average_slope': 0.0, 'confidence': 0.0, 'direction': <TrendDirection.UNKNOWN: 'unknown'>}
_____________ TestTrendAnalysis.test_calculate_global_trend_empty ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:748: in test_calculate_global_trend_empty
    assert global_trend["overall_direction"] == TrendDirection.UNKNOWN
E   KeyError: 'overall_direction'
________________ TestErrorHandling.test_accuracy_tracking_error ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:760: in test_accuracy_tracking_error
    assert str(error) == "Test error message"
E   AssertionError: assert 'Test error m...RACKING_ERROR' == 'Test error message'
E     - Test error message
E     + Test error message | Error Code: ACCURACY_TRACKING_ERROR
_________ TestErrorHandling.test_accuracy_tracking_error_with_severity _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_adaptation/test_tracker.py:765: in test_accuracy_tracking_error_with_severity
    error = AccuracyTrackingError("Critical error", severity=ErrorSeverity.CRITICAL)
src/adaptation/tracker.py:1564: in __init__
    super().__init__(
E   TypeError: src.core.exceptions.OccupancyPredictionError.__init__() got multiple values for keyword argument 'severity'
______ TestSensorEventAdvancedFeatures.test_get_sensor_efficiency_metrics ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.OperationalError: misuse of window function lag()
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:190: in test_get_sensor_efficiency_metrics
    metrics = await SensorEvent.get_sensor_efficiency_metrics(
src/data/storage/models.py:414: in get_sensor_efficiency_metrics
    result = await session.execute(efficiency_query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:306: in orm_execute_statement
    result = conn.execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) misuse of window function lag()
E   [SQL: SELECT sensor_events.sensor_id, sensor_events.sensor_type, count(*) AS total_events, count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state) AS state_changes, avg(sensor_events.confidence_score) AS avg_confidence, min(sensor_events.confidence_score) AS min_confidence, max(sensor_events.confidence_score) AS max_confidence, (count(*) FILTER (WHERE sensor_events.state != sensor_events.previous_state)) / (CAST(count(*) AS FLOAT) + 0.0) AS state_change_ratio, CAST(STRFTIME('%s', avg(sensor_events.timestamp - lag(sensor_events.timestamp) OVER (PARTITION BY sensor_events.sensor_id ORDER BY sensor_events.timestamp))) AS INTEGER) AS avg_interval_seconds
E   FROM sensor_events
E   WHERE sensor_events.room_id = ? AND sensor_events.timestamp >= ? GROUP BY sensor_events.sensor_id, sensor_events.sensor_type ORDER BY count(*) DESC]
E   [parameters: ('efficiency_room', '2025-08-21 22:51:06.993911')]
E   (Background on this error at: https://sqlalche.me/e/20/e3q8)
______ TestSensorEventAdvancedFeatures.test_efficiency_score_calculation _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_models_advanced.py:257: in test_efficiency_score_calculation
    assert (
E   AssertionError: Score 0.6200000000000001 not in range 0.2-0.6
E   assert 0.6200000000000001 <= 0.6
________ TestModelBackupManager.test_create_backup_uncompressed_success ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:463: in test_create_backup_uncompressed_success
    assert "test_models.tar" in call_args[-3]
E   AssertionError: assert 'test_models.tar' in '-C'
_______ TestModelBackupManager.test_create_backup_nonexistent_models_dir _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:514: in test_create_backup_nonexistent_models_dir
    assert Path(nonexistent_dir).exists()
E   AssertionError: assert False
E    +  where False = <bound method Path.exists of PosixPath('/path/that/does/not/exist')>()
E    +    where <bound method Path.exists of PosixPath('/path/that/does/not/exist')> = PosixPath('/path/that/does/not/exist').exists
E    +      where PosixPath('/path/that/does/not/exist') = Path('/path/that/does/not/exist')
________________ TestBackupManager.test_list_backups_with_data _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:739: in test_list_backups_with_data
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/pathlib.py:1311: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpi0m6i9j1/database'
_____________ TestBackupManager.test_list_backups_filtered_by_type _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:771: in test_list_backups_filtered_by_type
    db_dir.mkdir()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/pathlib.py:1311: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpbw_22rll/database'
_________________ TestBackupManager.test_get_backup_info_found _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:809: in test_get_backup_info_found
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/pathlib.py:1311: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmpl5z0uxzz/database'
________________ TestBackupManager.test_cleanup_expired_backups ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_backup_manager.py:867: in test_cleanup_expired_backups
    backup_dir.mkdir()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/pathlib.py:1311: in mkdir
    os.mkdir(self, mode)
E   FileExistsError: [Errno 17] File exists: '/tmp/tmp_w9f2izr/database'
__________ TestBackupManager.test_run_scheduled_backups_single_cycle ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'create_backup' to have been called once. Called 0 times.
During handling of the above exception, another exception occurred:
tests/unit/test_core/test_backup_manager.py:986: in test_run_scheduled_backups_single_cycle
    mock_config.assert_called_once()  # Config backup triggered due to hour 2
E   AssertionError: Expected 'create_backup' to have been called once. Called 0 times.
______ TestRoomStateAdvancedFeatures.test_get_precision_occupancy_metrics ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.OperationalError: near "(": syntax error
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:516: in test_get_precision_occupancy_metrics
    metrics = await RoomState.get_precision_occupancy_metrics(
src/data/storage/models.py:785: in get_precision_occupancy_metrics
    result = await session.execute(precision_query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:306: in orm_execute_statement
    result = conn.execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near "(": syntax error
E   [SQL: SELECT count(*) AS total_states, count(*) FILTER (WHERE room_states.occupancy_confidence >= ?) AS high_confidence_states, avg(room_states.occupancy_confidence) AS avg_confidence, stddev_samp(room_states.occupancy_confidence) AS confidence_stddev, min(room_states.occupancy_confidence) AS min_confidence, max(room_states.occupancy_confidence) AS max_confidence, percentile_cont(?) WITHIN GROUP (ORDER BY room_states.occupancy_confidence ASC) AS q1_confidence, percentile_cont(?) WITHIN GROUP (ORDER BY room_states.occupancy_confidence ASC) AS median_confidence, percentile_cont(?) WITHIN GROUP (ORDER BY room_states.occupancy_confidence ASC) AS q3_confidence
E   FROM room_states
E   WHERE room_states.room_id = ?]
E   [parameters: (0.8, 0.25, 0.5, 0.75, 'precision_room')]
E   (Background on this error at: https://sqlalche.me/e/20/e3q8)
_ TestConfigurationCorruptionAndRecovery.test_yaml_with_circular_references_deep _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_config.py:1196: in test_yaml_with_circular_references_deep
    config = loader.load_config()
src/core/config.py:425: in load_config
    ha_config = HomeAssistantConfig(**main_config["home_assistant"])
E   TypeError: HomeAssistantConfig.__init__() got an unexpected keyword argument 'reference'
__________ TestTimescaleDBFunctions.test_create_timescale_hypertables __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_models_advanced.py:1407: in test_create_timescale_hypertables
    assert mock_execute.call_count >= 6
E   AssertionError: assert 5 >= 6
E    +  where 5 = <AsyncMock name='execute' id='140473946829360'>.call_count
____ TestDatabaseCompatibilityHelpers.test_get_json_column_type_postgresql _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_models_advanced.py:1512: in test_get_json_column_type_postgresql
    assert json_type == JSONB
E   AssertionError: assert <class 'sqlalchemy.sql.sqltypes.JSON'> == <class 'sqlalchemy.dialects.postgresql.json.JSONB'>
_ TestConfigurationCorruptionAndRecovery.test_configuration_with_malformed_data_types _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_config.py:1277: in test_configuration_with_malformed_data_types
    with pytest.raises(TypeError):
E   Failed: DID NOT RAISE <class 'TypeError'>
_ TestConfigurationRecoveryMechanisms.test_configuration_fallback_to_defaults __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_config.py:1350: in test_configuration_fallback_to_defaults
    config = loader.load_config()
src/core/config.py:428: in load_config
    prediction_config = PredictionConfig(**main_config["prediction"])
E   KeyError: 'prediction'
__ TestConfigurationRecoveryMechanisms.test_configuration_validation_recovery __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_core/test_config.py:1420: in test_configuration_validation_recovery
    with pytest.raises(TypeError):
E   Failed: DID NOT RAISE <class 'TypeError'>
____________ TestModelsIntegration.test_model_relationships_cascade ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.IntegrityError: NOT NULL constraint failed: prediction_audits.prediction_id
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:1563: in test_model_relationships_cascade
    await test_db_session.commit()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:1014: in commit
    await greenlet_spawn(self.sync_session.commit)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2032: in commit
    trans.commit(_to_root=True)
<string>:2: in commit
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state_changes.py:137: in _go
    ret_value = fn(self, *arg, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:1313: in commit
    self._prepare_impl()
<string>:2: in _prepare_impl
    ???
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/state_changes.py:137: in _go
    ret_value = fn(self, *arg, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:1288: in _prepare_impl
    self.session.flush()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:4345: in flush
    self._flush(objects)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:4480: in _flush
    with util.safe_reraise():
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/langhelpers.py:224: in __exit__
    raise exc_value.with_traceback(exc_tb)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:4441: in _flush
    flush_context.execute()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/unitofwork.py:466: in execute
    rec.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/unitofwork.py:642: in execute
    util.preloaded.orm_persistence.save_obj(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/persistence.py:85: in save_obj
    _emit_update_statements(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/persistence.py:912: in _emit_update_statements
    c = connection.execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) NOT NULL constraint failed: prediction_audits.prediction_id
E   [SQL: UPDATE prediction_audits SET prediction_id=? WHERE prediction_audits.id = ?]
E   [parameters: (None, 1)]
E   (Background on this error at: https://sqlalche.me/e/20/gkpj)
___________ TestModelsIntegration.test_comprehensive_model_workflow ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlite3.OperationalError: near "(": syntax error
The above exception was the direct cause of the following exception:
tests/unit/test_data/test_models_advanced.py:1696: in test_comprehensive_model_workflow
    analytics = await SensorEvent.get_advanced_analytics(
src/data/storage/models.py:351: in get_advanced_analytics
    stats_result = await session.execute(stats_query)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/session.py:463: in execute
    result = await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:201: in greenlet_spawn
    result = context.throw(*sys.exc_info())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2365: in execute
    return self._execute_internal(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/session.py:2251: in _execute_internal
    result: Result[Any] = compile_state_cls.orm_execute_statement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/orm/context.py:306: in orm_execute_statement
    result = conn.execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:526: in _execute_on_connection
    return connection._execute_clauseelement(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1641: in _execute_clauseelement
    ret = self._execute_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1846: in _execute_context
    return self._exec_single_context(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1986: in _exec_single_context
    self._handle_dbapi_exception(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2355: in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1967: in _exec_single_context
    self.dialect.do_execute(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/default.py:951: in do_execute
    cursor.execute(statement, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:177: in execute
    self._adapt_connection._handle_exception(error)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:337: in _handle_exception
    raise error
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/aiosqlite.py:159: in execute
    self.await_(_cursor.execute(operation, parameters))
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:132: in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:196: in greenlet_spawn
    value = await result
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:40: in execute
    await self._execute(self._cursor.execute, sql, parameters)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/cursor.py:32: in _execute
    return await self._conn._execute(fn, *args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:122: in _execute
    return await future
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/aiosqlite/core.py:105: in run
    result = function()
E   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near "(": syntax error
E   [SQL: SELECT percentile_cont(?) WITHIN GROUP (ORDER BY sensor_events.confidence_score DESC) AS median_confidence, stddev_samp(sensor_events.confidence_score) AS confidence_stddev, CAST(STRFTIME('%s', max(sensor_events.timestamp) - min(sensor_events.timestamp)) AS INTEGER) AS time_span_seconds
E   FROM sensor_events
E   WHERE sensor_events.room_id = ? AND sensor_events.timestamp >= ? AND sensor_events.confidence_score IS NOT NULL]
E   [parameters: (0.5, 'workflow_room', '2025-08-22 20:51:10.666794')]
E   (Background on this error at: https://sqlalche.me/e/20/e3q8)
______ TestStatisticalPatternAnalyzer.test_statistical_anomaly_detection _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:627: in test_statistical_anomaly_detection
    assert "anomaly_count" in analysis
E   AssertionError: assert 'anomaly_count' in {'event_count': 11, 'mean_interval': 300.1, 'median_interval': 300.0, 'state_distribution': {'on': 1.0}, ...}
_______ TestStatisticalPatternAnalyzer.test_sensor_malfunction_detection _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:671: in test_sensor_malfunction_detection
    assert len(high_freq_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
_________ TestJSONSchemaValidator.test_sensor_event_schema_validation __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/test_data/test_validation.py:866: in test_sensor_event_schema_validation
    assert result.is_valid
E   assert False
E    +  where False = ValidationResult(is_valid=False, errors=[ValidationError(rule_id='SCH_JSON_ERROR', field='validation_process', value="'NoneType' object has no attribute 'checks'", message="Schema validation error: 'NoneType' object has no attribute 'checks'", severity=<ErrorSeverity.HIGH: 'high'>, suggestion='Check schema definition and data format', context={})], warnings=[], confidence_score=0.8, processing_time_ms=0.0, security_flags=[], integrity_hash=None, validation_id='b19df454-375f-457e-ac89-90778b9344a7').is_valid
Error: Process completed with exit code 2.
