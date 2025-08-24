[gw0] [ 51%] PASSED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_clear_caches_calls_all_extractors 
Exception ignored in: <function FeatureEngineeringEngine.__del__ at 0x7fed02011080>
Traceback (most recent call last):
  File "/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/features/engineering.py", line 841, in __del__
    if self.executor:
       ^^^^^^^^^^^^^
AttributeError: 'FeatureEngineeringEngine' object has no attribute 'executor'

==================================== ERRORS ====================================
______ ERROR collecting tests/unit/integration_layer/test_api_services.py ______
ImportError while importing test module '/home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_api_services.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/integration_layer/test_api_services.py:53: in <module>
    from src.integration.dashboard import DashboardIntegration
E   ImportError: cannot import name 'DashboardIntegration' from 'src.integration.dashboard' (/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/dashboard.py)
__ ERROR at setup of TestPredictionValidation.test_validation_record_creation __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:111: in validation_record
    return ValidationRecord(
E   TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestPredictionValidation.test_validation_record_validate_against_actual _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:111: in validation_record
    return ValidationRecord(
E   TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestPredictionValidation.test_validation_record_inaccurate_prediction _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:111: in validation_record
    return ValidationRecord(
E   TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
_ ERROR at setup of TestPredictionValidation.test_validation_record_mark_expired _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:111: in validation_record
    return ValidationRecord(
E   TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
__ ERROR at setup of TestPredictionValidation.test_validation_record_to_dict ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:111: in validation_record
    return ValidationRecord(
E   TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
____________ ERROR at setup of TestHAClient.test_ha_client_connect _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
__ ERROR at setup of TestHAClient.test_ha_client_test_authentication_success ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
__ ERROR at setup of TestHAClient.test_ha_client_test_authentication_failure ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
_______ ERROR at setup of TestHAClient.test_ha_client_connect_websocket ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
__ ERROR at setup of TestHAClient.test_ha_client_validate_and_normalize_state __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
______ ERROR at setup of TestHAClient.test_ha_client_should_process_event ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
________ ERROR at setup of TestHAClient.test_ha_client_get_entity_state ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
___ ERROR at setup of TestHAClient.test_ha_client_get_entity_state_not_found ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
_______ ERROR at setup of TestHAClient.test_ha_client_get_entity_history _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
_ ERROR at setup of TestHAClient.test_ha_client_convert_ha_event_to_sensor_event _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
___________ ERROR at setup of TestHAClient.test_ha_client_disconnect ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
_ ERROR at setup of TestDatabaseManager.test_database_manager_initialization_with_config _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_database_manager_initialization_default_config _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
__ ERROR at setup of TestDatabaseManager.test_connection_stats_initialization __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_create_engine_postgresql_url_conversion _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_create_engine_with_nullpool_for_testing _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_create_engine_invalid_connection_string _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
______ ERROR at setup of TestDatabaseManager.test_setup_connection_events ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_setup_connection_events_without_engine _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_______ ERROR at setup of TestDatabaseManager.test_setup_session_factory _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
__ ERROR at setup of TestDatabaseManager.test_database_manager_initialization __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
_ ERROR at setup of TestDatabaseManager.test_setup_session_factory_without_engine _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
____ ERROR at setup of TestDatabaseManager.test_database_manager_initialize ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
_____ ERROR at setup of TestDatabaseManager.test_verify_connection_success _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
___ ERROR at setup of TestDatabaseManager.test_database_manager_health_check ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
_ ERROR at setup of TestDatabaseManager.test_database_manager_health_check_failure _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
_ ERROR at setup of TestDatabaseManager.test_verify_connection_timescaledb_missing _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
___ ERROR at setup of TestDatabaseManager.test_database_manager_get_session ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
_ ERROR at setup of TestDatabaseManager.test_verify_connection_without_engine __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
__ ERROR at setup of TestDatabaseManager.test_database_manager_execute_query ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
________ ERROR at setup of TestDatabaseManager.test_get_session_success ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
______ ERROR at setup of TestDatabaseManager.test_database_manager_close _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1752: in mock_config
    config.database.connection_string = (
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'database'
____ ERROR at setup of TestDatabaseManager.test_get_session_without_factory ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_________ ERROR at setup of TestHAClient.test_ha_client_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1922: in mock_config
    config.home_assistant.url = "http://localhost:8123"
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'home_assistant'
_ ERROR at setup of TestDatabaseManager.test_get_session_with_retry_on_connection_error _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
___ ERROR at setup of TestDatabaseManager.test_get_session_retry_exhaustion ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
________ ERROR at setup of TestDatabaseManager.test_execute_query_basic ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
____ ERROR at setup of TestDatabaseManager.test_execute_query_with_timeout _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
____ ERROR at setup of TestDatabaseManager.test_execute_query_timeout_error ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
______ ERROR at setup of TestDatabaseManager.test_execute_query_sql_error ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_execute_optimized_query_with_prepared_statements _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_execute_optimized_query_prepared_statement_fallback _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_____ ERROR at setup of TestDatabaseManager.test_analyze_query_performance _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
___ ERROR at setup of TestDatabaseManager.test_get_optimization_suggestions ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
____ ERROR at setup of TestDatabaseManager.test_get_connection_pool_metrics ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_get_connection_pool_metrics_high_utilization _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestDatabaseManager.test_get_connection_pool_metrics_without_engine _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestMQTTIntegrationManager.test_integration_manager_publish_prediction_success _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:643: in mock_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
_ ERROR at setup of TestMQTTIntegrationManager.test_integration_manager_publish_prediction_failure _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:643: in mock_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
_______ ERROR at setup of TestDatabaseManager.test_health_check_success ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_ ERROR at setup of TestMQTTIntegrationManager.test_integration_manager_publish_prediction_inactive _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:643: in mock_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
_______ ERROR at setup of TestDatabaseManager.test_health_check_failure ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_________ ERROR at setup of TestDatabaseManager.test_health_check_loop _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
______ ERROR at setup of TestDatabaseManager.test_is_initialized_property ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
_______ ERROR at setup of TestDatabaseManager.test_get_connection_stats ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
______________ ERROR at setup of TestDatabaseManager.test_cleanup ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
____ ERROR at setup of TestDatabaseManager.test_cleanup_with_completed_task ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:101: in setup_patches
    ) as mock_event, patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
__ ERROR at setup of TestModelTrainingPipeline.test_initial_training_success ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
__ ERROR at setup of TestModelTrainingPipeline.test_train_room_models_success __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
________ ERROR at setup of TestModelTraining.test_train_models_ensemble ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
___________ ERROR at setup of TestModelTraining.test_validate_models ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
___ ERROR at setup of TestModelTraining.test_evaluate_and_select_best_model ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
________ ERROR at setup of TestModelTraining.test_deploy_trained_models ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
________ ERROR at setup of TestModelArtifacts.test_save_model_artifacts ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
_______ ERROR at setup of TestPipelineManagement.test_get_model_registry _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
_____ ERROR at setup of TestPipelineManagement.test_get_model_performance ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:140: in mock_base_predictor
    return_value=TrainingResult(
E   TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
_ ERROR at setup of TestMQTTIntegrationErrorHandling.test_publish_prediction_exception_handling _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:643: in mock_prediction_result
    return PredictionResult(
E   TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
=================================== FAILURES ===================================
_ TestEnhancedMonitoring.test_get_enhanced_tracking_manager_convenience_function _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2320: in test_get_enhanced_tracking_manager_convenience_function
    manager = get_enhanced_tracking_manager()
E   TypeError: get_enhanced_tracking_manager() missing 1 required positional argument: 'config'
___________ TestPredictionValidation.test_accuracy_metrics_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:165: in test_accuracy_metrics_creation
    metrics = AccuracyMetrics(
E   TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'room_id'
____________ TestModelBackupManager.test_create_backup_tar_failure _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/core/backup_manager.py:353: in create_backup
    raise RuntimeError(f"tar command failed: {result.stderr}")
E   RuntimeError: tar command failed: tar: cannot access

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:694: in test_create_backup_tar_failure
    model_backup_manager.create_backup()
src/core/backup_manager.py:376: in create_backup
    backup_file.unlink()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/pathlib.py:1342: in unlink
    os.unlink(self)
E   FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-0/popen-gw1/test_create_backup_tar_failure0/backups/models/models_20250824_081822.tar.gz'
------------------------------ Captured log call -------------------------------
ERROR    src.core.backup_manager:backup_manager.py:374 Failed to create models backup: tar command failed: tar: cannot access
____ TestPredictionValidation.test_accuracy_metrics_confidence_calibration _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:182: in test_accuracy_metrics_confidence_calibration
    metrics = AccuracyMetrics(
E   TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'room_id'
_____ TestConfigurationBackupManager.test_create_backup_auto_generated_id ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_backup_management.py:869: in test_create_backup_auto_generated_id
    assert len(result.backup_id) == 21  # "config_" + 8 + "_" + 6
E   AssertionError: assert 22 == 21
E    +  where 22 = len('config_20250824_081822')
E    +    where 'config_20250824_081822' = BackupMetadata(backup_id='config_20250824_081822', backup_type='config', timestamp=datetime.datetime(2025, 8, 24, 8, 18, 22, 925449), size_bytes=256000, compressed=True, checksum=None, retention_date=None, tags={'config_dir': '/tmp/pytest-of-runner/pytest-0/popen-gw1/test_create_backup_auto_genera2/config'}).backup_id
______ TestPredictionValidation.test_prediction_validator_initialization _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:198: in test_prediction_validator_initialization
    validator = PredictionValidator(
E   TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'validation_window_hours'
_____ TestPredictionValidation.test_prediction_validator_record_prediction _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:214: in test_prediction_validator_record_prediction
    validator = PredictionValidator(enable_background_tasks=False)
E   TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
____ TestPredictionValidation.test_prediction_validator_validate_prediction ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:235: in test_prediction_validator_validate_prediction
    validator = PredictionValidator(enable_background_tasks=False)
E   TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
___ TestPredictionValidation.test_prediction_validator_get_accuracy_metrics ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:260: in test_prediction_validator_get_accuracy_metrics
    validator = PredictionValidator(enable_background_tasks=False)
E   TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
__ TestPredictionValidation.test_prediction_validator_expire_old_predictions ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:287: in test_prediction_validator_expire_old_predictions
    validator = PredictionValidator(enable_background_tasks=False)
E   TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
_ TestBackupManagerOrchestration.test_run_scheduled_backups_model_backup_interval _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1643: in _get_target
    target, attribute = target.rsplit('.', 1)
E   ValueError: not enough values to unpack (expected 2, got 1)

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1018: in test_run_scheduled_backups_model_backup_interval
    with patch("datetime") as mock_datetime:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1803: in patch
    getter, attribute = _get_target(target)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1645: in _get_target
    raise TypeError(
E   TypeError: Need a valid target to patch. You supplied: 'datetime'
___________ TestPredictionValidation.test_validation_error_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:308: in test_validation_error_creation
    error = ValidationError("Test validation error", "VALIDATION_001")
E   TypeError: ValidationError.__init__() takes 2 positional arguments but 3 were given
________________ TestDriftDetection.test_statistical_test_enum _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:332: in test_statistical_test_enum
    assert StatisticalTest.KS_TEST.value == "ks_test"
E   AttributeError: type object 'StatisticalTest' has no attribute 'KS_TEST'
________________ TestDriftDetection.test_drift_metrics_creation ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:339: in test_drift_metrics_creation
    metrics = DriftMetrics(
E   TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
_________ TestDriftDetection.test_drift_metrics_severity_determination _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:357: in test_drift_metrics_severity_determination
    low_drift = DriftMetrics(room_id="test", drift_score=0.2)
E   TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
_ TestBackupManagerOrchestration.test_run_scheduled_backups_config_backup_daily _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1643: in _get_target
    target, attribute = target.rsplit('.', 1)
E   ValueError: not enough values to unpack (expected 2, got 1)

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1051: in test_run_scheduled_backups_config_backup_daily
    with patch("datetime") as mock_datetime:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1803: in patch
    getter, attribute = _get_target(target)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1645: in _get_target
    raise TypeError(
E   TypeError: Need a valid target to patch. You supplied: 'datetime'
____________ TestDriftDetection.test_drift_metrics_recommendations _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:369: in test_drift_metrics_recommendations
    metrics = DriftMetrics(
E   TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
________________ TestDriftDetection.test_drift_metrics_to_dict _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:382: in test_drift_metrics_to_dict
    metrics = DriftMetrics(
E   TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
_________________ TestDriftDetection.test_feature_drift_result _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:401: in test_feature_drift_result
    test_type=StatisticalTest.KS_TEST,
E   AttributeError: type object 'StatisticalTest' has no attribute 'KS_TEST'
________ TestDriftDetection.test_concept_drift_detector_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:414: in test_concept_drift_detector_initialization
    detector = ConceptDriftDetector(
E   TypeError: ConceptDriftDetector.__init__() got an unexpected keyword argument 'prediction_validator'
_ TestBackupManagerOrchestration.test_cleanup_expired_backups_metadata_discovery _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1581: in __enter__
    setattr(self.target, self.attribute, new_attr)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1120: in test_cleanup_expired_backups_metadata_discovery
    with patch.object(backup_manager.backup_dir, "rglob", return_value=mock_files):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1594: in __enter__
    if not self.__exit__(*sys.exc_info()):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1605: in __exit__
    delattr(self.target, self.attribute)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only
_________ TestDriftDetection.test_concept_drift_detector_detect_drift __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:431: in test_concept_drift_detector_detect_drift
    detector = ConceptDriftDetector(prediction_validator=mock_validator)
E   TypeError: ConceptDriftDetector.__init__() got an unexpected keyword argument 'prediction_validator'
_________ TestDriftDetection.test_concept_drift_detector_calculate_psi _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:464: in test_concept_drift_detector_calculate_psi
    assert 0.05 <= psi_score <= 0.5
E   assert 0.05 <= 0.0
------------------------------ Captured log call -------------------------------
WARNING  src.adaptation.drift_detector:drift_detector.py:828 Error calculating numerical PSI: 'numpy.ndarray' object has no attribute 'quantile'
_______ TestDriftDetection.test_concept_drift_detector_page_hinkley_test _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:474: in test_concept_drift_detector_page_hinkley_test
    drift_detected, change_point = detector._run_page_hinkley_test(errors)
E   TypeError: ConceptDriftDetector._run_page_hinkley_test() missing 1 required positional argument: 'room_id'
__________ TestBackupManagerOrchestration.test_list_backups_all_types __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1581: in __enter__
    setattr(self.target, self.attribute, new_attr)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1154: in test_list_backups_all_types
    with patch.object(backup_manager.backup_dir, "rglob", return_value=mock_files):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1594: in __enter__
    if not self.__exit__(*sys.exc_info()):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1605: in __exit__
    delattr(self.target, self.attribute)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only
________ TestDriftDetection.test_feature_drift_detector_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:484: in test_feature_drift_detector_initialization
    detector = FeatureDriftDetector(
E   TypeError: FeatureDriftDetector.__init__() got an unexpected keyword argument 'monitoring_interval_minutes'
_____ TestDriftDetection.test_feature_drift_detector_numerical_drift_test ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:526: in test_feature_drift_detector_numerical_drift_test
    assert result.feature_name == "test_feature"
E   AttributeError: 'coroutine' object has no attribute 'feature_name'
____ TestDriftDetection.test_feature_drift_detector_categorical_drift_test _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:546: in test_feature_drift_detector_categorical_drift_test
    assert result.feature_name == "category_feature"
E   AttributeError: 'coroutine' object has no attribute 'feature_name'
________________ TestDriftDetection.test_drift_detection_error _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:554: in test_drift_detection_error
    assert str(error) == "Drift detection failed"
src/core/exceptions.py:58: in __str__
    context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
E   AttributeError: 'ErrorSeverity' object has no attribute 'items'
_______ TestBackupManagerOrchestration.test_list_backups_type_filtering ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1581: in __enter__
    setattr(self.target, self.attribute, new_attr)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1176: in test_list_backups_type_filtering
    with patch.object(backup_manager.backup_dir, "rglob", return_value=[Mock()]):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1594: in __enter__
    if not self.__exit__(*sys.exc_info()):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1605: in __exit__
    delattr(self.target, self.attribute)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only
___________ TestAdaptiveRetraining.test_retraining_request_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:589: in test_retraining_request_creation
    request = RetrainingRequest(
E   TypeError: RetrainingRequest.__init__() got an unexpected keyword argument 'accuracy_threshold'
______ TestAdaptiveRetraining.test_retraining_request_priority_comparison ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:609: in test_retraining_request_priority_comparison
    high_priority = RetrainingRequest(
E   TypeError: RetrainingRequest.__init__() missing 2 required positional arguments: 'strategy' and 'created_time'
____________ TestAdaptiveRetraining.test_retraining_request_to_dict ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:629: in test_retraining_request_to_dict
    request = RetrainingRequest(
E   TypeError: RetrainingRequest.__init__() missing 2 required positional arguments: 'priority' and 'created_time'
___________ TestAdaptiveRetraining.test_retraining_progress_creation ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:645: in test_retraining_progress_creation
    progress = RetrainingProgress(
E   TypeError: RetrainingProgress.__init__() got an unexpected keyword argument 'status'
_____ TestBackupManagerOrchestration.test_get_backup_info_specific_lookup ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1581: in __enter__
    setattr(self.target, self.attribute, new_attr)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1194: in test_get_backup_info_specific_lookup
    with patch.object(backup_manager.backup_dir, "rglob", return_value=[Mock()]):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1594: in __enter__
    if not self.__exit__(*sys.exc_info()):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1605: in __exit__
    delattr(self.target, self.attribute)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only
____________ TestAdaptiveRetraining.test_retraining_progress_update ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:661: in test_retraining_progress_update
    progress = RetrainingProgress(
E   TypeError: RetrainingProgress.__init__() got an unexpected keyword argument 'status'
___________ TestAdaptiveRetraining.test_retraining_history_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:683: in test_retraining_history_creation
    assert len(history.retraining_records) == 0
E   AttributeError: 'RetrainingHistory' object has no attribute 'retraining_records'. Did you mean: 'add_retraining_record'?
__________ TestAdaptiveRetraining.test_retraining_history_add_record ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:692: in test_retraining_history_add_record
    history.add_retraining_record(
E   TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
________ TestBackupManagerOrchestration.test_get_backup_info_not_found _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1581: in __enter__
    setattr(self.target, self.attribute, new_attr)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_backup_management.py:1204: in test_get_backup_info_not_found
    with patch.object(backup_manager.backup_dir, "rglob", return_value=[]):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1594: in __enter__
    if not self.__exit__(*sys.exc_info()):
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1605: in __exit__
    delattr(self.target, self.attribute)
E   AttributeError: 'PosixPath' object attribute 'rglob' is read-only
_________ TestAdaptiveRetraining.test_retraining_history_success_rate __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:717: in test_retraining_history_success_rate
    history.add_retraining_record(
E   TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
______ TestAdaptiveRetraining.test_retraining_history_recent_performance _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:735: in test_retraining_history_recent_performance
    history.add_retraining_record(
E   TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
________ TestAdaptiveRetraining.test_adaptive_retrainer_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:756: in test_adaptive_retrainer_initialization
    retrainer = AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
_____ TestAdaptiveRetraining.test_adaptive_retrainer_initialization_async ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:775: in test_adaptive_retrainer_initialization_async
    retrainer = AdaptiveRetrainer(adaptive_retraining_enabled=False)
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
___ TestAdaptiveRetraining.test_adaptive_retrainer_evaluate_retraining_need ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:800: in test_adaptive_retrainer_evaluate_retraining_need
    retrainer = AdaptiveRetrainer(
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold_minutes'
______ TestAdaptiveRetraining.test_adaptive_retrainer_request_retraining _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:815: in test_adaptive_retrainer_request_retraining
    retrainer = AdaptiveRetrainer(adaptive_retraining_enabled=True)
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
______ TestAdaptiveRetraining.test_adaptive_retrainer_get_retrainer_stats ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:835: in test_adaptive_retrainer_get_retrainer_stats
    retrainer = AdaptiveRetrainer()
E   TypeError: AdaptiveRetrainer.__init__() missing 1 required positional argument: 'tracking_config'
_______ TestAdaptiveRetraining.test_adaptive_retrainer_cancel_retraining _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:854: in test_adaptive_retrainer_cancel_retraining
    retrainer = AdaptiveRetrainer(adaptive_retraining_enabled=True)
E   TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
_________________ TestAdaptiveRetraining.test_retraining_error _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:872: in test_retraining_error
    error = RetrainingError("Retraining failed", "RETRAIN_001")
E   TypeError: RetrainingError.__init__() takes 2 positional arguments but 3 were given
__________ TestConfigLoader.test_config_loader_load_yaml_missing_file __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1457: in test_config_loader_load_yaml_missing_file
    loader = ConfigLoader()
src/core/config.py:410: in __init__
    raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
E   FileNotFoundError: Configuration directory not found: config
___________ TestModelOptimization.test_hyperparameter_space_creation ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:911: in test_hyperparameter_space_creation
    space = HyperparameterSpace(
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
_______ TestModelOptimization.test_hyperparameter_space_parameter_names ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:927: in test_hyperparameter_space_parameter_names
    space = HyperparameterSpace({"param1": (0, 1), "param2": ["a", "b", "c"]})
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
_______ TestConfigLoader.test_config_loader_load_config_with_environment _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:986: in assert_has_calls
    raise AssertionError(
E   AssertionError: Calls not found.
E   Expected: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
E    call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8')]
E     Actual: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
E    call().__enter__(),
E    call().__exit__(None, None, None),
E    call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8'),
E    call().__enter__(),
E    call().__exit__(None, None, None)]

During handling of the above exception, another exception occurred:
tests/unit/core_system/test_configuration_system.py:1511: in test_config_loader_load_config_with_environment
    mock_open_file.assert_has_calls(expected_calls, any_order=False)
E   AssertionError: Calls not found.
E   Expected: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
E    call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8')]
E     Actual: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
E    call().__enter__(),
E    call().__exit__(None, None, None),
E    call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8'),
E    call().__enter__(),
E    call().__exit__(None, None, None)]
__ TestModelOptimization.test_hyperparameter_space_continuous_identification ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:936: in test_hyperparameter_space_continuous_identification
    space = HyperparameterSpace(
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
______ TestModelOptimization.test_hyperparameter_space_bounds_and_choices ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:945: in test_hyperparameter_space_bounds_and_choices
    space = HyperparameterSpace(
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
__________ TestGlobalConfiguration.test_get_config_singleton_behavior __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1675: in test_get_config_singleton_behavior
    assert config1 is mock_system_config
E   AssertionError: assert <Mock name='ConfigLoader()._create_system_config()' id='140554928732688'> is <Mock name='ConfigLoader().load_config()' spec='SystemConfig' id='140554928729088'>
___________ TestModelOptimization.test_hyperparameter_space_sampling ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:959: in test_hyperparameter_space_sampling
    space = HyperparameterSpace({"param1": (0, 10), "param2": ["A", "B", "C"]})
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
___________ TestModelOptimization.test_hyperparameter_space_to_dict ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:972: in test_hyperparameter_space_to_dict
    space = HyperparameterSpace(
E   TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
___________ TestModelOptimization.test_optimization_result_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:984: in test_optimization_result_creation
    result = OptimizationResult(
E   TypeError: OptimizationResult.__init__() got an unexpected keyword argument 'n_evaluations'
___ TestGlobalConfiguration.test_get_config_environment_manager_integration ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1703: in test_get_config_environment_manager_integration
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
___________ TestModelOptimization.test_optimization_config_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1006: in test_optimization_config_creation
    config = OptimizationConfig(
E   TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'timeout_minutes'
__________ TestModelOptimization.test_model_optimizer_initialization ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1044: in test_model_optimizer_initialization
    assert ModelType.LSTM in optimizer._parameter_spaces
E   AssertionError: assert <ModelType.LSTM: 'lstm'> in {'gaussian_process': [{'categories': ['rb', 'matern', 'rational_quadratic'], 'name': 'kernel', 'type': 'categorical'}, {'high': 0.1, 'low': 1e-12, 'name': 'alpha', 'type': 'continuous'}, {'high': 10, 'low': 0, 'name': 'n_restarts_optimizer', 'type': 'integer'}], 'hmm': [{'high': 8, 'low': 2, 'name': 'n_states', 'type': 'integer'}, {'categories': ['spherical', 'diag', 'full'], 'name': 'covariance_type', 'type': 'categorical'}, {'high': 200, 'low': 50, 'name': 'n_iter', 'type': 'integer'}, {'high': 0.01, 'low': 1e-06, 'name': 'tol', 'type': 'continuous'}], 'lstm': [{'high': 256, 'low': 32, 'name': 'hidden_size', 'type': 'integer'}, {'high': 4, 'low': 1, 'name': 'num_layers', 'type': 'integer'}, {'high': 0.5, 'low': 0.0, 'name': 'dropout', 'type': 'continuous'}, {'high': 0.01, 'low': 0.0001, 'name': 'learning_rate', 'type': 'continuous'}, {'categories': [16, 32, 64, 128], 'name': 'batch_size', 'type': 'categorical'}], 'xgboost': [{'high': 500, 'low': 50, 'name': 'n_estimators', 'type': 'integer'}, {'high': 10, 'low': 3, 'name': 'max_depth', 'type': 'integer'}, {'high': 0.3, 'low': 0.01, 'name': 'learning_rate', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'subsample', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'colsample_bytree', 'type': 'continuous'}, {'high': 1.0, 'low': 0.0, 'name': 'reg_alpha', 'type': 'continuous'}, ...]}
E    +  where <ModelType.LSTM: 'lstm'> = ModelType.LSTM
E    +  and   {'gaussian_process': [{'categories': ['rb', 'matern', 'rational_quadratic'], 'name': 'kernel', 'type': 'categorical'}, {'high': 0.1, 'low': 1e-12, 'name': 'alpha', 'type': 'continuous'}, {'high': 10, 'low': 0, 'name': 'n_restarts_optimizer', 'type': 'integer'}], 'hmm': [{'high': 8, 'low': 2, 'name': 'n_states', 'type': 'integer'}, {'categories': ['spherical', 'diag', 'full'], 'name': 'covariance_type', 'type': 'categorical'}, {'high': 200, 'low': 50, 'name': 'n_iter', 'type': 'integer'}, {'high': 0.01, 'low': 1e-06, 'name': 'tol', 'type': 'continuous'}], 'lstm': [{'high': 256, 'low': 32, 'name': 'hidden_size', 'type': 'integer'}, {'high': 4, 'low': 1, 'name': 'num_layers', 'type': 'integer'}, {'high': 0.5, 'low': 0.0, 'name': 'dropout', 'type': 'continuous'}, {'high': 0.01, 'low': 0.0001, 'name': 'learning_rate', 'type': 'continuous'}, {'categories': [16, 32, 64, 128], 'name': 'batch_size', 'type': 'categorical'}], 'xgboost': [{'high': 500, 'low': 50, 'name': 'n_estimators', 'type': 'integer'}, {'high': 10, 'low': 3, 'name': 'max_depth', 'type': 'integer'}, {'high': 0.3, 'low': 0.01, 'name': 'learning_rate', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'subsample', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'colsample_bytree', 'type': 'continuous'}, {'high': 1.0, 'low': 0.0, 'name': 'reg_alpha', 'type': 'continuous'}, ...]} = <src.adaptation.optimizer.ModelOptimizer object at 0x7fed019157f0>._parameter_spaces
________ TestGlobalConfiguration.test_get_config_import_error_fallback _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1732: in test_get_config_import_error_fallback
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
_____ TestModelOptimization.test_model_optimizer_optimize_model_parameters _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1052: in test_model_optimizer_optimize_model_parameters
    mock_model.get_parameters.return_value = {"param1": 0.5}
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'get_parameters'
_________ TestModelOptimization.test_model_optimizer_cached_parameters _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1088: in test_model_optimizer_cached_parameters
    optimizer = ModelOptimizer()
E   TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
______ TestModelOptimization.test_model_optimizer_get_optimization_stats _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1105: in test_model_optimizer_get_optimization_stats
    optimizer = ModelOptimizer()
E   TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
___________ TestGlobalConfiguration.test_reload_config_forced_reload ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1767: in test_reload_config_forced_reload
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
__ TestModelOptimization.test_model_optimizer_parameter_space_initialization ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1131: in test_model_optimizer_parameter_space_initialization
    optimizer = ModelOptimizer()
E   TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
_______ TestModelOptimization.test_model_optimizer_should_optimize_logic _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1146: in test_model_optimizer_should_optimize_logic
    optimizer = ModelOptimizer()
E   TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
________________ TestModelOptimization.test_optimization_error _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1180: in test_optimization_error
    error = OptimizationError("Optimization failed", "OPT_001")
E   TypeError: OptimizationError.__init__() takes 2 positional arguments but 3 were given
_____ TestGlobalConfiguration.test_reload_config_environment_manager_path ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1795: in test_reload_config_environment_manager_path
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
___________ TestPerformanceTracking.test_real_time_metrics_creation ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1204: in test_real_time_metrics_creation
    metrics = RealTimeMetrics(
E   TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
____ TestConfigurationEdgeCases.test_config_loader_with_unicode_characters _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1848: in test_config_loader_with_unicode_characters
    result = loader._load_yaml("unicode.yaml")
src/core/config.py:482: in _load_yaml
    with open(file_path, "r", encoding="utf-8") as file:
E   FileNotFoundError: [Errno 2] No such file or directory: 'config/unicode.yaml'
_____ TestPerformanceTracking.test_real_time_metrics_overall_health_score ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1225: in test_real_time_metrics_overall_health_score
    good_metrics = RealTimeMetrics(
E   TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
__________ TestPerformanceTracking.test_real_time_metrics_is_healthy ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1250: in test_real_time_metrics_is_healthy
    healthy_metrics = RealTimeMetrics(
E   TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'accuracy_1h'
_______ TestConfigurationEdgeCases.test_environment_variable_edge_cases ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/core_system/test_configuration_system.py:1917: in test_environment_variable_edge_cases
    config = APIConfig()
<string>:11: in __init__
    ???
<string>:13: in __init__
    ???
src/core/config.py:205: in __post_init__
    raise ValueError(
E   ValueError: JWT is enabled but JWT_SECRET_KEY environment variable is not set
____________ TestPerformanceTracking.test_real_time_metrics_to_dict ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1270: in test_real_time_metrics_to_dict
    metrics = RealTimeMetrics(
E   TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
_____________ TestPerformanceTracking.test_accuracy_alert_creation _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1288: in test_accuracy_alert_creation
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_________ TestPerformanceTracking.test_accuracy_alert_age_calculation __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1311: in test_accuracy_alert_age_calculation
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_____________________ TestHAClient.test_ha_event_is_valid ______________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1968: in test_ha_event_is_valid
    valid_event = HAEvent(
E   TypeError: HAEvent.__init__() missing 2 required positional arguments: 'previous_state' and 'attributes'
_______ TestPerformanceTracking.test_accuracy_alert_requires_escalation ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1325: in test_accuracy_alert_requires_escalation
    recent_warning = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
___________ TestPerformanceTracking.test_accuracy_alert_acknowledge ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1346: in test_accuracy_alert_acknowledge
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_____________ TestHAClient.test_rate_limiter_acquire_rate_limited ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2029: in test_rate_limiter_acquire_rate_limited
    await limiter.acquire()
src/data/ingestion/ha_client.py:77: in acquire
    req_time for req_time in self.requests if req_time >= cutoff_time
E   TypeError: '>=' not supported between instances of 'float' and 'datetime.datetime'
_____________ TestPerformanceTracking.test_accuracy_alert_resolve ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1361: in test_accuracy_alert_resolve
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_____________ TestPerformanceTracking.test_accuracy_alert_escalate _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1376: in test_accuracy_alert_escalate
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_____________ TestPerformanceTracking.test_accuracy_alert_to_dict ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1392: in test_accuracy_alert_to_dict
    alert = AccuracyAlert(
E   TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
_________ TestPerformanceTracking.test_accuracy_tracker_initialization _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1411: in test_accuracy_tracker_initialization
    tracker = AccuracyTracker(
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'accuracy_thresholds'
_____ TestPerformanceTracking.test_accuracy_tracker_start_stop_monitoring ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1426: in test_accuracy_tracker_start_stop_monitoring
    tracker = AccuracyTracker(enable_background_tasks=False)
E   TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'enable_background_tasks'
_____ TestPerformanceTracking.test_accuracy_tracker_get_real_time_metrics ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1448: in test_accuracy_tracker_get_real_time_metrics
    mock_validator.extract_recent_validation_records.return_value = mock_records
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'extract_recent_validation_records'
_______ TestPerformanceTracking.test_accuracy_tracker_get_active_alerts ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1457: in test_accuracy_tracker_get_active_alerts
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
_______ TestPerformanceTracking.test_accuracy_tracker_acknowledge_alert ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1492: in test_accuracy_tracker_acknowledge_alert
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
______ TestPerformanceTracking.test_accuracy_tracker_get_accuracy_trends _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1510: in test_accuracy_tracker_get_accuracy_trends
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
______ TestPerformanceTracking.test_accuracy_tracker_export_tracking_data ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1535: in test_accuracy_tracker_export_tracking_data
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
_ TestPerformanceTracking.test_accuracy_tracker_add_remove_notification_callback _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1561: in test_accuracy_tracker_add_remove_notification_callback
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
_______ TestPerformanceTracking.test_accuracy_tracker_get_tracker_stats ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1580: in test_accuracy_tracker_get_tracker_stats
    tracker = AccuracyTracker()
E   TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
_____________ TestPerformanceTracking.test_accuracy_tracking_error _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1600: in test_accuracy_tracking_error
    error = AccuracyTrackingError("Tracking failed", "TRACK_001")
E   TypeError: AccuracyTrackingError.__init__() takes 2 positional arguments but 3 were given
_____________ TestTrackingManagement.test_tracking_config_creation _____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1610: in test_tracking_config_creation
    config = TrackingConfig(
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'prediction_validation_enabled'
_________ TestTrackingManagement.test_tracking_manager_initialization __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1641: in test_tracking_manager_initialization
    manager = TrackingManager(
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
___________ TestTrackingManagement.test_tracking_manager_initialize ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1664: in test_tracking_manager_initialize
    manager = TrackingManager(
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
_______ TestTrackingManagement.test_tracking_manager_start_stop_tracking _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1681: in test_tracking_manager_start_stop_tracking
    config = TrackingConfig(enabled=True, enable_background_tasks=False)
E   TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'enable_background_tasks'
________ TestTrackingManagement.test_tracking_manager_record_prediction ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1695: in test_tracking_manager_record_prediction
    manager = TrackingManager(config=config, prediction_validator=mock_validator)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
____ TestTrackingManagement.test_tracking_manager_handle_room_state_change _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1713: in test_tracking_manager_handle_room_state_change
    manager = TrackingManager(config=config, prediction_validator=mock_validator)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
_______ TestTrackingManagement.test_tracking_manager_get_tracking_status _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1736: in test_tracking_manager_get_tracking_status
    manager = TrackingManager(
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
______ TestTrackingManagement.test_tracking_manager_get_real_time_metrics ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1758: in test_tracking_manager_get_real_time_metrics
    manager = TrackingManager(config=config, accuracy_tracker=mock_tracker)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
________ TestTrackingManagement.test_tracking_manager_get_active_alerts ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1777: in test_tracking_manager_get_active_alerts
    manager = TrackingManager(config=config, accuracy_tracker=mock_tracker)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
________ TestTrackingManagement.test_tracking_manager_acknowledge_alert ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1800: in test_tracking_manager_acknowledge_alert
    manager = TrackingManager(config=config, accuracy_tracker=mock_tracker)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
___________ TestTrackingManagement.test_tracking_manager_check_drift ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1817: in test_tracking_manager_check_drift
    manager = TrackingManager(config=config, drift_detector=mock_drift_detector)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'drift_detector'
____ TestTrackingManagement.test_tracking_manager_request_manual_retraining ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1837: in test_tracking_manager_request_manual_retraining
    manager = TrackingManager(config=config, adaptive_retrainer=mock_retrainer)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
______ TestTrackingManagement.test_tracking_manager_get_retraining_status ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1855: in test_tracking_manager_get_retraining_status
    manager = TrackingManager(config=config, adaptive_retrainer=mock_retrainer)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
________ TestTrackingManagement.test_tracking_manager_cancel_retraining ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1875: in test_tracking_manager_cancel_retraining
    manager = TrackingManager(config=config, adaptive_retrainer=mock_retrainer)
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
________ TestTrackingManagement.test_tracking_manager_get_system_stats _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1891: in test_tracking_manager_get_system_stats
    manager = TrackingManager(
E   TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
______________ TestTrackingManagement.test_tracking_manager_error ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1913: in test_tracking_manager_error
    error = TrackingManagerError("Tracking manager failed", "TRACK_MGR_001")
E   TypeError: TrackingManagerError.__init__() takes 2 positional arguments but 3 were given
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_initialization _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1931: in test_monitoring_enhanced_tracking_manager_initialization
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_______________ TestBulkImporter.test_import_progress_properties _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2317: in test_import_progress_properties
    assert progress.entity_progress_percent() == 25.0
E   TypeError: 'float' object is not callable
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_method_wrapping _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1959: in test_monitoring_enhanced_tracking_manager_method_wrapping
    assert hasattr(enhanced_manager, "_original_record_prediction")
E   AssertionError: assert False
E    +  where False = hasattr(<src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fed019e72f0>, '_original_record_prediction')
_________ TestBulkImporter.test_import_progress_properties_edge_cases __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2329: in test_import_progress_properties_edge_cases
    assert progress.entity_progress_percent() == 0.0  # 0/0 case
E   TypeError: 'float' object is not callable
______________ TestBulkImporter.test_bulk_importer_initialization ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2365: in test_bulk_importer_initialization
    )  # ImportProgress
E   AttributeError: type object 'int' has no attribute '__origin__'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_monitored_record_prediction _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:1974: in test_monitoring_enhanced_tracking_manager_monitored_record_prediction
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_____________ TestBulkImporter.test_bulk_importer_load_resume_data _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2424: in test_bulk_importer_load_resume_data
    assert importer._resume_data == resume_data
E   AssertionError: assert {} == {'completed_e...ntities': 50}}
E     Right contains 2 more items:
E     {'completed_entities': ['sensor.test1', 'sensor.test2'],
E      'progress': {'processed_entities': 50}}
E     Full diff:
E       {
E     +  ,
E     -  'completed_entities': ['sensor.test1',
E     -                         'sensor.test2'],
E     -  'progress': {'processed_entities': 50},
E       }
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_monitored_validate_prediction _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2005: in test_monitoring_enhanced_tracking_manager_monitored_validate_prediction
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_____________ TestBulkImporter.test_bulk_importer_save_resume_data _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2457: in test_bulk_importer_save_resume_data
    assert mock_pickle_dump.called
E   AssertionError: assert False
E    +  where False = <MagicMock name='dump' id='140554916339328'>.called
__________ TestBulkImporter.test_bulk_importer_determine_sensor_type ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2669: in test_bulk_importer_determine_sensor_type
    assert sensor_type == "motion"
E   AssertionError: assert 'presence' == 'motion'
E     - motion
E     + presence
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_monitored_start_tracking _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2043: in test_monitoring_enhanced_tracking_manager_monitored_start_tracking
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_____________ TestBulkImporter.test_bulk_importer_update_progress ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:960: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'mock' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/data_layer/test_data_models.py:2695: in test_bulk_importer_update_progress
    mock_callback.assert_called_once_with(importer.progress)
E   AssertionError: Expected 'mock' to be called once. Called 0 times.
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_monitored_stop_tracking _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2066: in test_monitoring_enhanced_tracking_manager_monitored_stop_tracking
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
__________ TestBulkImporter.test_bulk_importer_generate_import_report __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2719: in test_bulk_importer_generate_import_report
    assert "import_summary" in report
E   TypeError: argument of type 'coroutine' is not iterable
____________ TestPatternDetection.test_statistical_pattern_analysis ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2889: in test_statistical_pattern_analysis
    assert mean_interval == 600.0  # 10 minutes
E   assert 600.000004 == 600.0
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_record_concept_drift _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2088: in test_monitoring_enhanced_tracking_manager_record_concept_drift
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
______________ TestPatternDetection.test_anomaly_detection_logic _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:2909: in test_anomaly_detection_logic
    assert len(outliers) == 1  # Should detect the 1-hour outlier
E   assert 0 == 1
E    +  where 0 = len([])
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_record_feature_computation _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2113: in test_monitoring_enhanced_tracking_manager_record_feature_computation
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_record_database_operation _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2136: in test_monitoring_enhanced_tracking_manager_record_database_operation
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_record_mqtt_publish _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2162: in test_monitoring_enhanced_tracking_manager_record_mqtt_publish
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_update_connection_status _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2189: in test_monitoring_enhanced_tracking_manager_update_connection_status
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_get_monitoring_status _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2229: in test_monitoring_enhanced_tracking_manager_get_monitoring_status
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_track_model_training _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2250: in test_monitoring_enhanced_tracking_manager_track_model_training
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_monitoring_enhanced_tracking_manager_getattr_delegation _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2277: in test_monitoring_enhanced_tracking_manager_getattr_delegation
    enhanced_manager = MonitoringEnhancedTrackingManager(
src/adaptation/monitoring_enhanced_tracking.py:31: in __init__
    self._wrap_tracking_methods()
src/adaptation/monitoring_enhanced_tracking.py:38: in _wrap_tracking_methods
    "validate_prediction": self.tracking_manager.validate_prediction,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:660: in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
E   AttributeError: Mock object has no attribute 'validate_prediction'
_ TestEnhancedMonitoring.test_create_monitoring_enhanced_tracking_manager_factory _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/adaptation/test_model_adaptation.py:2301: in test_create_monitoring_enhanced_tracking_manager_factory
    enhanced_manager = create_monitoring_enhanced_tracking_manager(
E   TypeError: create_monitoring_enhanced_tracking_manager() missing 1 required positional argument: 'config'
________ TestPredictionModel.test_prediction_analyze_confidence_spread _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_data_models.py:1154: in test_prediction_analyze_confidence_spread
    assert spread["mean"] == 0.7  # (0.8 + 0.6 + 0.7) / 3
E   assert 0.7000000000000001 == 0.7
__________ TestTemporalFeatures.test_extract_features_error_handling ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:305: in test_extract_features_error_handling
    assert exc_info.value.feature_type == "temporal"
E   AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
------------------------------ Captured log call -------------------------------
ERROR    src.features.temporal:temporal.py:111 Failed to extract temporal features: unsupported operand type(s) for -: 'datetime.datetime' and 'str'
_ TestSequentialFeatures.test_extract_velocity_features_with_numpy_operations __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:421: in test_extract_velocity_features_with_numpy_operations
    assert "movement_velocity" in features
E   AssertionError: assert 'movement_velocity' in {'avg_event_interval': 300.0, 'burst_ratio': 300.0, 'event_interval_variance': 0.0, 'interval_autocorr': 0.0, ...}
____ TestSequentialFeatures.test_get_default_features_returns_complete_dict ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:448: in test_get_default_features_returns_complete_dict
    assert "movement_velocity" in defaults
E   AssertionError: assert 'movement_velocity' in {'active_room_count': 1.0, 'avg_event_interval': 300.0, 'avg_room_dwell_time': 1800.0, 'burst_ratio': 0.0, ...}
_________ TestSequentialFeatures.test_extract_features_error_handling __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:469: in test_extract_features_error_handling
    assert exc_info.value.feature_type == "sequential"
E   AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
------------------------------ Captured log call -------------------------------
ERROR    src.features.sequential:sequential.py:136 Failed to extract sequential features: '>=' not supported between instances of 'NoneType' and 'datetime.datetime'
_ TestContextualFeatures.test_extract_environmental_features_with_climate_sensors _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:546: in test_extract_environmental_features_with_climate_sensors
    features = contextual_extractor._extract_environmental_features(env_events)
E   TypeError: ContextualFeatureExtractor._extract_environmental_features() missing 1 required positional argument: 'target_time'
__ TestContextualFeatures.test_extract_door_state_features_with_door_sensors ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:570: in test_extract_door_state_features_with_door_sensors
    features = contextual_extractor._extract_door_state_features(door_events)
E   TypeError: ContextualFeatureExtractor._extract_door_state_features() missing 1 required positional argument: 'target_time'
___ TestContextualFeatures.test_extract_multi_room_features_with_correlation ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:615: in test_extract_multi_room_features_with_correlation
    features = contextual_extractor._extract_multi_room_features(
E   TypeError: ContextualFeatureExtractor._extract_multi_room_features() missing 1 required positional argument: 'target_time'
_________ TestContextualFeatures.test_extract_features_error_handling __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:673: in test_extract_features_error_handling
    assert exc_info.value.feature_type == "contextual"
E   AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
------------------------------ Captured log call -------------------------------
ERROR    src.features.contextual:contextual.py:137 Failed to extract contextual features: '>=' not supported between instances of 'str' and 'datetime.datetime'
____ TestFeatureEngineering.test_extract_features_validation_empty_room_id _____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:738: in test_extract_features_validation_empty_room_id
    assert "room_id cannot be empty" in str(exc_info.value)
E   AssertionError: assert 'room_id cannot be empty' in 'Feature extraction failed: general for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=general, room_id=unknown | Caused by: FeatureExtractionError: Feature extraction failed: validation for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=validation, room_id=unknown | Caused by: ValueError: Room ID is required'
E    +  where 'Feature extraction failed: general for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=general, room_id=unknown | Caused by: FeatureExtractionError: Feature extraction failed: validation for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=validation, room_id=unknown | Caused by: ValueError: Room ID is required' = str(FeatureExtractionError('Feature extraction failed: general for room unknown'))
E    +    where FeatureExtractionError('Feature extraction failed: general for room unknown') = <ExceptionInfo FeatureExtractionError('Feature extraction failed: general for room unknown') tblen=2>.value
------------------------------ Captured log call -------------------------------
ERROR    src.features.engineering:engineering.py:202 Feature extraction failed for room : Feature extraction failed: validation for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=validation, room_id=unknown | Caused by: ValueError: Room ID is required
______ TestFeatureEngineering.test_extract_features_sequential_processing ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:776: in test_extract_features_sequential_processing
    assert "sequential_seq_feature" in features
E   AssertionError: assert 'sequential_seq_feature' in {'meta_data_quality_score': 0.0, 'meta_event_count': 0.0, 'meta_extraction_day_of_week': 0.0, 'meta_extraction_hour': 12.0, ...}
------------------------------ Captured log call -------------------------------
ERROR    src.features.engineering:engineering.py:408 Failed to extract sequential features: Mock object has no attribute 'room_id'
ERROR    src.features.engineering:engineering.py:426 Failed to extract contextual features: Mock object has no attribute 'room_id'
_____ TestFeatureEngineering.test_validate_configuration_with_none_config ______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:815: in test_validate_configuration_with_none_config
    assert result is True  # Should pass validation
E   assert <coroutine object FeatureEngineeringEngine.validate_configuration at 0x7fed024bae30> is True
_ TestFeatureEngineering.test_validate_configuration_with_invalid_max_workers __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:830: in test_validate_configuration_with_invalid_max_workers
    assert "max_workers must be >= 1" in str(exc_info.value)
E   AssertionError: assert 'max_workers must be >= 1' in 'max_workers must be at least 1, got 0 | Error Code: FEATURE_ENGINE_INVALID_WORKERS'
E    +  where 'max_workers must be at least 1, got 0 | Error Code: FEATURE_ENGINE_INVALID_WORKERS' = str(ConfigurationError('max_workers must be at least 1, got 0'))
E    +    where ConfigurationError('max_workers must be at least 1, got 0') = <ExceptionInfo ConfigurationError('max_workers must be at least 1, got 0') tblen=3>.value
------------------------------ Captured log call -------------------------------
WARNING  src.features.engineering:engineering.py:730 No room configurations available - feature extraction may be limited
___ TestFeatureEngineering.test_compute_feature_correlations_with_dataframe ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:846: in test_compute_feature_correlations_with_dataframe
    result = feature_engine.compute_feature_correlations(feature_dicts)
src/features/engineering.py:628: in compute_feature_correlations
    if feature_matrix.empty:
E   AttributeError: 'list' object has no attribute 'empty'
__________ TestFeatureEngineering.test_destructor_shuts_down_executor __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:949: in assert_called_with
    raise AssertionError(_error_message()) from cause
E   AssertionError: expected call not found.
E   Expected: shutdown(wait=True)
E     Actual: shutdown(wait=False)

During handling of the above exception, another exception occurred:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:961: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
E   AssertionError: expected call not found.
E   Expected: shutdown(wait=True)
E     Actual: shutdown(wait=False)
E   
E   pytest introspection follows:
E   
E   Kwargs:
E   assert {'wait': False} == {'wait': True}
E     Differing items:
E     {'wait': False} != {'wait': True}
E     Full diff:
E     - {'wait': True}
E     ?          ^^^
E     + {'wait': False}
E     ?          ^^^^

During handling of the above exception, another exception occurred:
tests/unit/feature_engineering/test_feature_extraction.py:890: in test_destructor_shuts_down_executor
    mock_executor.shutdown.assert_called_once_with(wait=True)
E   AssertionError: expected call not found.
E   Expected: shutdown(wait=True)
E     Actual: shutdown(wait=False)
E   
E   pytest introspection follows:
E   
E   Kwargs:
E   assert {'wait': False} == {'wait': True}
E     Differing items:
E     {'wait': False} != {'wait': True}
E     Full diff:
E     - {'wait': True}
E     ?          ^^^
E     + {'wait': False}
E     ?          ^^^^
_______ TestFeatureStore.test_feature_record_is_valid_with_mock_datetime _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1393: in patched
    with self.decoration_helper(patched,
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:137: in __enter__
    return next(self.gen)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1375: in decoration_helper
    arg = exit_stack.enter_context(patching)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:526: in enter_context
    result = _enter(cm)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'datetime' from '/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/datetime.py'> does not have the attribute 'now'
_________________ TestFeatureStore.test_feature_cache_make_key _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:961: in test_feature_cache_make_key
    assert "living_room" in key
E   AssertionError: assert 'living_room' in 'a3a903e89a5ba1c24bc46f840d5fcf02'
_______________ TestFeatureStore.test_feature_cache_put_and_get ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:969: in test_feature_cache_put_and_get
    feature_cache.put(key, feature_record)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
_______________ TestFeatureStore.test_feature_cache_lru_eviction _______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:987: in test_feature_cache_lru_eviction
    cache.put("key1", feature_record)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
________________ TestFeatureStore.test_feature_cache_get_stats _________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1001: in test_feature_cache_get_stats
    feature_cache.put("key1", feature_record)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
__________________ TestFeatureStore.test_feature_cache_clear ___________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1015: in test_feature_cache_clear
    feature_cache.put("key1", feature_record)
E   TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
______________ TestFeatureStore.test_feature_store_initialization ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1033: in test_feature_store_initialization
    store = FeatureStore(persist_features=False)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
_____________ TestFeatureStore.test_feature_store_context_manager ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'initialize' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/feature_engineering/test_feature_extraction.py:1055: in test_feature_store_context_manager
    mock_db_manager.initialize.assert_called_once()
E   AssertionError: Expected 'initialize' to have been called once. Called 0 times.
------------------------------ Captured log call -------------------------------
WARNING  src.features.engineering:engineering.py:730 No room configurations available - feature extraction may be limited
_______________ TestFeatureStore.test_feature_store_health_check _______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1065: in test_feature_store_health_check
    store = FeatureStore(persist_features=False)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
______________ TestFeatureStore.test_feature_store_get_statistics ______________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1083: in test_feature_store_get_statistics
    store = FeatureStore(persist_features=False)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
_______________ TestFeatureStore.test_feature_store_clear_cache ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1102: in test_feature_store_clear_cache
    store = FeatureStore(persist_features=False)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
_______________ TestFeatureStore.test_feature_store_reset_stats ________________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:1118: in test_feature_store_reset_stats
    store = FeatureStore(persist_features=False)
E   TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
_________ TestMQTTPublisher.test_mqtt_publisher_get_connection_status __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:398: in test_mqtt_publisher_get_connection_status
    mock_datetime.utcnow.return_value = past_time.replace(
E   ValueError: second must be in 0..59
________ TestMQTTPublisher.test_mqtt_publisher_connect_to_broker_retry _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:516: in test_mqtt_publisher_connect_to_broker_retry
    assert publisher.connection_status.connection_attempts == 2
E   AssertionError: assert 1 == 2
E    +  where 1 = MQTTConnectionStatus(connected=True, last_connected=None, last_disconnected=None, connection_attempts=1, last_error='Connection failed', reconnect_count=0, uptime_seconds=0.0).connection_attempts
E    +    where MQTTConnectionStatus(connected=True, last_connected=None, last_disconnected=None, connection_attempts=1, last_error='Connection failed', reconnect_count=0, uptime_seconds=0.0) = <src.integration.mqtt_publisher.MQTTPublisher object at 0x7fed018107d0>.connection_status
------------------------------ Captured log call -------------------------------
ERROR    src.integration.mqtt_publisher:mqtt_publisher.py:443 Failed to connect to MQTT broker (attempt 1): Connection failed
________ TestMQTTIntegrationManager.test_integration_manager_get_stats _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:1031: in test_integration_manager_get_stats
    assert system_health["overall_status"] == "healthy"
E   AssertionError: assert 'degraded' == 'healthy'
E     - healthy
E     + degraded
_______ TestGlobalDatabaseFunctions.test_get_db_session_context_manager ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:974: in test_get_db_session_context_manager
    async with get_db_session() as session:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:210: in __aenter__
    return await anext(self.gen)
src/data/storage/database.py:839: in get_db_session
    async with db_manager.get_session() as session:
E   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
_______ TestDatabaseCompatibility.test_is_sqlite_engine_with_url_object ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1221: in test_is_sqlite_engine_with_url_object
    assert is_sqlite_engine(mock_engine) is True
src/data/storage/database_compatibility.py:17: in is_sqlite_engine
    url_str = str(engine_or_url.url)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:2022: in method
    return func(self, *args, **kw)
E   TypeError: TestDatabaseCompatibility.test_is_sqlite_engine_with_url_object.<locals>.<lambda>() takes 0 positional arguments but 1 was given
___ TestDatabaseCompatibility.test_is_postgresql_engine_with_postgresql_url ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1235: in test_is_postgresql_engine_with_postgresql_url
    assert is_postgresql_engine(mock_engine) is True
E   AssertionError: assert False is True
E    +  where False = <function is_postgresql_engine at 0x7fd57eaffba0>(<Mock id='140554921152096'>)
______ TestDatabaseCompatibility.test_configure_sensor_event_model_sqlite ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1260: in test_configure_sensor_event_model_sqlite
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database_compatibility' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database_compatibility.py'> does not have the attribute 'SensorEvent'
____ TestDatabaseCompatibility.test_configure_sensor_event_model_postgresql ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1284: in test_configure_sensor_event_model_postgresql
    with patch(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.database_compatibility' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database_compatibility.py'> does not have the attribute 'SensorEvent'
_ TestDatabaseCompatibility.test_create_database_specific_models_with_sensor_event _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1311: in test_create_database_specific_models_with_sensor_event
    create_database_specific_models(mock_engine, models)
src/data/storage/database_compatibility.py:99: in create_database_specific_models
    for name, model_class in base_model_classes.items():
E   TypeError: 'Mock' object is not iterable
_____ TestDatabaseCompatibility.test_create_database_specific_models_empty _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1326: in test_create_database_specific_models_empty
    create_database_specific_models(mock_engine, models)
src/data/storage/database_compatibility.py:99: in create_database_specific_models
    for name, model_class in base_model_classes.items():
E   TypeError: 'Mock' object is not iterable
_____ TestDatabaseCompatibility.test_patch_models_for_sqlite_compatibility _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1341: in test_patch_models_for_sqlite_compatibility
    patch_models_for_sqlite_compatibility([mock_model])
E   TypeError: patch_models_for_sqlite_compatibility() takes 0 positional arguments but 1 was given
_________ TestDatabaseCompatibility.test_configure_sqlite_for_testing __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1384: in test_configure_sqlite_for_testing
    configure_sqlite_for_testing(mock_connection, None)
src/data/storage/database_compatibility.py:113: in configure_sqlite_for_testing
    if "sqlite" in str(connection_record.info.get("url", "")):
E   AttributeError: 'NoneType' object has no attribute 'info'
__ TestDatabaseCompatibility.test_configure_database_on_first_connect_sqlite ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1412: in test_configure_database_on_first_connect_sqlite
    configure_database_on_first_connect(mock_connection, None, mock_engine)
E   TypeError: configure_database_on_first_connect() takes 2 positional arguments but 3 were given
_ TestDatabaseCompatibility.test_configure_database_on_first_connect_postgresql _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1430: in test_configure_database_on_first_connect_postgresql
    configure_database_on_first_connect(mock_connection, None, mock_engine)
E   TypeError: configure_database_on_first_connect() takes 2 positional arguments but 3 were given
___ TestDialectUtils.test_database_dialect_utils_get_dialect_name_postgresql ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1446: in test_database_dialect_utils_get_dialect_name_postgresql
    utils = DatabaseDialectUtils(mock_engine)
E   TypeError: DatabaseDialectUtils() takes no arguments
_____ TestDialectUtils.test_database_dialect_utils_get_dialect_name_sqlite _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1456: in test_database_dialect_utils_get_dialect_name_sqlite
    utils = DatabaseDialectUtils(mock_engine)
E   TypeError: DatabaseDialectUtils() takes no arguments
__________ TestDialectUtils.test_database_dialect_utils_is_postgresql __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1467: in test_database_dialect_utils_is_postgresql
    utils = DatabaseDialectUtils(mock_pg_engine)
E   TypeError: DatabaseDialectUtils() takes no arguments
____________ TestDialectUtils.test_database_dialect_utils_is_sqlite ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1485: in test_database_dialect_utils_is_sqlite
    utils = DatabaseDialectUtils(mock_sqlite_engine)
E   TypeError: DatabaseDialectUtils() takes no arguments
____ TestDialectUtils.test_statistical_functions_percentile_cont_postgresql ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1502: in test_statistical_functions_percentile_cont_postgresql
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
__ TestDialectUtils.test_statistical_functions_percentile_cont_sqlite_median ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1520: in test_statistical_functions_percentile_cont_sqlite_median
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
_ TestDialectUtils.test_statistical_functions_percentile_cont_sqlite_quartile __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1538: in test_statistical_functions_percentile_cont_sqlite_quartile
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
___ TestDialectUtils.test_statistical_functions_percentile_cont_sqlite_other ___
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1558: in test_statistical_functions_percentile_cont_sqlite_other
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
______ TestDialectUtils.test_statistical_functions_stddev_samp_postgresql ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1578: in test_statistical_functions_stddev_samp_postgresql
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
________ TestDialectUtils.test_statistical_functions_stddev_samp_sqlite ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1596: in test_statistical_functions_stddev_samp_sqlite
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
_____ TestDialectUtils.test_statistical_functions_extract_epoch_postgresql _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1614: in test_statistical_functions_extract_epoch_postgresql
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
_______ TestDialectUtils.test_statistical_functions_extract_epoch_sqlite _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1632: in test_statistical_functions_extract_epoch_sqlite
    stats = StatisticalFunctions(mock_engine)
E   TypeError: StatisticalFunctions() takes no arguments
______________ TestDialectUtils.test_query_builder_initialization ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1653: in test_query_builder_initialization
    assert builder.dialect_name == "postgresql"
E   AttributeError: 'QueryBuilder' object has no attribute 'dialect_name'
______ TestDialectUtils.test_query_builder_build_percentile_query_single _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1675: in test_query_builder_build_percentile_query_single
    with patch("src.data.storage.dialect_utils.select") as mock_select:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
_____ TestDialectUtils.test_query_builder_build_percentile_query_multiple ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1705: in test_query_builder_build_percentile_query_multiple
    with patch("src.data.storage.dialect_utils.select") as mock_select:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
_______ TestDialectUtils.test_query_builder_build_statistics_query_basic _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1727: in test_query_builder_build_statistics_query_basic
    with patch("src.data.storage.dialect_utils.select") as mock_select:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
_ TestDialectUtils.test_query_builder_build_statistics_query_with_percentiles __
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1763: in test_query_builder_build_statistics_query_with_percentiles
    with patch("src.data.storage.dialect_utils.select") as mock_select:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1467: in __enter__
    original, local = self.get_original()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1437: in get_original
    raise AttributeError(
E   AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
__________ TestDialectUtils.test_compatibility_manager_initialization __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1786: in test_compatibility_manager_initialization
    assert hasattr(manager, "dialect_utils")
E   AssertionError: assert False
E    +  where False = hasattr(<src.data.storage.dialect_utils.CompatibilityManager object at 0x7fd57e221b50>, 'dialect_utils')
__________ TestDialectUtils.test_global_utility_functions_with_engine __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1861: in test_global_utility_functions_with_engine
    result3 = extract_epoch_interval(mock_interval, engine=mock_engine)
E   TypeError: extract_epoch_interval() missing 1 required positional argument: 'end_time'
________ TestDialectUtils.test_global_utility_functions_without_engine _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/data_layer/test_database_operations.py:1899: in test_global_utility_functions_without_engine
    result3 = extract_epoch_interval(mock_interval)
E   TypeError: extract_epoch_interval() missing 1 required positional argument: 'end_time'
___________ TestDialectUtils.test_global_utility_functions_fallback ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/data/storage/dialect_utils.py:322: in stddev_samp
    manager = get_compatibility_manager()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1198: in _execute_mock_call
    raise effect
src/data/storage/dialect_utils.py:290: in percentile_cont
    manager = get_compatibility_manager()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1198: in _execute_mock_call
    raise effect
E   RuntimeError: Not initialized

During handling of the above exception, another exception occurred:
tests/unit/data_layer/test_database_operations.py:1928: in test_global_utility_functions_fallback
    result2 = stddev_samp(
src/data/storage/dialect_utils.py:327: in stddev_samp
    avg_squared = sql_func.avg(column * column)
E   TypeError: unsupported operand type(s) for *: 'Mock' and 'Mock'
_____ TestTemporalFeatures.test_extract_historical_patterns_with_dataframe _____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/feature_engineering/test_feature_extraction.py:220: in test_extract_historical_patterns_with_dataframe
    mock_df_instance.groupby.return_value.__getitem__.return_value.agg.return_value.fillna.return_value = pd.DataFrame(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:662: in __getattr__
    raise AttributeError(name)
E   AttributeError: __getitem__
_____________ TestLSTMPredictor.test_lstm_train_insufficient_data ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/lstm_predictor.py:145: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ValueError: Insufficient sequence data: only 0 sequences available

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1114: in test_lstm_train_insufficient_data
    result = await predictor.train(features, targets)
src/models/base/lstm_predictor.py:294: in train
    raise ModelTrainingError(model_type="lstm", room_id=self.room_id, cause=e)
E   src.core.exceptions.ModelTrainingError: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ModelTrainingError: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ValueError: Insufficient sequence data: only 0 sequences available
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.lstm_predictor:lstm_predictor.py:283 LSTM training failed: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ValueError: Insufficient sequence data: only 0 sequences available
_________________ TestLSTMPredictor.test_lstm_predict_success __________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1163: in test_lstm_predict_success
    assert len(results) == 1
E   AssertionError: assert 3 == 1
E    +  where 3 = len([PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'})])
__________ TestLSTMPredictor.test_create_sequences_insufficient_data ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1227: in test_create_sequences_insufficient_data
    X_seq, y_seq = predictor._create_sequences(features, targets)
src/models/base/lstm_predictor.py:490: in _create_sequences
    raise ValueError(
E   ValueError: Need at least 50 samples for sequence generation, got 10
_________________ TestLSTMPredictor.test_lstm_save_load_model __________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1280: in test_lstm_save_load_model
    assert success is True
E   assert False is True
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.lstm_predictor:lstm_predictor.py:679 Failed to save LSTM model: Can't pickle <class 'unittest.mock.Mock'>: it's not the same object as unittest.mock.Mock
_______________ TestXGBoostPredictor.test_xgboost_train_success ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/xgboost_predictor.py:156: in train
    for importance in self.model.feature_importances_
E   TypeError: 'Mock' object is not iterable

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1358: in test_xgboost_train_success
    result = await predictor.train(features, targets)
src/models/base/xgboost_predictor.py:256: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: xgboost for room living_room | Error Code: MODEL_TRAINING_ERROR | Context: model_type=xgboost, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.xgboost_predictor:xgboost_predictor.py:245 XGBoost training failed: 'Mock' object is not iterable
_______________ TestXGBoostPredictor.test_xgboost_train_failure ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/xgboost_predictor.py:101: in train
    y_train = self._prepare_targets(targets)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1198: in _execute_mock_call
    raise effect
E   Exception: Training failed

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1384: in test_xgboost_train_failure
    result = await predictor.train(features, targets)
src/models/base/xgboost_predictor.py:256: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: xgboost for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=xgboost, room_id=unknown | Caused by: Exception: Training failed
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.xgboost_predictor:xgboost_predictor.py:245 XGBoost training failed: Training failed
______________ TestXGBoostPredictor.test_xgboost_predict_success _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/xgboost_predictor.py:304: in predict
    predicted_time = prediction_time + timedelta(
E   TypeError: unsupported type for timedelta seconds component: numpy.int64

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1413: in test_xgboost_predict_success
    results = await predictor.predict(
src/models/base/xgboost_predictor.py:349: in predict
    raise ModelPredictionError(
E   src.core.exceptions.ModelPredictionError: Model prediction failed: xgboost for room bedroom | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=xgboost, room_id=bedroom | Caused by: TypeError: unsupported type for timedelta seconds component: numpy.int64
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.xgboost_predictor:xgboost_predictor.py:348 XGBoost prediction failed: unsupported type for timedelta seconds component: numpy.int64
_______ TestXGBoostPredictor.test_xgboost_get_feature_importance_trained _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1508: in test_xgboost_get_feature_importance_trained
    assert len(importance) == 3
E   assert 0 == 3
E    +  where 0 = len({})
______________ TestXGBoostPredictor.test_xgboost_save_load_model _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1552: in test_xgboost_save_load_model
    assert success is True
E   assert False is True
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.xgboost_predictor:xgboost_predictor.py:629 Failed to save XGBoost model: Can't pickle <class 'unittest.mock.Mock'>: it's not the same object as unittest.mock.Mock
______________ TestHMMPredictor.test_hmm_train_insufficient_data _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/hmm_predictor.py:130: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1707: in test_hmm_train_insufficient_data
    result = await predictor.train(features, targets)
src/models/base/hmm_predictor.py:277: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown | Caused by: ModelTrainingError: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.hmm_predictor:hmm_predictor.py:266 HMM training failed: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown
__________________ TestHMMPredictor.test_hmm_predict_success ___________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1740: in test_hmm_predict_success
    results = await predictor.predict(
src/models/base/hmm_predictor.py:299: in predict
    raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")
E   src.core.exceptions.ModelPredictionError: Model prediction failed: hmm for room laundry | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=hmm, room_id=laundry
_____________________ TestHMMPredictor.test_analyze_states _____________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1760: in test_analyze_states
    predictor._analyze_states(state_predictions, target_values)
E   TypeError: HMMPredictor._analyze_states() missing 3 required positional arguments: 'durations', 'feature_names', and 'state_probabilities'
________________ TestHMMPredictor.test_build_transition_matrix _________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1775: in test_build_transition_matrix
    assert isinstance(transition_matrix, np.ndarray)
E   AssertionError: assert False
E    +  where False = isinstance(None, <class 'numpy.ndarray'>)
E    +    where <class 'numpy.ndarray'> = np.ndarray
___________ TestHMMPredictor.test_hmm_get_feature_importance_trained ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1807: in test_hmm_get_feature_importance_trained
    assert len(importance) == 2
E   assert 0 == 2
E    +  where 0 = len({})
_____________________ TestHMMPredictor.test_get_state_info _____________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:1835: in test_get_state_info
    assert "labels" in info
E   AssertionError: assert 'labels' in {'n_states': 4, 'state_characteristics': {0: {'mean_duration': 600}, 1: {'mean_duration': 1800}, 2: {'mean_duration': 3600}}, 'state_labels': ['vacant_short', 'occupied', 'vacant_long'], 'transition_matrix': [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.4, 0.2, 0.4]]}
_________ TestGaussianProcessPredictor.test_gp_train_insufficient_data _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/gp_predictor.py:269: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ValueError: Insufficient training data: only 8 samples available

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1953: in test_gp_train_insufficient_data
    result = await predictor.train(features, targets)
src/models/base/gp_predictor.py:427: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ModelTrainingError: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ValueError: Insufficient training data: only 8 samples available
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.gp_predictor:gp_predictor.py:416 GP training failed: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ValueError: Insufficient training data: only 8 samples available
________ TestGaussianProcessPredictor.test_gp_predict_with_uncertainty _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/base/gp_predictor.py:470: in predict
    predicted_time = prediction_time + timedelta(seconds=mean_time_until)
E   TypeError: unsupported type for timedelta seconds component: numpy.int64

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:1987: in test_gp_predict_with_uncertainty
    results = await predictor.predict(
src/models/base/gp_predictor.py:541: in predict
    raise ModelPredictionError(
E   src.core.exceptions.ModelPredictionError: Model prediction failed: gp for room greenhouse | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=gp, room_id=greenhouse | Caused by: TypeError: unsupported type for timedelta seconds component: numpy.int64
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.gp_predictor:gp_predictor.py:540 GP prediction failed: unsupported type for timedelta seconds component: numpy.int64
_______________ TestGaussianProcessPredictor.test_create_kernel ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2012: in test_create_kernel
    kernel = predictor._create_kernel(kernel_type, n_features=3)
E   TypeError: GaussianProcessPredictor._create_kernel() got multiple values for argument 'n_features'
_______ TestGaussianProcessPredictor.test_gp_get_feature_importance_ard ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2046: in test_gp_get_feature_importance_ard
    assert importance["feature1"] > importance["feature2"]  # 0.5 < 1.2
E   assert 0.3333333333333333 > 0.3333333333333333
------------------------------ Captured log call -------------------------------
WARNING  src.models.base.gp_predictor:gp_predictor.py:592 Could not calculate feature importance: argument of type 'Mock' is not iterable
____________ TestGaussianProcessPredictor.test_uncertainty_metrics _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2059: in test_uncertainty_metrics
    assert "calibration" in metrics or "uncertainty_range" in metrics
E   AssertionError: assert ('calibration' in {'confidence_intervals': [0.68, 0.95], 'kernel_type': 'rb', 'log_marginal_likelihood': None, 'sparse_gp': False, ...} or 'uncertainty_range' in {'confidence_intervals': [0.68, 0.95], 'kernel_type': 'rb', 'log_marginal_likelihood': None, 'sparse_gp': False, ...})
_____________ TestGaussianProcessPredictor.test_gp_save_load_model _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2075: in test_gp_save_load_model
    assert success is True
E   assert False is True
------------------------------ Captured log call -------------------------------
ERROR    src.models.base.gp_predictor:gp_predictor.py:1083 Failed to save GP model: 'GaussianProcessPredictor' object has no attribute 'gp_model'
________________ TestEnsembleModel.test_ensemble_train_success _________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/ensemble.py:132: in train
    self._validate_training_data(
src/models/ensemble.py:1218: in _validate_training_data
    raise ValueError(
E   ValueError: Targets missing required columns: {'target_time', 'transition_type'}. Available columns: ['time_until_transition_seconds']

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:2169: in test_ensemble_train_success
    result = await ensemble.train(features, targets)
src/models/ensemble.py:254: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: ensemble for room lobby | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=lobby | Caused by: ValueError: Targets missing required columns: {'target_time', 'transition_type'}. Available columns: ['time_until_transition_seconds']
------------------------------ Captured log call -------------------------------
ERROR    src.models.ensemble:ensemble.py:243 Ensemble training failed: Targets missing required columns: {'target_time', 'transition_type'}. Available columns: ['time_until_transition_seconds']
___________ TestEnsembleModel.test_ensemble_train_insufficient_data ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
src/models/ensemble.py:132: in train
    self._validate_training_data(
src/models/ensemble.py:1188: in _validate_training_data
    raise ValueError(
E   ValueError: Insufficient data for ensemble training: only 30 samples. Need at least 50 samples.

During handling of the above exception, another exception occurred:
tests/unit/ml_models/test_predictive_models.py:2186: in test_ensemble_train_insufficient_data
    result = await ensemble.train(features, targets)
src/models/ensemble.py:254: in train
    raise ModelTrainingError(
E   src.core.exceptions.ModelTrainingError: Model training failed: ensemble for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=unknown | Caused by: ValueError: Insufficient data for ensemble training: only 30 samples. Need at least 50 samples.
------------------------------ Captured log call -------------------------------
ERROR    src.models.ensemble:ensemble.py:243 Ensemble training failed: Insufficient data for ensemble training: only 30 samples. Need at least 50 samples.
_______________ TestEnsembleModel.test_ensemble_predict_success ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2236: in test_ensemble_predict_success
    results = await ensemble.predict(features, prediction_time)
src/models/ensemble.py:276: in predict
    raise ModelPredictionError(
E   src.core.exceptions.ModelPredictionError: Model prediction failed: ensemble for room unknown | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=ensemble, room_id=unknown
____________ TestEnsembleModel.test_ensemble_get_feature_importance ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2282: in test_ensemble_get_feature_importance
    assert len(importance) == 3
E   assert 0 == 3
E    +  where 0 = len({})
_______________ TestEnsembleModel.test_ensemble_save_load_model ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_predictive_models.py:2305: in test_ensemble_save_load_model
    assert success is True
E   assert False is True
------------------------------ Captured log call -------------------------------
ERROR    src.models.ensemble:ensemble.py:1327 Failed to save ensemble model: 'list' object has no attribute 'items'
____________________ TestTrainingProgress.test_update_stage ____________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:226: in test_update_stage
    progress = TrainingProgress(
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_______________ TestTrainingProgress.test_stage_progress_mapping _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:248: in test_stage_progress_mapping
    progress = TrainingProgress(
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
______________ TestModelTrainingPipeline.test_retraining_pipeline ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:499: in test_retraining_pipeline
    mock_progress = TrainingProgress(
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_________ TestModelTrainingPipeline.test_retraining_with_full_retrain __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:540: in test_retraining_with_full_retrain
    mock_progress = TrainingProgress(
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
______________ TestDataSplittingStrategies.test_time_series_split ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:982: in test_time_series_split
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
___________ TestDataSplittingStrategies.test_expanding_window_split ____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1028: in test_expanding_window_split
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
____________ TestDataSplittingStrategies.test_rolling_window_split _____________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1065: in test_rolling_window_split
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
________________ TestDataSplittingStrategies.test_holdout_split ________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1102: in test_holdout_split
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_________ TestDataSplittingStrategies.test_split_insufficient_samples __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1140: in test_split_insufficient_samples
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
______________ TestModelTraining.test_train_models_specific_type _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1218: in test_train_models_specific_type
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_________________ TestModelTraining.test_train_models_failure __________________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1255: in test_train_models_failure
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_____ TestModelTraining.test_evaluate_and_select_best_model_empty_results ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1361: in test_evaluate_and_select_best_model_empty_results
    progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_______________ TestPipelineManagement.test_get_active_pipelines _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1569: in test_get_active_pipelines
    test_progress = TrainingProgress("test-123", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
_______________ TestPipelineManagement.test_get_pipeline_history _______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:1594: in test_get_pipeline_history
    test_progress = TrainingProgress(f"test-{i}", TrainingType.INITIAL)
E   TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
__________ TestTrainingIntegrationManager.test_get_cooldown_remaining __________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/ml_models/test_training_pipeline.py:2100: in test_get_cooldown_remaining
    assert (
E   assert 5.999999996944444 == 6.0
________ TestStructuredLogging.test_performance_logger_operation_timing ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:216: in test_performance_logger_operation_timing
    assert "Test_operation" in call_args[0][0]  # Message content
E   AssertionError: assert 'Test_operation' in 'Operation completed: test_operation'
______ TestPerformanceMetrics.test_ml_metrics_collector_record_prediction ______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:373: in test_ml_metrics_collector_record_prediction
    collector = MLMetricsCollector()
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
____ TestPerformanceMetrics.test_ml_metrics_collector_record_model_training ____
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:412: in test_ml_metrics_collector_record_model_training
    collector = MLMetricsCollector()
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
______ TestPerformanceMetrics.test_ml_metrics_collector_system_resources _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:455: in test_ml_metrics_collector_system_resources
    collector = MLMetricsCollector()
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
______ TestPerformanceMetrics.test_metrics_manager_background_collection _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:492: in test_metrics_manager_background_collection
    manager = MetricsManager()
src/utils/metrics.py:560: in __init__
    self.collector = MLMetricsCollector(self.registry)
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
___________ TestPerformanceMetrics.test_multiprocess_metrics_manager ___________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:513: in test_multiprocess_metrics_manager
    mp_manager = MultiProcessMetricsManager()
src/utils/metrics.py:690: in __init__
    multiprocess.MultiProcessCollector(self.registry)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/multiprocess.py:30: in __init__
    raise ValueError('env PROMETHEUS_MULTIPROC_DIR is not set or not a directory')
E   ValueError: env PROMETHEUS_MULTIPROC_DIR is not set or not a directory
_______ TestHealthMonitoring.test_health_monitor_system_resources_check ________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:927: in test_health_monitor_system_resources_check
    monitor = HealthMonitor()
src/utils/health_monitor.py:179: in __init__
    self.metrics_collector = get_metrics_collector()
src/utils/metrics.py:621: in get_metrics_collector
    return get_metrics_manager().get_collector()
src/utils/metrics.py:615: in get_metrics_manager
    _metrics_manager = MetricsManager()
src/utils/metrics.py:560: in __init__
    self.collector = MLMetricsCollector(self.registry)
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
_______ TestHealthMonitoring.test_health_monitor_critical_resource_usage _______
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:955: in test_health_monitor_critical_resource_usage
    monitor = HealthMonitor()
src/utils/health_monitor.py:179: in __init__
    self.metrics_collector = get_metrics_collector()
src/utils/metrics.py:621: in get_metrics_collector
    return get_metrics_manager().get_collector()
src/utils/metrics.py:615: in get_metrics_manager
    _metrics_manager = MetricsManager()
src/utils/metrics.py:560: in __init__
    self.collector = MLMetricsCollector(self.registry)
src/utils/metrics.py:125: in __init__
    self._setup_metrics()
src/utils/metrics.py:132: in _setup_metrics
    self.system_info = Info(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/metrics.py:132: in __init__
    registry.register(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/prometheus_client/registry.py:43: in register
    raise ValueError(
E   ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
_ TestMonitoringIntegration.test_monitoring_integration_prediction_error_handling _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:1257: in test_monitoring_integration_prediction_error_handling
    raise ValueError("Test prediction error")
E   ValueError: Test prediction error

During handling of the above exception, another exception occurred:
tests/unit/utilities/test_system_utilities.py:1254: in test_monitoring_integration_prediction_error_handling
    async with integration.track_prediction_operation(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/contextlib.py:231: in __aexit__
    await self.gen.athrow(value)
src/utils/monitoring_integration.py:163: in track_prediction_operation
    await self.alert_manager.handle_prediction_error(
E   TypeError: object MagicMock can't be used in 'await' expression
_ TestMonitoringIntegration.test_monitoring_integration_concept_drift_recording _
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:1377: in test_monitoring_integration_concept_drift_recording
    integration.record_concept_drift(
src/utils/monitoring_integration.py:307: in record_concept_drift
    asyncio.create_task(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/tasks.py:417: in create_task
    loop = events.get_running_loop()
E   RuntimeError: no running event loop
_____________ TestAlertSystem.test_alert_manager_alert_resolution ______________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:1678: in test_alert_manager_alert_resolution
    assert resolved_alert.resolution_notes == "Issue resolved by system restart"
E   AttributeError: 'AlertEvent' object has no attribute 'resolution_notes'
________ TestIncidentResponse.test_recovery_action_creation_and_limits _________
[gw1] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/utilities/test_system_utilities.py:1739: in test_recovery_action_creation_and_limits
    assert action.can_attempt() is True
E   AssertionError: assert False is True
E    +  where False = <bound method RecoveryAction.can_attempt of RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0)>()
E    +    where <bound method RecoveryAction.can_attempt of RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0)> = RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0).can_attempt
____ TestMQTTIntegrationManager.test_integration_manager_system_status_loop ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:1131: in test_integration_manager_system_status_loop
    await manager._system_status_publishing_loop()
src/integration/mqtt_integration_manager.py:480: in _system_status_publishing_loop
    await self.publish_system_status(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:2284: in _execute_mock_call
    self.await_count += 1
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:772: in __setattr__
    return object.__setattr__(self, name, value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:337: in _set
    def _set(self, value, name=name, _the_name=_the_name):
E   Failed: Timeout (>30.0s) from pytest-timeout.
----------------------------- Captured stdout call -----------------------------
~~~~~~~~~~~~~~~~~~~~~ Stack of <unknown> (140656961910464) ~~~~~~~~~~~~~~~~~~~~~
  File "/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/execnet/gateway_base.py", line 411, in _perform_spawn
    reply.run()
  File "/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/execnet/gateway_base.py", line 341, in run
    self._result = func(*args, **kwargs)
  File "/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/execnet/gateway_base.py", line 1160, in _thread_receiver
    msg = Message.from_io(io)
  File "/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/execnet/gateway_base.py", line 567, in from_io
    header = io.read(9)  # type 1, channel 4, payload 4
  File "/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/execnet/gateway_base.py", line 534, in read
    data = self._read(numbytes - len(buf))
___ TestMQTTIntegrationPerformance.test_connection_status_uptime_calculation ___
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:1250: in test_connection_status_uptime_calculation
    mock_datetime.utcnow.return_value = base_time.replace(
E   ValueError: second must be in 0..59
___ TestMQTTIntegrationCompatibility.test_mqtt_client_callback_api_versions ____
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
tests/unit/integration_layer/test_mqtt_integration.py:1377: in test_mqtt_client_callback_api_versions
    mock_client = Mock(spec=mqtt_client.Client)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:1121: in __init__
    _safe_super(CallableMixin, self).__init__(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:460: in __init__
    self._mock_add_spec(spec, spec_set, _spec_as_instance, _eat_self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:511: in _mock_add_spec
    raise InvalidSpecError(f'Cannot spec a Mock object. [object={spec!r}]')
E   unittest.mock.InvalidSpecError: Cannot spec a Mock object. [object=<MagicMock name='Client' id='140653862784560'>]
_ TestMQTTIntegrationCompatibility.test_mqtt_publisher_graceful_shutdown_with_queued_messages _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected '_process_message_queue' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests/unit/integration_layer/test_mqtt_integration.py:1411: in test_mqtt_publisher_graceful_shutdown_with_queued_messages
    mock_process.assert_called_once()
E   AssertionError: Expected '_process_message_queue' to have been called once. Called 0 times.
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_fields.py:161
../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_fields.py:161
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_fields.py:161: UserWarning: Field "model_info" has conflict with protected namespace "model_".
  
  You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    warnings.warn(

../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:291
../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:291
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:291: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.8/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:341
../../../../../opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:341
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pydantic/_internal/_config.py:341: UserWarning: Valid config keys have changed in V2:
  * 'schema_extra' has been renamed to 'json_schema_extra'
    warnings.warn(message, UserWarning)

tests/unit/data_layer/test_data_models.py: 12 warnings
tests/unit/ingestion/test_ha_integration.py: 38 warnings
  <string>:8: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

tests/unit/data_layer/test_data_models.py: 12 warnings
tests/unit/ingestion/test_ha_integration.py: 38 warnings
  <string>:9: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_import_progress_properties
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2315: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    progress.start_time = datetime.utcnow() - timedelta(seconds=50)

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_initialize_components
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_cleanup_components
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_estimate_total_events
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_process_entities_batch
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_process_entities_batch_skip_completed
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_process_single_entity
tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_bulk_insert_events
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/_pytest/python.py:183: PytestUnhandledCoroutineWarning: async def functions are not natively supported and have been skipped.
  You need to install a suitable plugin for your async framework, for example:
    - anyio
    - pytest-asyncio
    - pytest-tornasync
    - pytest-trio
    - pytest-twisted
    warnings.warn(PytestUnhandledCoroutineWarning(msg.format(nodeid)))

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_load_resume_data
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2422: RuntimeWarning: coroutine 'BulkImporter._load_resume_data' was never awaited
    importer._load_resume_data()
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_load_resume_data_no_file
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2435: RuntimeWarning: coroutine 'BulkImporter._load_resume_data' was never awaited
    importer._load_resume_data()  # Should return early
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_save_resume_data
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2454: RuntimeWarning: coroutine 'BulkImporter._save_resume_data' was never awaited
    importer._save_resume_data()
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_update_progress
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2692: RuntimeWarning: coroutine 'BulkImporter._update_progress' was never awaited
    importer._update_progress()
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_generate_import_report
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/data_layer/test_data_models.py:2707: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(minutes=30)

tests/unit/ingestion/test_ha_integration.py::TestHomeAssistantClient::test_client_connect_auth_failure
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/ha_client.py:197: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    async with self.session.get(api_url) as response:
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/ingestion/test_ha_integration.py: 32 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:56: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    return (datetime.utcnow() - self.start_time).total_seconds()

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_optimize_import_performance
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/ingestion/test_ha_integration.py:1022: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_create_import_checkpoint
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:968: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp": datetime.utcnow().isoformat(),

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_create_import_checkpoint
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:980: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    checkpoint_file = f"checkpoint_{checkpoint_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_verify_import_integrity_success
tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_verify_import_integrity_with_issues
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:877: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "verification_timestamp": datetime.utcnow().isoformat(),

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_convert_ha_events_to_sensor_events
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:554: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    created_at=datetime.utcnow(),

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_save_resume_data_success
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:268: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp": datetime.utcnow().isoformat(),

tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_update_progress_with_callback
tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_update_progress_callback_sync
tests/unit/ingestion/test_ha_integration.py::TestBulkImporter::test_update_progress_callback_error
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:636: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.progress.last_update = datetime.utcnow()

tests/unit/ingestion/test_ha_integration.py::TestHomeAssistantIntegration::test_bulk_import_workflow
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/ingestion/bulk_importer.py:172: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    end_date = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTConnectionStatus::test_connection_status_with_values
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:109: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    now = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublishResult::test_publish_result_success
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:132: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    now = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublishResult::test_publish_result_failure
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:150: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    now = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py: 31 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:85: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    client_id or f"ha_ml_predictor_{int(datetime.utcnow().timestamp())}"

tests/unit/integration_layer/test_mqtt_integration.py: 21 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:298: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.last_publish_time = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py: 21 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:308: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    publish_time=datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py: 156 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:266: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "queued_at": datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py: 156 warnings
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:287: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    publish_time=datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_publish_failure
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:320: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    publish_time=datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_get_connection_status
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:393: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    past_time = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_get_publisher_stats
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:412: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    publisher.last_publish_time = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_start_disabled
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/selectors.py:351: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    def register(self, fileobj, events, data=None):
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_connect_to_broker_success
tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_connect_to_broker_retry
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:423: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    wait_start = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_process_message_queue
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:553: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "queued_at": datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_process_message_queue
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:560: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "queued_at": datetime.utcnow(),

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_callbacks
tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationErrorHandling::test_callback_error_handling
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:571: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.connection_status.last_connected = datetime.utcnow()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_callbacks
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/integration/mqtt_publisher.py:603: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.connection_status.last_disconnected = datetime.utcnow()

tests/unit/data_layer/test_database_operations.py::TestGlobalDatabaseFunctions::test_get_db_session_context_manager
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py:839: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    async with db_manager.get_session() as session:
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_queue_retraining_request
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/models/training_integration.py:291: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "requested_at": datetime.utcnow(),

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_can_retrain_room
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/ml_models/test_training_pipeline.py:2061: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    recent_time = datetime.utcnow() - timedelta(hours=6)  # 6 hours ago

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_can_retrain_room
tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_can_retrain_room
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/models/training_integration.py:330: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    return (datetime.utcnow() - last_training) >= cooldown_period

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_can_retrain_room
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/ml_models/test_training_pipeline.py:2066: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    old_time = datetime.utcnow() - timedelta(hours=24)  # 24 hours ago

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_cooldown_remaining
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/ml_models/test_training_pipeline.py:2097: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    recent_time = datetime.utcnow() - timedelta(hours=6)  # 6 hours ago

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_cooldown_remaining
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/models/training_integration.py:338: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    elapsed = (datetime.utcnow() - last_training).total_seconds() / 3600

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_training_queue_status
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/ml_models/test_training_pipeline.py:2210: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    now = datetime.utcnow()

tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_training_queue_status
tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_training_queue_status
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/models/training_integration.py:781: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    datetime.utcnow() - req["requested_at"]

tests/unit/utilities/test_system_utilities.py::TestHealthMonitoring::test_health_monitor_lifecycle
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/src/utils/health_monitor.py:610: DeprecationWarning: Callback API version 1 is deprecated, update to latest version
    client = mqtt.Client()

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_system_status_loop
  /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/unittest/mock.py:337: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    def _set(self, value, name=name, _the_name=_the_name):
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationPerformance::test_connection_status_uptime_calculation
  /home/runner/work/ha-ml-predictor/ha-ml-predictor/tests/unit/integration_layer/test_mqtt_integration.py:1245: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    base_time = datetime.utcnow()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- generated xml file: /home/runner/work/ha-ml-predictor/ha-ml-predictor/junit-unit.xml -

---------- coverage: platform linux, python 3.12.11-final-0 ----------
Name                                              Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------------
src/__init__.py                                       0      0   100%
src/adaptation/__init__.py                            7      0   100%
src/adaptation/drift_detector.py                    544    393    28%   125-131, 140-185, 189-196, 201-208, 222-226, 239, 318, 387-447, 456-518, 524-581, 590-622, 638-660, 675-728, 746-771, 781-825, 835-882, 888-926, 932-969, 973-1014, 1024-1033, 1039-1079, 1087-1112, 1120-1137, 1143-1175, 1220-1227, 1231-1241, 1256-1307, 1317-1338, 1347-1373, 1388-1440, 1453-1485, 1489-1495, 1514-1521
src/adaptation/monitoring_enhanced_tracking.py       95     60    37%   57-83, 89-126, 130-157, 161-176, 183-192, 200, 208, 216, 222, 228-238, 250-253, 270-275, 283
src/adaptation/optimizer.py                         567    438    23%   30-36, 99-100, 104-121, 127, 131-133, 137-143, 149-155, 169-180, 184, 230, 371-461, 477-479, 483-500, 531-575, 581-589, 595-605, 618-688, 693-707, 713-734, 750-866, 872-943, 959-1023, 1044-1086, 1100, 1114-1119, 1231-1268, 1274-1302, 1317-1361, 1367-1387, 1391-1432, 1439
src/adaptation/retrainer.py                         936    779    17%   127-130, 134, 138, 237-247, 296-380, 384-408, 412-414, 418-451, 455, 527-566, 572-589, 593-605, 626-748, 773-828, 842-935, 941, 953-989, 993-1059, 1065-1107, 1115-1146, 1150-1163, 1167-1206, 1210-1233, 1237-1280, 1284-1298, 1304-1352, 1358-1403, 1409-1418, 1429-1557, 1568-1581, 1587-1610, 1621-1634, 1640-1670, 1680-1723, 1732-1787, 1793-1827, 1833-1854, 1860-1879, 1885-1941, 1945-1963, 1967-1985, 1991-2009, 2013, 2019-2092, 2096-2151, 2157-2200, 2206-2220, 2230-2261, 2265-2288, 2294-2324, 2331
src/adaptation/tracker.py                           617    483    22%   109-144, 154, 162, 264, 269-284, 288-292, 296-299, 305-316, 320, 436-454, 458-475, 492-533, 550-585, 598-608, 623-646, 667-712, 718-720, 726-728, 732-766, 772-798, 802-828, 832-874, 883-960, 964-975, 979-1040, 1050-1093, 1103-1136, 1144-1153, 1159-1184, 1190-1227, 1231-1246, 1252-1411, 1415-1423, 1427-1463, 1467-1539, 1545-1556, 1564-1565
src/adaptation/tracking_manager.py                  835    694    17%   22-26, 59-64, 208-269, 273-358, 362-399, 403-447, 459-578, 600-627, 632-708, 712-723, 731-741, 747-761, 765-775, 781-787, 793-799, 814-850, 862-897, 903-926, 930-981, 987-995, 999-1021, 1025-1063, 1067-1089, 1093-1154, 1158-1174, 1180-1295, 1299-1339, 1347-1384, 1388-1435, 1440-1446, 1467-1498, 1504-1512, 1516-1524, 1537-1544, 1554-1561, 1565-1574, 1583-1599, 1603-1609, 1613-1632, 1636-1711, 1715-1764, 1781-1807, 1822-1858, 1889-1937, 1953-2023, 2034-2070, 2075-2082, 2086-2152, 2158-2170, 2176-2188, 2192-2244, 2249-2256, 2260-2302, 2309-2310
src/adaptation/validator.py                         866    728    16%   108-144, 148-156, 160-167, 171, 244-253, 279-281, 286-288, 293-298, 307, 311, 383-427, 434-447, 451-463, 488-570, 595-685, 711-782, 788, 794, 800, 815-843, 855-894, 915-950, 954-1009, 1021-1065, 1069-1105, 1114-1132, 1152-1182, 1192-1283, 1292, 1301-1316, 1337-1404, 1415-1504, 1510-1522, 1526-1554, 1570-1594, 1600-1637, 1648-1670, 1684-1711, 1723-1754, 1760-1874, 1878-1883, 1887-1897, 1901-1909, 1913-1930, 1934-1960, 1964-1997, 2001-2123, 2133-2243, 2254-2308, 2314-2321, 2329
src/core/__init__.py                                  0      0   100%
src/core/backup_manager.py                          351     72    79%   107-109, 119-123, 207, 223-228, 235-242, 383, 390, 431-438, 478, 498-502, 565-571, 575-581, 598-635, 653-654, 661-669
src/core/config.py                                  319     14    96%   480, 561-564, 571-583
src/core/config_validator.py                        426    426     0%   6-746
src/core/constants.py                                48      0   100%
src/core/environment.py                             291    168    42%   35-45, 95-99, 103-104, 108-110, 116-132, 143-158, 162-168, 172-220, 346, 350, 353-361, 375, 382, 402, 407, 412, 428-430, 435-440, 447-448, 453-455, 462, 469-473, 477-498, 502-523, 527-546, 550-556, 560-598, 606-627
src/core/exceptions.py                              416      3    99%   117, 633, 643
src/data/__init__.py                                  0      0   100%
src/data/ingestion/__init__.py                        0      0   100%
src/data/ingestion/bulk_importer.py                 419     89    79%   76, 180, 206-209, 242, 256-257, 280-281, 308-311, 341-361, 371-372, 397-404, 425-428, 435, 450-453, 479-508, 528-530, 568-569, 578-581, 615-616, 793-800, 833-834, 842-843, 854-858, 937, 947-951, 987-989, 1006
src/data/ingestion/event_processor.py               558    178    68%   179-182, 234, 290, 321, 358-359, 365-373, 388, 396-400, 424, 454-456, 465, 489, 558, 562, 567, 612-635, 651-665, 685-712, 719-740, 759, 843-845, 906-907, 912-919, 945-971, 983-995, 1019-1081, 1122, 1136-1137, 1154-1159, 1165, 1169-1185, 1190-1191, 1215-1227, 1250, 1261
src/data/ingestion/ha_client.py                     377    132    65%   133-134, 138, 143, 181-182, 199, 203, 207, 230-231, 240, 259-275, 283-294, 301, 309, 313, 332, 348-349, 362, 385, 387, 389, 394-399, 404, 409-411, 422-424, 428-453, 463, 483-485, 497-498, 504-508, 526, 542-543, 579, 582-583, 590, 609-614, 635-676, 694-696, 734, 738-742
src/data/storage/__init__.py                          3      0   100%
src/data/storage/database.py                        389    262    33%   81-82, 90-92, 101-102, 112, 114, 133-135, 150, 155-173, 178, 183, 188-192, 203, 216, 220-232, 245-321, 347-388, 413-455, 478-531, 535-560, 569-627, 636-762, 766-775, 779, 784-798, 802, 807, 840
src/data/storage/database_compatibility.py           75     46    39%   28, 44-81, 100-106, 115-117, 123-139, 148-161, 180-182
src/data/storage/dialect_utils.py                   136     45    67%   59-82, 91, 98-109, 117-119, 126-135, 144-150, 182-191, 210-232, 308, 342, 351
src/data/storage/models.py                          447      5    99%   889-891, 1010, 1179, 1187
src/data/validation/__init__.py                       4      4     0%   12-37
src/data/validation/event_validator.py              300    300     0%   12-922
src/data/validation/pattern_detector.py             336    336     0%   12-851
src/data/validation/schema_validator.py             332    332     0%   14-1054
src/features/__init__.py                              0      0   100%
src/features/contextual.py                          449    383    15%   97-134, 148-311, 317-403, 412-514, 518-557, 563-610, 619-690, 696-721, 728, 758-762, 768-809, 815-882, 888-920, 924-932, 992, 1010-1056, 1068-1099
src/features/engineering.py                         303    178    41%   130-131, 135, 140, 157, 226-276, 289-369, 393-394, 399-406, 413-424, 519-533, 538-550, 554, 583-622, 629-655, 668-714, 724, 744-791, 799-805, 809-815, 819-826, 830-837
src/features/sequential.py                          350    219    37%   88-133, 209-210, 236-240, 249, 290-326, 344, 400-536, 542-722, 726-775, 781-824, 830-833, 894
src/features/store.py                               220    142    35%   54-78, 128-142, 165-181, 185-187, 191-194, 260-262, 287-353, 374-405, 430-472, 482-499, 505-551, 563, 577, 583-584, 588-591, 599-600, 604-612, 616-649, 653
src/features/temporal.py                            323     77    76%   78-79, 100, 221-222, 271, 300-301, 304-306, 318-321, 435, 510-511, 522-524, 532-538, 551, 582-584, 588-590, 602-613, 625-670, 755-792, 813-820
src/integration/__init__.py                          34     18    47%   85-92, 215-229, 248-258, 263
src/integration/api_server.py                       675    477    29%   83-92, 98-100, 130-132, 138-140, 146-148, 154-159, 176-183, 189-192, 198-200, 224-227, 242-261, 277-297, 303, 308-309, 316-327, 332-341, 350-369, 374-436, 445, 454-506, 531-573, 584-604, 663-674, 687-704, 718-730, 749-785, 815-824, 836, 847-1020, 1030-1069, 1093-1115, 1125-1133, 1143-1155, 1165-1186, 1196-1216, 1226-1241, 1251-1264, 1274-1294, 1304-1312, 1322-1344, 1356-1376, 1386-1399, 1409-1422, 1436-1475, 1487-1504, 1519-1544, 1559-1587, 1599-1613, 1625-1659, 1677-1680, 1684-1702, 1706-1711, 1715, 1730-1733
src/integration/auth/__init__.py                      6      0   100%
src/integration/auth/auth_models.py                 156     58    63%   33-48, 54-60, 64, 68, 72, 95-99, 105-120, 161-169, 218-221, 227-242, 259-264, 270-274, 296-298, 302
src/integration/auth/dependencies.py                 93     71    24%   32-41, 65-102, 125-128, 151-164, 187-198, 218-224, 250-280, 298-317, 330-345
src/integration/auth/endpoints.py                   148    112    24%   84, 89, 94, 109-192, 207-227, 244-263, 276, 288-321, 337-366, 382-397, 409-455, 468-496
src/integration/auth/exceptions.py                   65     43    34%   22-26, 44-50, 67-71, 83-87, 99-103, 115-117, 131-139, 152, 165-171, 188-194
src/integration/auth/jwt_manager.py                 181    151    17%   28-31, 35-38, 42-50, 54-60, 81-95, 117-150, 162-183, 199-245, 261-279, 291-308, 320-340, 353-367, 385-416, 420-425, 429-436, 440-448, 460-480
src/integration/auth/middleware.py                  146    120    18%   34-35, 39-92, 105-145, 155-241, 264-322, 327-348, 353-361, 366-376, 388-410, 422-423, 427-476
src/integration/dashboard.py                        668    558    16%   34-46, 156, 188-193, 212-236, 240-250, 265-285, 297-320, 324-330, 368-398, 406-426, 432-617, 621-658, 662-694, 698-734, 738-820, 829-927, 936-999, 1009-1074, 1083-1181, 1193-1275, 1287-1323, 1333-1372, 1376-1396, 1404-1420, 1429-1462, 1466-1482, 1486-1511, 1515-1556, 1560-1570, 1574-1582, 1594, 1615-1627, 1641-1647
src/integration/discovery_publisher.py              392    263    33%   202-262, 271-326, 341-379, 388-431, 443-469, 484-487, 491, 536-584, 599-661, 680-709, 723-747, 760-777, 791-847, 859, 876, 889, 903, 915, 928, 942, 957, 970, 983, 996, 1008, 1024-1159, 1172
src/integration/enhanced_integration_manager.py     383    383     0%   17-922
src/integration/enhanced_mqtt_manager.py            224    180    20%   85-129, 136-157, 163-181, 203-235, 258-305, 309, 315, 319, 323, 327-366, 370-393, 399-401, 405-406, 414-421, 427-439, 445-482, 486-529, 535-551, 555-576, 580-611, 618
src/integration/ha_entity_definitions.py            531    531     0%   18-1718
src/integration/ha_tracking_bridge.py               249    249     0%   16-605
src/integration/monitoring_api.py                   131    131     0%   6-285
src/integration/mqtt_integration_manager.py         310    112    64%   149-153, 173-174, 188-190, 196, 214-215, 237-267, 292, 308-317, 340-349, 496-497, 500-501, 503, 518-522, 538-543, 561-585, 625, 641-642, 648-650, 662-676, 687-703, 719-733
src/integration/mqtt_publisher.py                   318     97    69%   169-170, 187-189, 202-208, 214, 225-226, 324-339, 371, 428-432, 457-463, 467-492, 496-519, 529, 548-555, 561-562, 578, 590-597, 608, 615, 624-628, 640-641, 651, 656-660, 669, 671, 673, 677-678
src/integration/prediction_publisher.py             163    106    35%   116-131, 150-243, 272-362, 385-397, 401, 420-457, 461-479, 483-488, 498-503
src/integration/realtime_api_endpoints.py           292    230    21%   50, 55-60, 97-118, 128-231, 241-257, 270-294, 308-354, 367-405, 418-431, 444-486, 499-531, 550-613, 633-648, 652-656, 660-676, 680-695, 699, 703, 707, 711-722, 739-777
src/integration/realtime_publisher.py               482    387    20%   65, 85-86, 101, 113, 124-126, 132-147, 151-157, 161-164, 168-171, 177-199, 203-224, 228, 258-260, 264-281, 285-291, 295-298, 304-326, 330-351, 355, 406-438, 445-462, 466-482, 503-608, 622-663, 669-707, 714-747, 760-764, 768-770, 775-780, 802-810, 839-901, 905-923, 927-962, 966-983, 987-999, 1006, 1023-1031, 1037-1090, 1099-1114
src/integration/tracking_integration.py             186    145    22%   76-90, 94-132, 136-153, 157-159, 163-165, 169-202, 206-207, 211-212, 218-232, 236-250, 254-286, 290-350, 354-375, 396-411, 434-457, 464
src/integration/websocket_api.py                    628    628     0%   21-1440
src/main_system.py                                   78     78     0%   9-157
src/models/__init__.py                                0      0   100%
src/models/base/__init__.py                           6      0   100%
src/models/base/gp_predictor.py                     411    262    36%   34-42, 133-229, 282-284, 305, 318, 336-351, 449, 452, 473-536, 567-589, 605-624, 637-663, 678-713, 727-752, 773-789, 801-821, 825-834, 840-848, 855, 866, 869, 893-1041, 1076-1080, 1096-1127, 1131-1175
src/models/base/hmm_predictor.py                    372    206    45%   51, 60-61, 68-69, 209-215, 235, 301-378, 392-440, 447-455, 458-461, 472-527, 543-550, 588-612, 638-646, 655-668, 678-692, 750-752, 794-795, 800-802, 824-943, 949-972, 982-1003
src/models/base/lstm_predictor.py                   306    126    59%   55, 167-172, 215-227, 251, 322, 336, 373, 378-382, 409-412, 431, 465-467, 484, 501-511, 516, 528, 535, 544, 554, 561, 589-595, 610, 614-615, 619-639, 675-676, 692-723, 745-859
src/models/base/predictor.py                        160      8    95%   62, 64, 66, 163, 186, 196, 401-402
src/models/base/xgboost_predictor.py                230    131    43%   104, 117-122, 162-240, 283, 309-344, 363, 372-373, 387-397, 421-455, 477-481, 487-520, 535-559, 568-571, 576, 625-626, 642-672
src/models/ensemble.py                              562    456    19%   105, 136-238, 280-341, 353, 361-369, 373, 400-537, 545-625, 634-692, 702-732, 736-764, 775-881, 887-901, 906-936, 957-1072, 1081-1143, 1147-1159, 1171, 1173, 1177, 1184, 1196-1197, 1203-1204, 1207-1208, 1224-1255, 1262, 1295-1324, 1340-1391
src/models/training_config.py                       263    131    50%   53-54, 56-57, 60, 126-143, 212-225, 250, 387-411, 418-442, 456-503, 507-515, 519-523, 527, 533-540, 546-553, 559-560, 564-576, 580-628, 632-653, 657-670, 691-692
src/models/training_integration.py                  369    238    36%   89-91, 106-107, 111-128, 132-152, 193-198, 215-216, 233-238, 244-258, 272-275, 279-280, 303-304, 344-367, 371-420, 424-476, 480-487, 493-522, 526-549, 555-586, 590-612, 616-639, 643-657, 661-670, 674-681, 685, 689-704, 708-715, 739-741, 766-768, 788-790, 827-841
src/models/training_pipeline.py                     725    411    43%   269, 316-317, 341-342, 350, 365-367, 396-398, 421-449, 516-630, 643-644, 686-688, 694-723, 743-744, 750, 762, 782-784, 827, 830, 835, 838, 847-850, 885, 929-931, 944-977, 999-1034, 1051-1080, 1097-1131, 1148-1174, 1189-1266, 1276-1336, 1345-1406, 1428-1486, 1490-1491, 1501-1583, 1589-1602, 1610-1611, 1616-1632, 1636-1647, 1682-1683, 1689, 1693, 1701, 1705, 1717-1747, 1769-1771, 1780-1819, 1823-1829, 1835-1876, 1880-1885
src/utils/__init__.py                                 0      0   100%
src/utils/alerts.py                                 252     55    78%   107-109, 145-147, 209-215, 228, 257-264, 277-305, 311, 358-372, 477-478, 482-484, 497-498, 507-508, 550, 583-584, 602-620, 633-639, 650, 675-677
src/utils/health_monitor.py                         533    243    54%   132, 142-143, 224-225, 229, 234-235, 258, 267-268, 285-307, 311-333, 347-351, 369, 389-415, 418-454, 458-462, 479-480, 482-484, 487-488, 490-492, 496-497, 499-501, 532-533, 552-570, 582, 603-606, 618-627, 635-636, 656-657, 691-700, 705, 720-728, 732-733, 765-766, 789-790, 792-793, 817-818, 837-838, 840-841, 864-865, 900-901, 906-928, 947, 970-974, 980-982, 988-991, 999-1004, 1027-1028, 1040-1111, 1130-1136, 1141-1176, 1189-1239, 1243, 1272, 1278-1281, 1287-1291, 1299, 1303-1305, 1330-1332
src/utils/incident_response.py                      392    179    54%   71, 88-89, 175-176, 196, 200, 350-351, 368, 377-378, 401-403, 407-437, 441-520, 524-624, 633-649, 654-662, 670-677, 689-712, 727-737, 740, 784-785, 791-803, 807-821, 825-847, 851-867, 873, 877, 881-882, 916-926, 930-945, 955-957
src/utils/logger.py                                 124     23    81%   140, 154, 189, 205-211, 217-222, 228-233, 266, 286, 302, 337, 352-364, 380, 399-409, 456
src/utils/metrics.py                                307    194    37%   29-117, 353-367, 381-391, 399-411, 423, 431, 439, 447, 453, 459, 465-469, 475-496, 500, 506, 510, 514-515, 521-541, 547-552, 561-563, 567-586, 590-592, 596-600, 604, 616, 626, 655-664, 694, 698, 707-738, 747-754, 758-766, 776-778, 787-796, 809-825, 838-845
src/utils/monitoring.py                             273     88    68%   156, 202-203, 214-218, 233-241, 250, 260, 311, 334-335, 337-338, 358-359, 376-377, 379-380, 398-399, 419-420, 422-423, 446-447, 464-465, 467-468, 487-488, 514-516, 529, 540, 551, 553, 566-567, 591-597, 603-614, 620-653, 693-695
src/utils/monitoring_integration.py                 111     45    59%   42-59, 63-70, 74-85, 170-183, 229-244, 326-331, 342-347, 361, 367, 373-379, 391-418, 428-430
src/utils/time_utils.py                             233     52    78%   48-52, 71-76, 90, 97-98, 103-105, 120, 126-132, 173-174, 221, 241, 245, 247, 266, 270, 272, 294, 329-336, 412-416, 434, 467-470, 518-521, 545, 550, 555, 560
-------------------------------------------------------------------------------
TOTAL                                             23507  14962    36%
Coverage HTML written to dir htmlcov-unit
Coverage XML written to file coverage-unit.xml

- Generated html report: file:///home/runner/work/ha-ml-predictor/ha-ml-predictor/report-unit.html -
=========================== short test summary info ============================
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_get_enhanced_tracking_manager_convenience_function - TypeError: get_enhanced_tracking_manager() missing 1 required positional argument: 'config'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_accuracy_metrics_creation - TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/core_system/test_backup_management.py::TestModelBackupManager::test_create_backup_tar_failure - FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-0/popen-gw1/test_create_backup_tar_failure0/backups/models/models_20250824_081822.tar.gz'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_accuracy_metrics_confidence_calibration - TypeError: AccuracyMetrics.__init__() got an unexpected keyword argument 'room_id'
FAILED tests/unit/core_system/test_backup_management.py::TestConfigurationBackupManager::test_create_backup_auto_generated_id - AssertionError: assert 22 == 21
 +  where 22 = len('config_20250824_081822')
 +    where 'config_20250824_081822' = BackupMetadata(backup_id='config_20250824_081822', backup_type='config', timestamp=datetime.datetime(2025, 8, 24, 8, 18, 22, 925449), size_bytes=256000, compressed=True, checksum=None, retention_date=None, tags={'config_dir': '/tmp/pytest-of-runner/pytest-0/popen-gw1/test_create_backup_auto_genera2/config'}).backup_id
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_prediction_validator_initialization - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'validation_window_hours'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_prediction_validator_record_prediction - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_prediction_validator_validate_prediction - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_prediction_validator_get_accuracy_metrics - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_prediction_validator_expire_old_predictions - TypeError: PredictionValidator.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_run_scheduled_backups_model_backup_interval - TypeError: Need a valid target to patch. You supplied: 'datetime'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_error_creation - TypeError: ValidationError.__init__() takes 2 positional arguments but 3 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_statistical_test_enum - AttributeError: type object 'StatisticalTest' has no attribute 'KS_TEST'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_drift_metrics_creation - TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_drift_metrics_severity_determination - TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_run_scheduled_backups_config_backup_daily - TypeError: Need a valid target to patch. You supplied: 'datetime'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_drift_metrics_recommendations - TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_drift_metrics_to_dict - TypeError: DriftMetrics.__init__() got an unexpected keyword argument 'drift_score'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_feature_drift_result - AttributeError: type object 'StatisticalTest' has no attribute 'KS_TEST'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_concept_drift_detector_initialization - TypeError: ConceptDriftDetector.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_cleanup_expired_backups_metadata_discovery - AttributeError: 'PosixPath' object attribute 'rglob' is read-only
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_concept_drift_detector_detect_drift - TypeError: ConceptDriftDetector.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_concept_drift_detector_calculate_psi - assert 0.05 <= 0.0
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_concept_drift_detector_page_hinkley_test - TypeError: ConceptDriftDetector._run_page_hinkley_test() missing 1 required positional argument: 'room_id'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_list_backups_all_types - AttributeError: 'PosixPath' object attribute 'rglob' is read-only
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_feature_drift_detector_initialization - TypeError: FeatureDriftDetector.__init__() got an unexpected keyword argument 'monitoring_interval_minutes'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_feature_drift_detector_numerical_drift_test - AttributeError: 'coroutine' object has no attribute 'feature_name'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_feature_drift_detector_categorical_drift_test - AttributeError: 'coroutine' object has no attribute 'feature_name'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestDriftDetection::test_drift_detection_error - AttributeError: 'ErrorSeverity' object has no attribute 'items'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_list_backups_type_filtering - AttributeError: 'PosixPath' object attribute 'rglob' is read-only
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_request_creation - TypeError: RetrainingRequest.__init__() got an unexpected keyword argument 'accuracy_threshold'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_request_priority_comparison - TypeError: RetrainingRequest.__init__() missing 2 required positional arguments: 'strategy' and 'created_time'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_request_to_dict - TypeError: RetrainingRequest.__init__() missing 2 required positional arguments: 'priority' and 'created_time'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_progress_creation - TypeError: RetrainingProgress.__init__() got an unexpected keyword argument 'status'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_get_backup_info_specific_lookup - AttributeError: 'PosixPath' object attribute 'rglob' is read-only
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_progress_update - TypeError: RetrainingProgress.__init__() got an unexpected keyword argument 'status'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_history_creation - AttributeError: 'RetrainingHistory' object has no attribute 'retraining_records'. Did you mean: 'add_retraining_record'?
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_history_add_record - TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
FAILED tests/unit/core_system/test_backup_management.py::TestBackupManagerOrchestration::test_get_backup_info_not_found - AttributeError: 'PosixPath' object attribute 'rglob' is read-only
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_history_success_rate - TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_history_recent_performance - TypeError: RetrainingHistory.add_retraining_record() got an unexpected keyword argument 'request_id'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_initialization - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_initialization_async - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_evaluate_retraining_need - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'accuracy_threshold_minutes'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_request_retraining - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_get_retrainer_stats - TypeError: AdaptiveRetrainer.__init__() missing 1 required positional argument: 'tracking_config'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_adaptive_retrainer_cancel_retraining - TypeError: AdaptiveRetrainer.__init__() got an unexpected keyword argument 'adaptive_retraining_enabled'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestAdaptiveRetraining::test_retraining_error - TypeError: RetrainingError.__init__() takes 2 positional arguments but 3 were given
FAILED tests/unit/core_system/test_configuration_system.py::TestConfigLoader::test_config_loader_load_yaml_missing_file - FileNotFoundError: Configuration directory not found: config
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_creation - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_parameter_names - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/core_system/test_configuration_system.py::TestConfigLoader::test_config_loader_load_config_with_environment - AssertionError: Calls not found.
Expected: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
 call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8')]
  Actual: [call(PosixPath('config/config.production.yaml'), 'r', encoding='utf-8'),
 call().__enter__(),
 call().__exit__(None, None, None),
 call(PosixPath('config/rooms.yaml'), 'r', encoding='utf-8'),
 call().__enter__(),
 call().__exit__(None, None, None)]
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_continuous_identification - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_bounds_and_choices - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/core_system/test_configuration_system.py::TestGlobalConfiguration::test_get_config_singleton_behavior - AssertionError: assert <Mock name='ConfigLoader()._create_system_config()' id='140554928732688'> is <Mock name='ConfigLoader().load_config()' spec='SystemConfig' id='140554928729088'>
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_sampling - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_hyperparameter_space_to_dict - TypeError: HyperparameterSpace.__init__() takes 1 positional argument but 2 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_optimization_result_creation - TypeError: OptimizationResult.__init__() got an unexpected keyword argument 'n_evaluations'
FAILED tests/unit/core_system/test_configuration_system.py::TestGlobalConfiguration::test_get_config_environment_manager_integration - AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_optimization_config_creation - TypeError: OptimizationConfig.__init__() got an unexpected keyword argument 'timeout_minutes'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_initialization - AssertionError: assert <ModelType.LSTM: 'lstm'> in {'gaussian_process': [{'categories': ['rb', 'matern', 'rational_quadratic'], 'name': 'kernel', 'type': 'categorical'}, {'high': 0.1, 'low': 1e-12, 'name': 'alpha', 'type': 'continuous'}, {'high': 10, 'low': 0, 'name': 'n_restarts_optimizer', 'type': 'integer'}], 'hmm': [{'high': 8, 'low': 2, 'name': 'n_states', 'type': 'integer'}, {'categories': ['spherical', 'diag', 'full'], 'name': 'covariance_type', 'type': 'categorical'}, {'high': 200, 'low': 50, 'name': 'n_iter', 'type': 'integer'}, {'high': 0.01, 'low': 1e-06, 'name': 'tol', 'type': 'continuous'}], 'lstm': [{'high': 256, 'low': 32, 'name': 'hidden_size', 'type': 'integer'}, {'high': 4, 'low': 1, 'name': 'num_layers', 'type': 'integer'}, {'high': 0.5, 'low': 0.0, 'name': 'dropout', 'type': 'continuous'}, {'high': 0.01, 'low': 0.0001, 'name': 'learning_rate', 'type': 'continuous'}, {'categories': [16, 32, 64, 128], 'name': 'batch_size', 'type': 'categorical'}], 'xgboost': [{'high': 500, 'low': 50, 'name': 'n_estimators', 'type': 'integer'}, {'high': 10, 'low': 3, 'name': 'max_depth', 'type': 'integer'}, {'high': 0.3, 'low': 0.01, 'name': 'learning_rate', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'subsample', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'colsample_bytree', 'type': 'continuous'}, {'high': 1.0, 'low': 0.0, 'name': 'reg_alpha', 'type': 'continuous'}, ...]}
 +  where <ModelType.LSTM: 'lstm'> = ModelType.LSTM
 +  and   {'gaussian_process': [{'categories': ['rb', 'matern', 'rational_quadratic'], 'name': 'kernel', 'type': 'categorical'}, {'high': 0.1, 'low': 1e-12, 'name': 'alpha', 'type': 'continuous'}, {'high': 10, 'low': 0, 'name': 'n_restarts_optimizer', 'type': 'integer'}], 'hmm': [{'high': 8, 'low': 2, 'name': 'n_states', 'type': 'integer'}, {'categories': ['spherical', 'diag', 'full'], 'name': 'covariance_type', 'type': 'categorical'}, {'high': 200, 'low': 50, 'name': 'n_iter', 'type': 'integer'}, {'high': 0.01, 'low': 1e-06, 'name': 'tol', 'type': 'continuous'}], 'lstm': [{'high': 256, 'low': 32, 'name': 'hidden_size', 'type': 'integer'}, {'high': 4, 'low': 1, 'name': 'num_layers', 'type': 'integer'}, {'high': 0.5, 'low': 0.0, 'name': 'dropout', 'type': 'continuous'}, {'high': 0.01, 'low': 0.0001, 'name': 'learning_rate', 'type': 'continuous'}, {'categories': [16, 32, 64, 128], 'name': 'batch_size', 'type': 'categorical'}], 'xgboost': [{'high': 500, 'low': 50, 'name': 'n_estimators', 'type': 'integer'}, {'high': 10, 'low': 3, 'name': 'max_depth', 'type': 'integer'}, {'high': 0.3, 'low': 0.01, 'name': 'learning_rate', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'subsample', 'type': 'continuous'}, {'high': 1.0, 'low': 0.6, 'name': 'colsample_bytree', 'type': 'continuous'}, {'high': 1.0, 'low': 0.0, 'name': 'reg_alpha', 'type': 'continuous'}, ...]} = <src.adaptation.optimizer.ModelOptimizer object at 0x7fed019157f0>._parameter_spaces
FAILED tests/unit/core_system/test_configuration_system.py::TestGlobalConfiguration::test_get_config_import_error_fallback - AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_optimize_model_parameters - AttributeError: Mock object has no attribute 'get_parameters'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_cached_parameters - TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_get_optimization_stats - TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
FAILED tests/unit/core_system/test_configuration_system.py::TestGlobalConfiguration::test_reload_config_forced_reload - AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_parameter_space_initialization - TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_model_optimizer_should_optimize_logic - TypeError: ModelOptimizer.__init__() missing 1 required positional argument: 'config'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestModelOptimization::test_optimization_error - TypeError: OptimizationError.__init__() takes 2 positional arguments but 3 were given
FAILED tests/unit/core_system/test_configuration_system.py::TestGlobalConfiguration::test_reload_config_environment_manager_path - AttributeError: <module 'src.core.config' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/core/config.py'> does not have the attribute 'get_environment_manager'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_real_time_metrics_creation - TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
FAILED tests/unit/core_system/test_configuration_system.py::TestConfigurationEdgeCases::test_config_loader_with_unicode_characters - FileNotFoundError: [Errno 2] No such file or directory: 'config/unicode.yaml'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_real_time_metrics_overall_health_score - TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_real_time_metrics_is_healthy - TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'accuracy_1h'
FAILED tests/unit/core_system/test_configuration_system.py::TestConfigurationEdgeCases::test_environment_variable_edge_cases - ValueError: JWT is enabled but JWT_SECRET_KEY environment variable is not set
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_real_time_metrics_to_dict - TypeError: RealTimeMetrics.__init__() got an unexpected keyword argument 'predictions_1h'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_creation - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_age_calculation - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_event_is_valid - TypeError: HAEvent.__init__() missing 2 required positional arguments: 'previous_state' and 'attributes'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_requires_escalation - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_acknowledge - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/data_layer/test_data_models.py::TestHAClient::test_rate_limiter_acquire_rate_limited - TypeError: '>=' not supported between instances of 'float' and 'datetime.datetime'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_resolve - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_escalate - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_alert_to_dict - TypeError: AccuracyAlert.__init__() got an unexpected keyword argument 'condition_type'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_initialization - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'accuracy_thresholds'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_start_stop_monitoring - TypeError: AccuracyTracker.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_get_real_time_metrics - AttributeError: Mock object has no attribute 'extract_recent_validation_records'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_get_active_alerts - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_acknowledge_alert - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_get_accuracy_trends - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_export_tracking_data - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_add_remove_notification_callback - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracker_get_tracker_stats - TypeError: AccuracyTracker.__init__() missing 1 required positional argument: 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestPerformanceTracking::test_accuracy_tracking_error - TypeError: AccuracyTrackingError.__init__() takes 2 positional arguments but 3 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_config_creation - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'prediction_validation_enabled'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_initialization - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_initialize - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_start_stop_tracking - TypeError: TrackingConfig.__init__() got an unexpected keyword argument 'enable_background_tasks'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_record_prediction - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_handle_room_state_change - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_get_tracking_status - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_get_real_time_metrics - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_get_active_alerts - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_acknowledge_alert - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'accuracy_tracker'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_check_drift - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'drift_detector'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_request_manual_retraining - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_get_retraining_status - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_cancel_retraining - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'adaptive_retrainer'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_get_system_stats - TypeError: TrackingManager.__init__() got an unexpected keyword argument 'prediction_validator'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestTrackingManagement::test_tracking_manager_error - TypeError: TrackingManagerError.__init__() takes 2 positional arguments but 3 were given
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_initialization - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_import_progress_properties - TypeError: 'float' object is not callable
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_method_wrapping - AssertionError: assert False
 +  where False = hasattr(<src.adaptation.monitoring_enhanced_tracking.MonitoringEnhancedTrackingManager object at 0x7fed019e72f0>, '_original_record_prediction')
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_import_progress_properties_edge_cases - TypeError: 'float' object is not callable
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_initialization - AttributeError: type object 'int' has no attribute '__origin__'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_monitored_record_prediction - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_load_resume_data - AssertionError: assert {} == {'completed_e...ntities': 50}}
  Right contains 2 more items:
  {'completed_entities': ['sensor.test1', 'sensor.test2'],
   'progress': {'processed_entities': 50}}
  Full diff:
    {
  +  ,
  -  'completed_entities': ['sensor.test1',
  -                         'sensor.test2'],
  -  'progress': {'processed_entities': 50},
    }
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_monitored_validate_prediction - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_save_resume_data - AssertionError: assert False
 +  where False = <MagicMock name='dump' id='140554916339328'>.called
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_determine_sensor_type - AssertionError: assert 'presence' == 'motion'
  - motion
  + presence
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_monitored_start_tracking - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_update_progress - AssertionError: Expected 'mock' to be called once. Called 0 times.
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_monitored_stop_tracking - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestBulkImporter::test_bulk_importer_generate_import_report - TypeError: argument of type 'coroutine' is not iterable
FAILED tests/unit/data_layer/test_data_models.py::TestPatternDetection::test_statistical_pattern_analysis - assert 600.000004 == 600.0
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_record_concept_drift - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/data_layer/test_data_models.py::TestPatternDetection::test_anomaly_detection_logic - assert 0 == 1
 +  where 0 = len([])
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_record_feature_computation - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_record_database_operation - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_record_mqtt_publish - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_update_connection_status - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_get_monitoring_status - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_track_model_training - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_monitoring_enhanced_tracking_manager_getattr_delegation - AttributeError: Mock object has no attribute 'validate_prediction'
FAILED tests/unit/adaptation/test_model_adaptation.py::TestEnhancedMonitoring::test_create_monitoring_enhanced_tracking_manager_factory - TypeError: create_monitoring_enhanced_tracking_manager() missing 1 required positional argument: 'config'
FAILED tests/unit/data_layer/test_data_models.py::TestPredictionModel::test_prediction_analyze_confidence_spread - assert 0.7000000000000001 == 0.7
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestTemporalFeatures::test_extract_features_error_handling - AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestSequentialFeatures::test_extract_velocity_features_with_numpy_operations - AssertionError: assert 'movement_velocity' in {'avg_event_interval': 300.0, 'burst_ratio': 300.0, 'event_interval_variance': 0.0, 'interval_autocorr': 0.0, ...}
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestSequentialFeatures::test_get_default_features_returns_complete_dict - AssertionError: assert 'movement_velocity' in {'active_room_count': 1.0, 'avg_event_interval': 300.0, 'avg_room_dwell_time': 1800.0, 'burst_ratio': 0.0, ...}
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestSequentialFeatures::test_extract_features_error_handling - AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestContextualFeatures::test_extract_environmental_features_with_climate_sensors - TypeError: ContextualFeatureExtractor._extract_environmental_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestContextualFeatures::test_extract_door_state_features_with_door_sensors - TypeError: ContextualFeatureExtractor._extract_door_state_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestContextualFeatures::test_extract_multi_room_features_with_correlation - TypeError: ContextualFeatureExtractor._extract_multi_room_features() missing 1 required positional argument: 'target_time'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestContextualFeatures::test_extract_features_error_handling - AttributeError: 'FeatureExtractionError' object has no attribute 'feature_type'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_extract_features_validation_empty_room_id - AssertionError: assert 'room_id cannot be empty' in 'Feature extraction failed: general for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=general, room_id=unknown | Caused by: FeatureExtractionError: Feature extraction failed: validation for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=validation, room_id=unknown | Caused by: ValueError: Room ID is required'
 +  where 'Feature extraction failed: general for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=general, room_id=unknown | Caused by: FeatureExtractionError: Feature extraction failed: validation for room unknown | Error Code: FEATURE_EXTRACTION_ERROR | Context: feature_type=validation, room_id=unknown | Caused by: ValueError: Room ID is required' = str(FeatureExtractionError('Feature extraction failed: general for room unknown'))
 +    where FeatureExtractionError('Feature extraction failed: general for room unknown') = <ExceptionInfo FeatureExtractionError('Feature extraction failed: general for room unknown') tblen=2>.value
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_extract_features_sequential_processing - AssertionError: assert 'sequential_seq_feature' in {'meta_data_quality_score': 0.0, 'meta_event_count': 0.0, 'meta_extraction_day_of_week': 0.0, 'meta_extraction_hour': 12.0, ...}
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_validate_configuration_with_none_config - assert <coroutine object FeatureEngineeringEngine.validate_configuration at 0x7fed024bae30> is True
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_validate_configuration_with_invalid_max_workers - AssertionError: assert 'max_workers must be >= 1' in 'max_workers must be at least 1, got 0 | Error Code: FEATURE_ENGINE_INVALID_WORKERS'
 +  where 'max_workers must be at least 1, got 0 | Error Code: FEATURE_ENGINE_INVALID_WORKERS' = str(ConfigurationError('max_workers must be at least 1, got 0'))
 +    where ConfigurationError('max_workers must be at least 1, got 0') = <ExceptionInfo ConfigurationError('max_workers must be at least 1, got 0') tblen=3>.value
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_compute_feature_correlations_with_dataframe - AttributeError: 'list' object has no attribute 'empty'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureEngineering::test_destructor_shuts_down_executor - AssertionError: expected call not found.
Expected: shutdown(wait=True)
  Actual: shutdown(wait=False)

pytest introspection follows:

Kwargs:
assert {'wait': False} == {'wait': True}
  Differing items:
  {'wait': False} != {'wait': True}
  Full diff:
  - {'wait': True}
  ?          ^^^
  + {'wait': False}
  ?          ^^^^
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_record_is_valid_with_mock_datetime - AttributeError: <module 'datetime' from '/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/datetime.py'> does not have the attribute 'now'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_cache_make_key - AssertionError: assert 'living_room' in 'a3a903e89a5ba1c24bc46f840d5fcf02'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_cache_put_and_get - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_cache_lru_eviction - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_cache_get_stats - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_cache_clear - TypeError: FeatureCache.put() missing 4 required positional arguments: 'lookback_hours', 'feature_types', 'features', and 'data_hash'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_initialization - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_context_manager - AssertionError: Expected 'initialize' to have been called once. Called 0 times.
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_health_check - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_get_statistics - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_clear_cache - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestFeatureStore::test_feature_store_reset_stats - TypeError: FeatureStore.__init__() got an unexpected keyword argument 'persist_features'
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_get_connection_status - ValueError: second must be in 0..59
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTPublisher::test_mqtt_publisher_connect_to_broker_retry - AssertionError: assert 1 == 2
 +  where 1 = MQTTConnectionStatus(connected=True, last_connected=None, last_disconnected=None, connection_attempts=1, last_error='Connection failed', reconnect_count=0, uptime_seconds=0.0).connection_attempts
 +    where MQTTConnectionStatus(connected=True, last_connected=None, last_disconnected=None, connection_attempts=1, last_error='Connection failed', reconnect_count=0, uptime_seconds=0.0) = <src.integration.mqtt_publisher.MQTTPublisher object at 0x7fed018107d0>.connection_status
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_get_stats - AssertionError: assert 'degraded' == 'healthy'
  - healthy
  + degraded
FAILED tests/unit/data_layer/test_database_operations.py::TestGlobalDatabaseFunctions::test_get_db_session_context_manager - TypeError: 'coroutine' object does not support the asynchronous context manager protocol
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_is_sqlite_engine_with_url_object - TypeError: TestDatabaseCompatibility.test_is_sqlite_engine_with_url_object.<locals>.<lambda>() takes 0 positional arguments but 1 was given
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_is_postgresql_engine_with_postgresql_url - AssertionError: assert False is True
 +  where False = <function is_postgresql_engine at 0x7fd57eaffba0>(<Mock id='140554921152096'>)
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_configure_sensor_event_model_sqlite - AttributeError: <module 'src.data.storage.database_compatibility' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database_compatibility.py'> does not have the attribute 'SensorEvent'
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_configure_sensor_event_model_postgresql - AttributeError: <module 'src.data.storage.database_compatibility' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database_compatibility.py'> does not have the attribute 'SensorEvent'
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_create_database_specific_models_with_sensor_event - TypeError: 'Mock' object is not iterable
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_create_database_specific_models_empty - TypeError: 'Mock' object is not iterable
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_patch_models_for_sqlite_compatibility - TypeError: patch_models_for_sqlite_compatibility() takes 0 positional arguments but 1 was given
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_configure_sqlite_for_testing - AttributeError: 'NoneType' object has no attribute 'info'
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_configure_database_on_first_connect_sqlite - TypeError: configure_database_on_first_connect() takes 2 positional arguments but 3 were given
FAILED tests/unit/data_layer/test_database_operations.py::TestDatabaseCompatibility::test_configure_database_on_first_connect_postgresql - TypeError: configure_database_on_first_connect() takes 2 positional arguments but 3 were given
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_database_dialect_utils_get_dialect_name_postgresql - TypeError: DatabaseDialectUtils() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_database_dialect_utils_get_dialect_name_sqlite - TypeError: DatabaseDialectUtils() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_database_dialect_utils_is_postgresql - TypeError: DatabaseDialectUtils() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_database_dialect_utils_is_sqlite - TypeError: DatabaseDialectUtils() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_percentile_cont_postgresql - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_percentile_cont_sqlite_median - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_percentile_cont_sqlite_quartile - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_percentile_cont_sqlite_other - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_stddev_samp_postgresql - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_stddev_samp_sqlite - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_extract_epoch_postgresql - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_statistical_functions_extract_epoch_sqlite - TypeError: StatisticalFunctions() takes no arguments
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_query_builder_initialization - AttributeError: 'QueryBuilder' object has no attribute 'dialect_name'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_query_builder_build_percentile_query_single - AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_query_builder_build_percentile_query_multiple - AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_query_builder_build_statistics_query_basic - AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_query_builder_build_statistics_query_with_percentiles - AttributeError: <module 'src.data.storage.dialect_utils' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/dialect_utils.py'> does not have the attribute 'select'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_compatibility_manager_initialization - AssertionError: assert False
 +  where False = hasattr(<src.data.storage.dialect_utils.CompatibilityManager object at 0x7fd57e221b50>, 'dialect_utils')
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_global_utility_functions_with_engine - TypeError: extract_epoch_interval() missing 1 required positional argument: 'end_time'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_global_utility_functions_without_engine - TypeError: extract_epoch_interval() missing 1 required positional argument: 'end_time'
FAILED tests/unit/data_layer/test_database_operations.py::TestDialectUtils::test_global_utility_functions_fallback - TypeError: unsupported operand type(s) for *: 'Mock' and 'Mock'
FAILED tests/unit/feature_engineering/test_feature_extraction.py::TestTemporalFeatures::test_extract_historical_patterns_with_dataframe - AttributeError: __getitem__
FAILED tests/unit/ml_models/test_predictive_models.py::TestLSTMPredictor::test_lstm_train_insufficient_data - src.core.exceptions.ModelTrainingError: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ModelTrainingError: Model training failed: lstm for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=lstm, room_id=None | Caused by: ValueError: Insufficient sequence data: only 0 sequences available
FAILED tests/unit/ml_models/test_predictive_models.py::TestLSTMPredictor::test_lstm_predict_success - AssertionError: assert 3 == 1
 +  where 3 = len([PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'}), PredictionResult(predicted_time=datetime.datetime(2025, 8, 24, 8, 53, 36, 794359, tzinfo=datetime.timezone.utc), transition_type='vacant_to_occupied', confidence_score=0.7, prediction_interval=None, alternatives=None, model_type='lstm', model_version='v1.0', features_used=['feature1', 'feature2', 'feature3'], prediction_metadata={'time_until_transition_seconds': 2100.0, 'sequence_length_used': 20, 'prediction_method': 'lstm_neural_network'})])
FAILED tests/unit/ml_models/test_predictive_models.py::TestLSTMPredictor::test_create_sequences_insufficient_data - ValueError: Need at least 50 samples for sequence generation, got 10
FAILED tests/unit/ml_models/test_predictive_models.py::TestLSTMPredictor::test_lstm_save_load_model - assert False is True
FAILED tests/unit/ml_models/test_predictive_models.py::TestXGBoostPredictor::test_xgboost_train_success - src.core.exceptions.ModelTrainingError: Model training failed: xgboost for room living_room | Error Code: MODEL_TRAINING_ERROR | Context: model_type=xgboost, room_id=living_room | Caused by: TypeError: 'Mock' object is not iterable
FAILED tests/unit/ml_models/test_predictive_models.py::TestXGBoostPredictor::test_xgboost_train_failure - src.core.exceptions.ModelTrainingError: Model training failed: xgboost for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=xgboost, room_id=unknown | Caused by: Exception: Training failed
FAILED tests/unit/ml_models/test_predictive_models.py::TestXGBoostPredictor::test_xgboost_predict_success - src.core.exceptions.ModelPredictionError: Model prediction failed: xgboost for room bedroom | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=xgboost, room_id=bedroom | Caused by: TypeError: unsupported type for timedelta seconds component: numpy.int64
FAILED tests/unit/ml_models/test_predictive_models.py::TestXGBoostPredictor::test_xgboost_get_feature_importance_trained - assert 0 == 3
 +  where 0 = len({})
FAILED tests/unit/ml_models/test_predictive_models.py::TestXGBoostPredictor::test_xgboost_save_load_model - assert False is True
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_hmm_train_insufficient_data - src.core.exceptions.ModelTrainingError: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown | Caused by: ModelTrainingError: Model training failed: hmm for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=hmm, room_id=unknown
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_hmm_predict_success - src.core.exceptions.ModelPredictionError: Model prediction failed: hmm for room laundry | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=hmm, room_id=laundry
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_analyze_states - TypeError: HMMPredictor._analyze_states() missing 3 required positional arguments: 'durations', 'feature_names', and 'state_probabilities'
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_build_transition_matrix - AssertionError: assert False
 +  where False = isinstance(None, <class 'numpy.ndarray'>)
 +    where <class 'numpy.ndarray'> = np.ndarray
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_hmm_get_feature_importance_trained - assert 0 == 2
 +  where 0 = len({})
FAILED tests/unit/ml_models/test_predictive_models.py::TestHMMPredictor::test_get_state_info - AssertionError: assert 'labels' in {'n_states': 4, 'state_characteristics': {0: {'mean_duration': 600}, 1: {'mean_duration': 1800}, 2: {'mean_duration': 3600}}, 'state_labels': ['vacant_short', 'occupied', 'vacant_long'], 'transition_matrix': [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.4, 0.2, 0.4]]}
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_gp_train_insufficient_data - src.core.exceptions.ModelTrainingError: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ModelTrainingError: Model training failed: gaussian_process for room None | Error Code: MODEL_TRAINING_ERROR | Context: model_type=gaussian_process, room_id=None | Caused by: ValueError: Insufficient training data: only 8 samples available
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_gp_predict_with_uncertainty - src.core.exceptions.ModelPredictionError: Model prediction failed: gp for room greenhouse | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=gp, room_id=greenhouse | Caused by: TypeError: unsupported type for timedelta seconds component: numpy.int64
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_create_kernel - TypeError: GaussianProcessPredictor._create_kernel() got multiple values for argument 'n_features'
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_gp_get_feature_importance_ard - assert 0.3333333333333333 > 0.3333333333333333
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_uncertainty_metrics - AssertionError: assert ('calibration' in {'confidence_intervals': [0.68, 0.95], 'kernel_type': 'rb', 'log_marginal_likelihood': None, 'sparse_gp': False, ...} or 'uncertainty_range' in {'confidence_intervals': [0.68, 0.95], 'kernel_type': 'rb', 'log_marginal_likelihood': None, 'sparse_gp': False, ...})
FAILED tests/unit/ml_models/test_predictive_models.py::TestGaussianProcessPredictor::test_gp_save_load_model - assert False is True
FAILED tests/unit/ml_models/test_predictive_models.py::TestEnsembleModel::test_ensemble_train_success - src.core.exceptions.ModelTrainingError: Model training failed: ensemble for room lobby | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=lobby | Caused by: ValueError: Targets missing required columns: {'target_time', 'transition_type'}. Available columns: ['time_until_transition_seconds']
FAILED tests/unit/ml_models/test_predictive_models.py::TestEnsembleModel::test_ensemble_train_insufficient_data - src.core.exceptions.ModelTrainingError: Model training failed: ensemble for room unknown | Error Code: MODEL_TRAINING_ERROR | Context: model_type=ensemble, room_id=unknown | Caused by: ValueError: Insufficient data for ensemble training: only 30 samples. Need at least 50 samples.
FAILED tests/unit/ml_models/test_predictive_models.py::TestEnsembleModel::test_ensemble_predict_success - src.core.exceptions.ModelPredictionError: Model prediction failed: ensemble for room unknown | Error Code: MODEL_PREDICTION_ERROR | Context: model_type=ensemble, room_id=unknown
FAILED tests/unit/ml_models/test_predictive_models.py::TestEnsembleModel::test_ensemble_get_feature_importance - assert 0 == 3
 +  where 0 = len({})
FAILED tests/unit/ml_models/test_predictive_models.py::TestEnsembleModel::test_ensemble_save_load_model - assert False is True
FAILED tests/unit/ml_models/test_training_pipeline.py::TestTrainingProgress::test_update_stage - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestTrainingProgress::test_stage_progress_mapping - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestModelTrainingPipeline::test_retraining_pipeline - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestModelTrainingPipeline::test_retraining_with_full_retrain - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestDataSplittingStrategies::test_time_series_split - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestDataSplittingStrategies::test_expanding_window_split - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestDataSplittingStrategies::test_rolling_window_split - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestDataSplittingStrategies::test_holdout_split - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestDataSplittingStrategies::test_split_insufficient_samples - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_train_models_specific_type - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_train_models_failure - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_evaluate_and_select_best_model_empty_results - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestPipelineManagement::test_get_active_pipelines - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestPipelineManagement::test_get_pipeline_history - TypeError: TrainingProgress.__init__() missing 1 required positional argument: 'room_id'
FAILED tests/unit/ml_models/test_training_pipeline.py::TestTrainingIntegrationManager::test_get_cooldown_remaining - assert 5.999999996944444 == 6.0
FAILED tests/unit/utilities/test_system_utilities.py::TestStructuredLogging::test_performance_logger_operation_timing - AssertionError: assert 'Test_operation' in 'Operation completed: test_operation'
FAILED tests/unit/utilities/test_system_utilities.py::TestPerformanceMetrics::test_ml_metrics_collector_record_prediction - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestPerformanceMetrics::test_ml_metrics_collector_record_model_training - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestPerformanceMetrics::test_ml_metrics_collector_system_resources - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestPerformanceMetrics::test_metrics_manager_background_collection - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestPerformanceMetrics::test_multiprocess_metrics_manager - ValueError: env PROMETHEUS_MULTIPROC_DIR is not set or not a directory
FAILED tests/unit/utilities/test_system_utilities.py::TestHealthMonitoring::test_health_monitor_system_resources_check - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestHealthMonitoring::test_health_monitor_critical_resource_usage - ValueError: Duplicated timeseries in CollectorRegistry: {'occupancy_predictor_system_info_info', 'occupancy_predictor_system_info'}
FAILED tests/unit/utilities/test_system_utilities.py::TestMonitoringIntegration::test_monitoring_integration_prediction_error_handling - TypeError: object MagicMock can't be used in 'await' expression
FAILED tests/unit/utilities/test_system_utilities.py::TestMonitoringIntegration::test_monitoring_integration_concept_drift_recording - RuntimeError: no running event loop
FAILED tests/unit/utilities/test_system_utilities.py::TestAlertSystem::test_alert_manager_alert_resolution - AttributeError: 'AlertEvent' object has no attribute 'resolution_notes'
FAILED tests/unit/utilities/test_system_utilities.py::TestIncidentResponse::test_recovery_action_creation_and_limits - AssertionError: assert False is True
 +  where False = <bound method RecoveryAction.can_attempt of RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0)>()
 +    where <bound method RecoveryAction.can_attempt of RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0)> = RecoveryAction(action_type=<RecoveryActionType.RESTART_SERVICE: 'restart_service'>, component='test_service', description='Restart test service', function=<function TestIncidentResponse.test_recovery_action_creation_and_limits.<locals>.mock_recovery_function at 0x7fd57dbecfe0>, conditions={'consecutive_failures': 3}, max_attempts=2, cooldown_minutes=10, last_attempted=datetime.datetime(2025, 8, 24, 8, 18, 43, 869822), attempt_count=1, success_count=0).can_attempt
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_system_status_loop - Failed: Timeout (>30.0s) from pytest-timeout.
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationPerformance::test_connection_status_uptime_calculation - ValueError: second must be in 0..59
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationCompatibility::test_mqtt_client_callback_api_versions - unittest.mock.InvalidSpecError: Cannot spec a Mock object. [object=<MagicMock name='Client' id='140653862784560'>]
FAILED tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationCompatibility::test_mqtt_publisher_graceful_shutdown_with_queued_messages - AssertionError: Expected '_process_message_queue' to have been called once. Called 0 times.
ERROR tests/unit/integration_layer/test_api_services.py
ERROR tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_record_creation - TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_record_validate_against_actual - TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_record_inaccurate_prediction - TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_record_mark_expired - TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/adaptation/test_model_adaptation.py::TestPredictionValidation::test_validation_record_to_dict - TypeError: ValidationRecord.__init__() got an unexpected keyword argument 'confidence'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_connect - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_test_authentication_success - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_test_authentication_failure - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_connect_websocket - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_validate_and_normalize_state - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_should_process_event - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_get_entity_state - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_get_entity_state_not_found - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_get_entity_history - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_convert_ha_event_to_sensor_event - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_disconnect - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_database_manager_initialization_with_config - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_database_manager_initialization_default_config - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_connection_stats_initialization - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_create_engine_postgresql_url_conversion - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_create_engine_with_nullpool_for_testing - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_create_engine_invalid_connection_string - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_setup_connection_events - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_setup_connection_events_without_engine - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_setup_session_factory - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_initialization - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_setup_session_factory_without_engine - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_initialize - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_verify_connection_success - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_health_check - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_health_check_failure - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_verify_connection_timescaledb_missing - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_get_session - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_verify_connection_without_engine - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_execute_query - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_session_success - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestDatabaseManager::test_database_manager_close - AttributeError: Mock object has no attribute 'database'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_session_without_factory - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_data_models.py::TestHAClient::test_ha_client_initialization - AttributeError: Mock object has no attribute 'home_assistant'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_session_with_retry_on_connection_error - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_session_retry_exhaustion - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_query_basic - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_query_with_timeout - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_query_timeout_error - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_query_sql_error - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_optimized_query_with_prepared_statements - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_execute_optimized_query_prepared_statement_fallback - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_analyze_query_performance - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_optimization_suggestions - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_connection_pool_metrics - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_connection_pool_metrics_high_utilization - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_connection_pool_metrics_without_engine - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_publish_prediction_success - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
ERROR tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_publish_prediction_failure - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_health_check_success - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationManager::test_integration_manager_publish_prediction_inactive - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_health_check_failure - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_health_check_loop - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_is_initialized_property - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_get_connection_stats - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_cleanup - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/data_layer/test_database_operations.py::TestDatabaseManager::test_cleanup_with_completed_task - AttributeError: <module 'src.data.storage.database' from '/home/runner/work/ha-ml-predictor/ha-ml-predictor/src/data/storage/database.py'> does not have the attribute 'CompatibilityManager'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTrainingPipeline::test_initial_training_success - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTrainingPipeline::test_train_room_models_success - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_train_models_ensemble - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_validate_models - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_evaluate_and_select_best_model - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelTraining::test_deploy_trained_models - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestModelArtifacts::test_save_model_artifacts - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestPipelineManagement::test_get_model_registry - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/ml_models/test_training_pipeline.py::TestPipelineManagement::test_get_model_performance - TypeError: TrainingResult.__init__() missing 1 required positional argument: 'model_version'
ERROR tests/unit/integration_layer/test_mqtt_integration.py::TestMQTTIntegrationErrorHandling::test_publish_prediction_exception_handling - TypeError: PredictionResult.__init__() got an unexpected keyword argument 'prediction_time'
= 256 failed, 750 passed, 7 skipped, 576 warnings, 73 errors in 70.87s (0:01:10) =
Error: Process completed with exit code 1.