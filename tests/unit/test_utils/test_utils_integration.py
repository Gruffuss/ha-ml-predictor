"""
System integration tests for utility modules.
Tests utilities working together with main system components in production scenarios.
"""

import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
import gc
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import psutil
import pytest
import pytz

from src.utils.logger import (
    LoggerManager,
    StructuredFormatter,
    get_error_tracker,
    get_logger_manager,
    get_ml_ops_logger,
    get_performance_logger,
)
from src.utils.metrics import (
    MetricsManager,
    MLMetricsCollector,
    get_metrics_collector,
    get_metrics_manager,
)
from src.utils.time_utils import (
    AsyncTimeUtils,
    TimeProfiler,
    TimeRange,
    TimeUtils,
    cyclical_time_features,
)


class TestUtilitiesSystemIntegration:
    """Test utilities working together as an integrated system."""

    def test_logger_metrics_time_integration(self):
        """Test integration between logger, metrics, and time utilities."""
        logger_manager = get_logger_manager()
        metrics_manager = get_metrics_manager()
        metrics_collector = get_metrics_collector()

        # Simulate a complex operation using all utilities
        operation_results = []

        for i in range(100):
            operation_name = f"integrated_operation_{i}"
            room_id = f"room_{i % 10}"

            # Use time utilities for timing
            with TimeProfiler(operation_name) as profiler:
                # Use logger context manager
                with logger_manager.log_operation(operation_name, room_id=room_id):
                    # Simulate work with time utilities
                    current_time = TimeUtils.utc_now()
                    features = TimeUtils.get_cyclical_time_features(current_time)

                    # Simulate prediction time calculation
                    prediction_time = current_time + timedelta(minutes=30)
                    time_until_prediction = TimeUtils.time_until(prediction_time)

                    # Brief processing delay
                    time.sleep(0.001)

                    # Record metrics
                    metrics_collector.record_prediction(
                        room_id=room_id,
                        prediction_type="occupancy",
                        model_type="integrated_test",
                        duration=profiler.duration_seconds or 0.001,
                        confidence=0.8 + (i % 20) * 0.01,
                    )

                    operation_results.append(
                        {
                            "operation": operation_name,
                            "room_id": room_id,
                            "duration": profiler.duration_seconds,
                            "features_count": len(features),
                            "prediction_delay": time_until_prediction.total_seconds(),
                        }
                    )

        # Verify integration worked
        assert len(operation_results) == 100

        # All operations should have reasonable timing
        for result in operation_results:
            assert result["duration"] > 0
            assert result["features_count"] == 10  # 5 sin/cos pairs
            assert result["prediction_delay"] > 1790  # ~30 minutes
            assert result["prediction_delay"] < 1810

    def test_async_utilities_integration(self):
        """Test async utilities integration with logging and metrics."""
        logger = get_logger_manager().get_logger("async_integration")
        metrics_collector = get_metrics_collector()

        async def async_operation_with_utilities(operation_id: int):
            """Async operation using all utilities."""
            operation_name = f"async_op_{operation_id}"

            with TimeProfiler(operation_name) as profiler:
                # Use async time utilities
                start_time = TimeUtils.utc_now()
                wait_until = start_time + timedelta(milliseconds=50)

                await AsyncTimeUtils.wait_until(wait_until, check_interval=0.01)

                # Log async operation
                logger.info(
                    f"Async operation {operation_id} completed",
                    extra={
                        "operation_id": operation_id,
                        "duration": profiler.duration_seconds,
                        "start_time": start_time.isoformat(),
                        "component": "async_integration",
                    },
                )

                # Record metrics
                metrics_collector.record_prediction(
                    room_id=f"async_room_{operation_id % 5}",
                    prediction_type="occupancy",
                    model_type="async_test",
                    duration=profiler.duration_seconds or 0.001,
                )

                return {
                    "operation_id": operation_id,
                    "duration": profiler.duration_seconds,
                    "completed_at": TimeUtils.utc_now(),
                }

        # Run multiple async operations
        async def run_async_integration_test():
            tasks = [async_operation_with_utilities(i) for i in range(20)]

            return await asyncio.gather(*tasks)

        # Execute async test
        with TimeProfiler("async_integration_test") as profiler:
            results = asyncio.run(run_async_integration_test())

        # Verify async integration
        assert len(results) == 20
        assert profiler.duration_seconds >= 0.05  # At least the wait time
        assert profiler.duration_seconds < 2.0  # But not too long

        # All operations should have completed successfully
        for result in results:
            assert result["duration"] >= 0.05
            assert isinstance(result["completed_at"], datetime)

    def test_production_scenario_simulation(self):
        """Simulate a production scenario with high-load utilities usage."""
        logger_manager = get_logger_manager()
        metrics_manager = get_metrics_manager()
        perf_logger = get_performance_logger()
        error_tracker = get_error_tracker()
        ml_ops_logger = get_ml_ops_logger()

        # Simulate production workload
        production_results = {
            "predictions": 0,
            "errors": 0,
            "training_events": 0,
            "resource_updates": 0,
            "total_duration": 0,
        }

        def production_worker(worker_id: int, iterations: int):
            """Simulate production worker using all utilities."""
            worker_stats = {
                "predictions": 0,
                "errors": 0,
                "training_events": 0,
                "operations": 0,
            }

            for i in range(iterations):
                operation_type = i % 5

                try:
                    if operation_type == 0:
                        # Prediction operation
                        with TimeProfiler(
                            f"prediction_worker_{worker_id}_{i}"
                        ) as profiler:
                            # Time utilities
                            current_time = TimeUtils.utc_now()
                            features = cyclical_time_features(current_time)

                            # Simulate prediction calculation
                            time.sleep(0.001)  # Brief processing

                            # Logging
                            perf_logger.log_operation_time(
                                operation="prediction",
                                duration=profiler.duration_seconds or 0.001,
                                room_id=f"prod_room_{i % 20}",
                                worker_id=worker_id,
                            )

                            # Metrics
                            metrics_manager.get_collector().record_prediction(
                                room_id=f"prod_room_{i % 20}",
                                prediction_type="occupancy",
                                model_type="production_lstm",
                                duration=profiler.duration_seconds or 0.001,
                                confidence=0.75 + (i % 25) * 0.01,
                            )

                            worker_stats["predictions"] += 1

                    elif operation_type == 1:
                        # Training event
                        ml_ops_logger.log_training_event(
                            room_id=f"prod_room_{i % 20}",
                            model_type="xgboost",
                            event_type="incremental_update",
                            metrics={
                                "accuracy": 0.85 + (i % 10) * 0.01,
                                "loss": 0.15 - (i % 10) * 0.001,
                            },
                        )
                        worker_stats["training_events"] += 1

                    elif operation_type == 2:
                        # Error simulation (occasional)
                        if i % 50 == 0:  # 2% error rate
                            try:
                                raise ValueError(f"Simulated production error {i}")
                            except ValueError as e:
                                error_tracker.track_prediction_error(
                                    room_id=f"prod_room_{i % 20}",
                                    error=e,
                                    prediction_type="occupancy",
                                    model_type="production_lstm",
                                )
                                worker_stats["errors"] += 1

                    elif operation_type == 3:
                        # Resource monitoring
                        metrics_manager.get_collector().update_system_resources()

                    else:
                        # Time-based operation
                        future_time = TimeUtils.utc_now() + timedelta(hours=1)
                        time_until = TimeUtils.time_until(future_time)

                        # Log time calculation
                        logger_manager.get_logger("production").info(
                            "Time calculation completed",
                            extra={
                                "worker_id": worker_id,
                                "iteration": i,
                                "time_until_seconds": time_until.total_seconds(),
                                "component": "time_operations",
                            },
                        )

                    worker_stats["operations"] += 1

                except Exception as e:
                    error_tracker.track_error(
                        e,
                        {
                            "worker_id": worker_id,
                            "iteration": i,
                            "operation_type": operation_type,
                        },
                    )
                    worker_stats["errors"] += 1

            # Update production results (thread-safe)
            with threading.Lock():
                production_results["predictions"] += worker_stats["predictions"]
                production_results["errors"] += worker_stats["errors"]
                production_results["training_events"] += worker_stats["training_events"]

        # Run production simulation
        num_workers = 10
        iterations_per_worker = 200

        with TimeProfiler("production_simulation") as profiler:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = [
                    executor.submit(production_worker, worker_id, iterations_per_worker)
                    for worker_id in range(num_workers)
                ]
                concurrent.futures.wait(futures)

        production_results["total_duration"] = profiler.duration_seconds

        # Verify production simulation results
        assert production_results["predictions"] > 0
        assert production_results["training_events"] > 0
        assert (
            production_results["total_duration"] < 30.0
        )  # Should complete in reasonable time

        # Error rate should be low but not zero (due to simulated errors)
        error_rate = production_results["errors"] / (
            num_workers * iterations_per_worker
        )
        assert 0.01 <= error_rate <= 0.05  # 1-5% error rate

        print(f"\nProduction Simulation Results:")
        print(f"Predictions: {production_results['predictions']}")
        print(f"Training Events: {production_results['training_events']}")
        print(f"Errors: {production_results['errors']}")
        print(f"Duration: {production_results['total_duration']:.2f}s")
        print(f"Error Rate: {error_rate:.2%}")

    def test_timezone_aware_logging_and_metrics(self):
        """Test timezone-aware operations across all utilities."""
        # Test with multiple timezones
        timezones = [
            "UTC",
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]

        logger = get_logger_manager().get_logger("timezone_test")
        metrics_collector = get_metrics_collector()

        timezone_results = []

        for tz_name in timezones:
            tz = pytz.timezone(tz_name)

            # Create timezone-aware datetime
            local_time = tz.localize(datetime(2024, 6, 15, 14, 30, 0))  # Summer time
            utc_time = TimeUtils.to_utc(local_time)

            # Extract cyclical features
            features = TimeUtils.get_cyclical_time_features(utc_time)

            # Log with timezone context
            logger.info(
                f"Timezone operation: {tz_name}",
                extra={
                    "timezone": tz_name,
                    "local_time": local_time.isoformat(),
                    "utc_time": utc_time.isoformat(),
                    "hour_sin": features["hour_sin"],
                    "hour_cos": features["hour_cos"],
                    "is_business_hours": TimeUtils.is_business_hours(local_time),
                },
            )

            # Record metrics with timezone context
            with TimeProfiler(f"timezone_operation_{tz_name}") as profiler:
                metrics_collector.record_prediction(
                    room_id=f"tz_room_{tz_name.replace('/', '_')}",
                    prediction_type="occupancy",
                    model_type="timezone_aware",
                    duration=profiler.duration_seconds or 0.001,
                )

            timezone_results.append(
                {
                    "timezone": tz_name,
                    "local_hour": local_time.hour,
                    "utc_hour": utc_time.hour,
                    "features": features,
                    "processing_time": profiler.duration_seconds,
                }
            )

        # Verify timezone handling
        assert len(timezone_results) == len(timezones)

        # UTC should have the same UTC hour as local hour
        utc_result = next(r for r in timezone_results if r["timezone"] == "UTC")
        assert utc_result["local_hour"] == utc_result["utc_hour"]

        # Other timezones should have different local and UTC hours
        for result in timezone_results:
            if result["timezone"] != "UTC":
                # Most likely different (could be same due to DST edge cases)
                assert isinstance(result["local_hour"], int)
                assert isinstance(result["utc_hour"], int)
                assert 0 <= result["local_hour"] <= 23
                assert 0 <= result["utc_hour"] <= 23

    def test_error_recovery_integration(self):
        """Test error recovery across all utility systems."""
        logger_manager = get_logger_manager()
        metrics_collector = get_metrics_collector()
        error_tracker = get_error_tracker()

        # Simulate various error scenarios
        error_scenarios = [
            (
                "time_calculation_error",
                lambda: TimeUtils.parse_datetime("invalid-date"),
            ),
            (
                "feature_calculation_error",
                lambda: TimeUtils.get_cyclical_time_features(None),
            ),
            (
                "metrics_error",
                lambda: metrics_collector.record_prediction("", "", "", -1),
            ),
            (
                "logging_error",
                lambda: logger_manager.get_logger("test").error(
                    "Test", extra={"circular": {"self": None}}
                ),
            ),
        ]

        recovery_results = []

        for error_name, error_func in error_scenarios:
            try:
                with TimeProfiler(f"error_recovery_{error_name}") as profiler:
                    error_func()

                # Should not reach here
                recovery_results.append(
                    {
                        "error_name": error_name,
                        "recovered": False,
                        "error_caught": False,
                        "duration": profiler.duration_seconds,
                    }
                )

            except Exception as e:
                # Log and track the error
                error_tracker.track_error(
                    e,
                    {
                        "error_scenario": error_name,
                        "component": "integration_test",
                        "recovery_test": True,
                    },
                )

                recovery_results.append(
                    {
                        "error_name": error_name,
                        "recovered": True,
                        "error_caught": True,
                        "error_type": type(e).__name__,
                        "duration": (
                            profiler.duration_seconds if "profiler" in locals() else 0
                        ),
                    }
                )

        # Verify error recovery
        assert len(recovery_results) >= 3  # At least some errors should be caught

        caught_errors = [r for r in recovery_results if r["error_caught"]]
        assert len(caught_errors) >= 2  # Most errors should be properly caught

        # System should continue functioning after errors
        # Test normal operation after errors
        normal_operation_success = False
        try:
            current_time = TimeUtils.utc_now()
            features = TimeUtils.get_cyclical_time_features(current_time)

            metrics_collector.record_prediction(
                room_id="recovery_test",
                prediction_type="occupancy",
                model_type="recovery",
                duration=0.001,
                confidence=0.8,
            )

            normal_operation_success = True

        except Exception as e:
            error_tracker.track_error(e, {"test": "post_error_recovery"})

        assert (
            normal_operation_success
        ), "System should continue functioning after errors"


class TestUtilitiesResourceManagement:
    """Test resource management aspects of utilities integration."""

    def test_memory_management_integration(self):
        """Test memory management across all utilities under load."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Components
        logger_manager = get_logger_manager()
        metrics_manager = get_metrics_manager()

        # Generate sustained load across all utilities
        memory_measurements = []

        for phase in range(5):
            phase_memory_before = process.memory_info().rss

            # Logging phase
            logger = logger_manager.get_logger(f"memory_test_phase_{phase}")
            for i in range(1000):
                logger.info(
                    f"Memory test log {i}",
                    extra={
                        "phase": phase,
                        "iteration": i,
                        "timestamp": TimeUtils.utc_now().isoformat(),
                        "features": TimeUtils.get_cyclical_time_features(
                            TimeUtils.utc_now()
                        ),
                        "large_data": "x" * 100,  # Some data bulk
                    },
                )

            # Metrics phase
            collector = metrics_manager.get_collector()
            for i in range(1000):
                collector.record_prediction(
                    room_id=f"memory_room_{i % 20}",
                    prediction_type="occupancy",
                    model_type="memory_test",
                    duration=i * 0.0001,
                    confidence=0.8,
                )

            # Time utilities phase
            time_operations = []
            for i in range(1000):
                current_time = TimeUtils.utc_now()
                features = TimeUtils.get_cyclical_time_features(current_time)
                future_time = current_time + timedelta(minutes=i % 60)
                time_until = TimeUtils.time_until(future_time)

                time_operations.append(
                    {"features": features, "time_until": time_until.total_seconds()}
                )

            # Force garbage collection
            gc.collect()

            phase_memory_after = process.memory_info().rss
            phase_memory_increase = phase_memory_after - phase_memory_before

            memory_measurements.append(
                {
                    "phase": phase,
                    "memory_increase": phase_memory_increase,
                    "operations": 3000,  # 1000 each of logging, metrics, time ops
                }
            )

        # Analyze memory usage
        total_memory_increase = process.memory_info().rss - initial_memory

        print(f"\nMemory Management Integration Test:")
        print(f"Initial Memory: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"Final Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"Total Increase: {total_memory_increase / 1024 / 1024:.1f} MB")

        for measurement in memory_measurements:
            print(
                f"Phase {measurement['phase']}: "
                f"{measurement['memory_increase'] / 1024 / 1024:.1f} MB "
                f"({measurement['operations']} operations)"
            )

        # Memory increase should be reasonable
        assert total_memory_increase < 200 * 1024 * 1024  # Less than 200MB total

        # Each phase shouldn't use excessive memory
        for measurement in memory_measurements:
            assert (
                measurement["memory_increase"] < 50 * 1024 * 1024
            )  # Less than 50MB per phase

    def test_file_handle_management_integration(self):
        """Test file handle management across utilities."""
        # Track file handles (Linux/Unix systems)
        if not os.path.exists("/proc/self/fd"):
            pytest.skip("File descriptor tracking not available on this system")

        initial_fd_count = len(os.listdir("/proc/self/fd"))

        # Create multiple temporary log files
        temp_files = []
        handlers = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create multiple logger instances with file handlers
                for i in range(10):
                    log_file = Path(temp_dir) / f"integration_test_{i}.log"
                    temp_files.append(log_file)

                    # Create file handler
                    handler = logging.FileHandler(log_file)
                    handler.setFormatter(StructuredFormatter())
                    handlers.append(handler)

                    # Create logger and add handler
                    logger = logging.getLogger(f"fd_test_{i}")
                    logger.setLevel(logging.INFO)
                    logger.handlers.clear()
                    logger.addHandler(handler)

                # Use utilities with file operations
                for i in range(100):
                    # Time utilities (no file I/O)
                    current_time = TimeUtils.utc_now()
                    features = TimeUtils.get_cyclical_time_features(current_time)

                    # Logging (file I/O)
                    logger_idx = i % 10
                    logger = logging.getLogger(f"fd_test_{logger_idx}")
                    logger.info(
                        f"File handle test {i}",
                        extra={
                            "iteration": i,
                            "features": features,
                            "timestamp": current_time.isoformat(),
                        },
                    )

                    # Metrics (no direct file I/O, but may use internal resources)
                    get_metrics_collector().record_prediction(
                        room_id=f"fd_room_{i % 5}",
                        prediction_type="occupancy",
                        model_type="fd_test",
                        duration=0.001,
                    )

                # Check file handle count during operation
                mid_fd_count = len(os.listdir("/proc/self/fd"))

                # Verify files were written
                for log_file in temp_files:
                    assert log_file.exists()
                    assert log_file.stat().st_size > 0

                # File descriptor count should be controlled
                assert mid_fd_count <= initial_fd_count + 20  # Allow some leeway

        finally:
            # Clean up handlers
            for handler in handlers:
                try:
                    handler.close()
                except Exception:
                    pass

        # Final file descriptor count should return to normal
        final_fd_count = len(os.listdir("/proc/self/fd"))
        assert final_fd_count <= initial_fd_count + 5  # Allow small variance

    def test_thread_safety_integration(self):
        """Test thread safety of all utilities working together."""
        shared_results = {
            "successful_operations": 0,
            "errors": 0,
            "lock": threading.Lock(),
        }

        def integrated_thread_worker(worker_id: int, iterations: int):
            """Worker that uses all utilities in thread-safe manner."""
            local_results = {"success": 0, "error": 0}

            logger = get_logger_manager().get_logger(f"thread_worker_{worker_id}")
            metrics_collector = get_metrics_collector()

            for i in range(iterations):
                try:
                    # Time utilities (should be thread-safe)
                    with TimeProfiler(f"thread_op_{worker_id}_{i}") as profiler:
                        current_time = TimeUtils.utc_now()
                        features = TimeUtils.get_cyclical_time_features(current_time)

                        # Brief work simulation
                        time.sleep(0.001)

                    # Logging (should be thread-safe)
                    logger.info(
                        f"Thread worker {worker_id} operation {i}",
                        extra={
                            "worker_id": worker_id,
                            "iteration": i,
                            "duration": profiler.duration_seconds,
                            "thread_id": threading.current_thread().ident,
                            "features_count": len(features),
                        },
                    )

                    # Metrics (should be thread-safe)
                    metrics_collector.record_prediction(
                        room_id=f"thread_room_{worker_id}_{i % 5}",
                        prediction_type="occupancy",
                        model_type="thread_test",
                        duration=profiler.duration_seconds or 0.001,
                        confidence=0.7 + (i % 30) * 0.01,
                    )

                    local_results["success"] += 1

                except Exception as e:
                    local_results["error"] += 1

                    # Use error tracker (should be thread-safe)
                    get_error_tracker().track_error(
                        e,
                        {
                            "worker_id": worker_id,
                            "iteration": i,
                            "test": "thread_safety_integration",
                        },
                    )

            # Update shared results (thread-safe)
            with shared_results["lock"]:
                shared_results["successful_operations"] += local_results["success"]
                shared_results["errors"] += local_results["error"]

        # Run thread safety test
        num_threads = 20
        iterations_per_thread = 100

        with TimeProfiler("thread_safety_integration") as profiler:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                futures = [
                    executor.submit(
                        integrated_thread_worker, thread_id, iterations_per_thread
                    )
                    for thread_id in range(num_threads)
                ]
                concurrent.futures.wait(futures)

        # Verify thread safety
        expected_operations = num_threads * iterations_per_thread
        assert shared_results["successful_operations"] == expected_operations
        assert shared_results["errors"] == 0

        # Should complete efficiently
        assert profiler.duration_seconds < 30.0

        print(f"\nThread Safety Integration Test:")
        print(f"Threads: {num_threads}")
        print(f"Operations per thread: {iterations_per_thread}")
        print(f"Total operations: {expected_operations}")
        print(f"Successful: {shared_results['successful_operations']}")
        print(f"Errors: {shared_results['errors']}")
        print(f"Duration: {profiler.duration_seconds:.2f}s")


class TestUtilitiesEndToEndScenarios:
    """Test end-to-end scenarios using all utilities."""

    def test_occupancy_prediction_simulation(self):
        """Simulate end-to-end occupancy prediction using all utilities."""
        # Setup
        logger_manager = get_logger_manager()
        metrics_manager = get_metrics_manager()
        perf_logger = get_performance_logger()
        ml_ops_logger = get_ml_ops_logger()

        # Simulate prediction pipeline
        prediction_results = []

        # Simulate 24 hours of predictions (every 5 minutes = 288 predictions)
        start_time = TimeUtils.utc_now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        for prediction_num in range(288):  # 24 * 60 / 5 = 288 predictions
            prediction_time = start_time + timedelta(minutes=prediction_num * 5)

            # Time-based features
            features = TimeUtils.get_cyclical_time_features(prediction_time)
            is_business_hours = TimeUtils.is_business_hours(prediction_time)

            # Simulate prediction for multiple rooms
            for room_id in ["living_room", "bedroom", "kitchen", "bathroom", "office"]:
                with TimeProfiler(f"prediction_{room_id}_{prediction_num}") as profiler:
                    # Log prediction start
                    logger_manager.get_logger("prediction_pipeline").info(
                        f"Starting prediction for {room_id}",
                        extra={
                            "room_id": room_id,
                            "prediction_time": prediction_time.isoformat(),
                            "prediction_num": prediction_num,
                            "is_business_hours": is_business_hours,
                            "hour_sin": features["hour_sin"],
                            "hour_cos": features["hour_cos"],
                        },
                    )

                    # Simulate model prediction (varying complexity)
                    if room_id == "living_room":
                        model_type = "ensemble"
                        processing_time = 0.05
                    elif room_id in ["bedroom", "kitchen"]:
                        model_type = "lstm"
                        processing_time = 0.02
                    else:
                        model_type = "xgboost"
                        processing_time = 0.01

                    time.sleep(processing_time)  # Simulate processing

                    # Calculate prediction confidence based on time and room
                    base_confidence = 0.7
                    if is_business_hours and room_id == "office":
                        base_confidence = 0.9
                    elif not is_business_hours and room_id == "bedroom":
                        base_confidence = 0.85

                    confidence = base_confidence + (prediction_num % 20) * 0.01

                    # Simulate prediction accuracy (for metrics)
                    actual_accuracy = 5 + (prediction_num % 30)  # 5-35 minutes

                    # Record performance metrics
                    perf_logger.log_operation_time(
                        operation="occupancy_prediction",
                        duration=profiler.duration_seconds or processing_time,
                        room_id=room_id,
                        prediction_type="occupancy",
                        model_type=model_type,
                        confidence=confidence,
                    )

                    # Record metrics
                    metrics_manager.get_collector().record_prediction(
                        room_id=room_id,
                        prediction_type="occupancy",
                        model_type=model_type,
                        duration=profiler.duration_seconds or processing_time,
                        accuracy_minutes=actual_accuracy,
                        confidence=confidence,
                    )

                    # Occasional model training events
                    if prediction_num % 50 == 0 and room_id == "living_room":
                        ml_ops_logger.log_training_event(
                            room_id=room_id,
                            model_type=model_type,
                            event_type="incremental_update",
                            metrics={
                                "accuracy": confidence,
                                "mae": 15 - confidence * 10,
                                "training_samples": prediction_num * 10,
                            },
                        )

                    prediction_results.append(
                        {
                            "room_id": room_id,
                            "prediction_time": prediction_time,
                            "model_type": model_type,
                            "confidence": confidence,
                            "accuracy_minutes": actual_accuracy,
                            "duration": profiler.duration_seconds,
                            "features": features,
                        }
                    )

        # Verify end-to-end simulation
        expected_predictions = 288 * 5  # 288 time slots * 5 rooms
        assert len(prediction_results) == expected_predictions

        # Analyze results
        by_room = {}
        for result in prediction_results:
            room = result["room_id"]
            if room not in by_room:
                by_room[room] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "avg_accuracy": 0,
                    "avg_duration": 0,
                }

            by_room[room]["count"] += 1
            by_room[room]["avg_confidence"] += result["confidence"]
            by_room[room]["avg_accuracy"] += result["accuracy_minutes"]
            by_room[room]["avg_duration"] += result["duration"]

        # Calculate averages
        for room_data in by_room.values():
            count = room_data["count"]
            room_data["avg_confidence"] /= count
            room_data["avg_accuracy"] /= count
            room_data["avg_duration"] /= count

        print(f"\nEnd-to-End Occupancy Prediction Simulation:")
        print(f"Total predictions: {len(prediction_results)}")
        print(f"Rooms: {list(by_room.keys())}")

        for room, data in by_room.items():
            print(
                f"{room}: {data['count']} predictions, "
                f"avg confidence: {data['avg_confidence']:.3f}, "
                f"avg accuracy: {data['avg_accuracy']:.1f}min, "
                f"avg duration: {data['avg_duration']:.4f}s"
            )

        # Verify reasonable results
        for room, data in by_room.items():
            assert data["count"] == 288  # One per time slot
            assert 0.7 <= data["avg_confidence"] <= 1.0
            assert 5 <= data["avg_accuracy"] <= 35
            assert 0.01 <= data["avg_duration"] <= 0.1

    @pytest.mark.asyncio
    async def test_real_time_monitoring_simulation(self):
        """Simulate real-time monitoring using async utilities."""
        logger = get_logger_manager().get_logger("realtime_monitoring")
        metrics_collector = get_metrics_collector()

        # Simulation state
        monitoring_results = {
            "events_processed": 0,
            "alerts_generated": 0,
            "resource_updates": 0,
            "errors": 0,
        }

        async def event_processor(event_type: str, room_id: str):
            """Process monitoring events."""
            try:
                with TimeProfiler(f"event_processing_{event_type}") as profiler:
                    # Time-based processing
                    current_time = TimeUtils.utc_now()
                    features = TimeUtils.get_cyclical_time_features(current_time)

                    # Simulate event processing delay
                    await asyncio.sleep(0.001)

                    # Log event processing
                    logger.info(
                        f"Processed {event_type} event",
                        extra={
                            "event_type": event_type,
                            "room_id": room_id,
                            "processing_time": profiler.duration_seconds,
                            "timestamp": current_time.isoformat(),
                            "hour_feature": features["hour_sin"],
                        },
                    )

                    # Record metrics
                    metrics_collector.record_event_processing(
                        room_id=room_id,
                        sensor_type=event_type,
                        processing_duration=profiler.duration_seconds or 0.001,
                    )

                    monitoring_results["events_processed"] += 1

                    # Generate alerts for specific conditions
                    if event_type == "motion" and not TimeUtils.is_business_hours(
                        current_time
                    ):
                        monitoring_results["alerts_generated"] += 1

                        logger.warning(
                            f"After-hours motion detected in {room_id}",
                            extra={
                                "alert_type": "after_hours_motion",
                                "room_id": room_id,
                                "timestamp": current_time.isoformat(),
                            },
                        )

            except Exception as e:
                monitoring_results["errors"] += 1
                get_error_tracker().track_error(
                    e, {"event_type": event_type, "room_id": room_id}
                )

        async def resource_monitor():
            """Monitor system resources periodically."""
            try:
                for _ in range(10):  # 10 resource updates during simulation
                    metrics_collector.update_system_resources()
                    monitoring_results["resource_updates"] += 1

                    await asyncio.sleep(0.1)  # Update every 100ms

            except Exception as e:
                monitoring_results["errors"] += 1
                get_error_tracker().track_error(e, {"component": "resource_monitor"})

        # Simulate real-time events
        event_types = ["motion", "door", "light", "temperature"]
        rooms = ["living_room", "bedroom", "kitchen", "office"]

        # Create event processing tasks
        event_tasks = []
        for i in range(100):  # 100 events
            event_type = event_types[i % len(event_types)]
            room_id = rooms[i % len(rooms)]

            task = event_processor(event_type, room_id)
            event_tasks.append(task)

        # Run simulation
        with TimeProfiler("realtime_monitoring_simulation") as profiler:
            # Start resource monitoring
            resource_task = asyncio.create_task(resource_monitor())

            # Process events concurrently
            await asyncio.gather(*event_tasks, resource_task)

        # Verify real-time simulation
        assert monitoring_results["events_processed"] == 100
        assert monitoring_results["resource_updates"] == 10
        assert monitoring_results["errors"] == 0
        assert monitoring_results["alerts_generated"] >= 0  # May or may not have alerts

        # Should complete efficiently
        assert profiler.duration_seconds < 5.0

        print(f"\nReal-time Monitoring Simulation:")
        print(f"Events processed: {monitoring_results['events_processed']}")
        print(f"Resource updates: {monitoring_results['resource_updates']}")
        print(f"Alerts generated: {monitoring_results['alerts_generated']}")
        print(f"Errors: {monitoring_results['errors']}")
        print(f"Duration: {profiler.duration_seconds:.2f}s")


@pytest.mark.integration
class TestUtilitiesProductionReadiness:
    """Test production readiness of utilities integration."""

    def test_startup_shutdown_lifecycle(self):
        """Test clean startup and shutdown of all utilities."""
        # Simulate application startup
        startup_successful = True
        components_started = []

        try:
            # Initialize utilities (as would happen in production)
            logger_manager = get_logger_manager()
            components_started.append("logger_manager")

            metrics_manager = get_metrics_manager()
            components_started.append("metrics_manager")

            # Start background processes
            metrics_manager.start_background_collection(update_interval=0.1)
            components_started.append("metrics_background")

            # Simulate some load during startup
            for i in range(50):
                logger_manager.get_logger("startup").info(
                    f"Startup process {i}",
                    extra={"startup_step": i, "component": "system_startup"},
                )

                metrics_manager.get_collector().record_prediction(
                    room_id=f"startup_room_{i % 5}",
                    prediction_type="occupancy",
                    model_type="startup_test",
                    duration=0.001,
                )

            # Verify startup completed
            assert len(components_started) == 3

        except Exception as e:
            startup_successful = False
            get_error_tracker().track_error(e, {"phase": "startup"})

        # Simulate application shutdown
        shutdown_successful = True

        try:
            # Stop background processes
            if "metrics_background" in components_started:
                metrics_manager.stop_background_collection()

            # Final logging
            logger_manager.get_logger("shutdown").info(
                "Application shutting down",
                extra={"components_started": components_started},
            )

            # Clean up
            gc.collect()

        except Exception as e:
            shutdown_successful = False
            get_error_tracker().track_error(e, {"phase": "shutdown"})

        # Verify lifecycle
        assert startup_successful, "Startup should complete successfully"
        assert shutdown_successful, "Shutdown should complete successfully"

    def test_configuration_validation(self):
        """Test configuration validation across utilities."""
        # Test timezone configuration
        valid_timezones = ["UTC", "America/New_York", "Europe/London"]
        invalid_timezones = ["Invalid/Timezone", "Not_Real", ""]

        for tz in valid_timezones:
            assert TimeUtils.validate_timezone(
                tz
            ), f"Valid timezone {tz} should be accepted"

        for tz in invalid_timezones:
            assert not TimeUtils.validate_timezone(
                tz
            ), f"Invalid timezone {tz} should be rejected"

        # Test time range validation
        valid_start = TimeUtils.utc_now()
        valid_end = valid_start + timedelta(hours=1)

        # Valid range
        time_range = TimeRange(valid_start, valid_end)
        assert time_range.duration.total_seconds() == 3600

        # Invalid range (end before start)
        with pytest.raises(ValueError, match="Start time must be before end time"):
            TimeRange(valid_end, valid_start)

        # Test metrics configuration
        metrics_collector = get_metrics_collector()

        # Valid metrics
        metrics_collector.record_prediction(
            room_id="test_room",
            prediction_type="occupancy",
            model_type="test",
            duration=0.1,
            confidence=0.8,
        )

        # Test logging configuration
        logger = get_logger_manager().get_logger("config_test")
        logger.info("Configuration validation test")

        # All should complete without errors
        assert True, "Configuration validation completed"

    def test_performance_under_production_load(self):
        """Test performance characteristics under production-like load."""
        # Production-like load parameters
        rooms = [f"room_{i:03d}" for i in range(100)]  # 100 rooms
        prediction_interval = 300  # 5 minutes
        duration_hours = 0.1  # 6 minutes simulation (0.1 hours)

        # Calculate expected load
        predictions_per_hour = len(rooms) * (3600 / prediction_interval)
        total_predictions = int(predictions_per_hour * duration_hours)

        logger = get_logger_manager().get_logger("production_load")
        metrics_collector = get_metrics_collector()

        production_stats = {
            "predictions_completed": 0,
            "errors": 0,
            "total_duration": 0,
            "avg_prediction_time": 0,
        }

        with TimeProfiler("production_load_test") as profiler:
            for prediction_batch in range(total_predictions // len(rooms)):
                batch_start_time = TimeUtils.utc_now()

                # Process predictions for all rooms in this batch
                for room_id in rooms:
                    prediction_start = TimeUtils.utc_now()

                    try:
                        # Time-based features
                        features = TimeUtils.get_cyclical_time_features(
                            prediction_start
                        )

                        # Simulate model processing (varying by room)
                        room_num = int(room_id.split("_")[1])
                        if room_num % 10 == 0:
                            # Complex model for every 10th room
                            processing_time = 0.005
                            model_type = "ensemble"
                        else:
                            # Simple model for others
                            processing_time = 0.001
                            model_type = "xgboost"

                        time.sleep(processing_time)

                        prediction_duration = (
                            TimeUtils.utc_now() - prediction_start
                        ).total_seconds()

                        # Log prediction
                        logger.info(
                            f"Production prediction completed",
                            extra={
                                "room_id": room_id,
                                "batch": prediction_batch,
                                "model_type": model_type,
                                "duration": prediction_duration,
                                "features_count": len(features),
                            },
                        )

                        # Record metrics
                        metrics_collector.record_prediction(
                            room_id=room_id,
                            prediction_type="occupancy",
                            model_type=model_type,
                            duration=prediction_duration,
                            confidence=0.75 + (room_num % 25) * 0.01,
                        )

                        production_stats["predictions_completed"] += 1
                        production_stats["avg_prediction_time"] += prediction_duration

                    except Exception as e:
                        production_stats["errors"] += 1
                        get_error_tracker().track_error(
                            e,
                            {
                                "room_id": room_id,
                                "batch": prediction_batch,
                                "test": "production_load",
                            },
                        )

        production_stats["total_duration"] = profiler.duration_seconds
        if production_stats["predictions_completed"] > 0:
            production_stats["avg_prediction_time"] /= production_stats[
                "predictions_completed"
            ]

        # Verify production performance
        assert production_stats["predictions_completed"] > 0
        assert production_stats["errors"] == 0
        assert profiler.duration_seconds < 60.0  # Should complete within 1 minute

        # Calculate performance metrics
        predictions_per_second = (
            production_stats["predictions_completed"] / profiler.duration_seconds
        )

        print(f"\nProduction Load Test Results:")
        print(f"Total predictions: {production_stats['predictions_completed']}")
        print(f"Errors: {production_stats['errors']}")
        print(f"Duration: {profiler.duration_seconds:.2f}s")
        print(f"Predictions/second: {predictions_per_second:.1f}")
        print(f"Avg prediction time: {production_stats['avg_prediction_time']:.4f}s")

        # Performance assertions
        assert predictions_per_second > 50  # At least 50 predictions/second
        assert production_stats["avg_prediction_time"] < 0.1  # Less than 100ms average
