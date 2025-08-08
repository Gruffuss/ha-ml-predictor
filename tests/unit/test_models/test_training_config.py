"""
Comprehensive unit tests for training configuration management.

This module tests configuration profiles, validation, resource limits,
quality thresholds, optimization settings, and profile-based configuration
management.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import yaml

from src.models.training_config import OptimizationConfig
from src.models.training_config import OptimizationLevel
from src.models.training_config import QualityThresholds
from src.models.training_config import ResourceLimits
from src.models.training_config import TrainingConfigManager
from src.models.training_config import TrainingEnvironmentConfig
from src.models.training_config import TrainingProfile
from src.models.training_config import get_training_config
from src.models.training_config import get_training_config_manager
from src.models.training_pipeline import TrainingConfig
from src.models.training_pipeline import ValidationStrategy


class TestResourceLimits:
    """Test resource limits configuration and validation."""

    def test_resource_limits_initialization(self):
        """Test ResourceLimits initialization with default values."""
        limits = ResourceLimits()

        assert limits.max_memory_gb is None
        assert limits.max_cpu_cores is None
        assert limits.max_training_time_minutes == 60
        assert limits.max_parallel_models == 2
        assert limits.disk_space_gb is None

    def test_resource_limits_custom_values(self):
        """Test ResourceLimits initialization with custom values."""
        limits = ResourceLimits(
            max_memory_gb=16.0,
            max_cpu_cores=8,
            max_training_time_minutes=120,
            max_parallel_models=4,
            disk_space_gb=100.0,
        )

        assert limits.max_memory_gb == 16.0
        assert limits.max_cpu_cores == 8
        assert limits.max_training_time_minutes == 120
        assert limits.max_parallel_models == 4
        assert limits.disk_space_gb == 100.0

    def test_resource_limits_validation_success(self):
        """Test successful resource limits validation."""
        limits = ResourceLimits(
            max_memory_gb=8.0,
            max_cpu_cores=4,
            max_training_time_minutes=60,
            max_parallel_models=2,
        )

        issues = limits.validate()
        assert len(issues) == 0

    def test_resource_limits_validation_failures(self):
        """Test resource limits validation with invalid values."""
        limits = ResourceLimits(
            max_memory_gb=-1.0,  # Invalid: negative
            max_cpu_cores=0,  # Invalid: zero
            max_training_time_minutes=-30,  # Invalid: negative
            max_parallel_models=0,  # Invalid: zero
        )

        issues = limits.validate()
        assert len(issues) == 4
        assert "max_memory_gb must be positive" in issues
        assert "max_cpu_cores must be positive" in issues
        assert "max_training_time_minutes must be positive" in issues
        assert "max_parallel_models must be positive" in issues

    def test_resource_limits_partial_validation(self):
        """Test resource limits validation with some None values."""
        limits = ResourceLimits(
            max_memory_gb=None,  # Valid: can be None
            max_cpu_cores=None,  # Valid: can be None
            max_training_time_minutes=30,  # Valid: positive
            max_parallel_models=1,  # Valid: positive
        )

        issues = limits.validate()
        assert len(issues) == 0


class TestQualityThresholds:
    """Test quality thresholds configuration and validation."""

    def test_quality_thresholds_initialization(self):
        """Test QualityThresholds initialization with default values."""
        thresholds = QualityThresholds()

        assert thresholds.min_accuracy_threshold == 0.6
        assert thresholds.max_error_threshold_minutes == 30.0
        assert thresholds.min_confidence_calibration == 0.5
        assert thresholds.min_samples_per_room == 100
        assert thresholds.max_missing_data_percent == 20.0
        assert thresholds.min_data_freshness_hours == 24

    def test_quality_thresholds_custom_values(self):
        """Test QualityThresholds initialization with custom values."""
        thresholds = QualityThresholds(
            min_accuracy_threshold=0.8,
            max_error_threshold_minutes=15.0,
            min_confidence_calibration=0.7,
            min_samples_per_room=200,
            max_missing_data_percent=10.0,
            min_data_freshness_hours=12,
        )

        assert thresholds.min_accuracy_threshold == 0.8
        assert thresholds.max_error_threshold_minutes == 15.0
        assert thresholds.min_confidence_calibration == 0.7
        assert thresholds.min_samples_per_room == 200
        assert thresholds.max_missing_data_percent == 10.0
        assert thresholds.min_data_freshness_hours == 12

    def test_quality_thresholds_validation_success(self):
        """Test successful quality thresholds validation."""
        thresholds = QualityThresholds(
            min_accuracy_threshold=0.75,
            max_error_threshold_minutes=20.0,
            min_confidence_calibration=0.6,
            min_samples_per_room=150,
            max_missing_data_percent=15.0,
        )

        issues = thresholds.validate()
        assert len(issues) == 0

    def test_quality_thresholds_validation_failures(self):
        """Test quality thresholds validation with invalid values."""
        thresholds = QualityThresholds(
            min_accuracy_threshold=1.5,  # Invalid: > 1.0
            max_error_threshold_minutes=-10.0,  # Invalid: negative
            min_confidence_calibration=-0.1,  # Invalid: < 0.0
            min_samples_per_room=-50,  # Invalid: negative
            max_missing_data_percent=150.0,  # Invalid: > 100.0
        )

        issues = thresholds.validate()
        assert len(issues) == 5
        assert "min_accuracy_threshold must be between 0.0 and 1.0" in issues
        assert "max_error_threshold_minutes must be positive" in issues
        assert "min_confidence_calibration must be between 0.0 and 1.0" in issues
        assert "min_samples_per_room must be positive" in issues
        assert "max_missing_data_percent must be between 0.0 and 100.0" in issues

    def test_quality_thresholds_boundary_values(self):
        """Test quality thresholds validation with boundary values."""
        # Test valid boundary values
        thresholds = QualityThresholds(
            min_accuracy_threshold=0.0,  # Valid: minimum
            max_error_threshold_minutes=0.1,  # Valid: small positive
            min_confidence_calibration=1.0,  # Valid: maximum
            min_samples_per_room=1,  # Valid: minimum positive
            max_missing_data_percent=0.0,  # Valid: minimum
        )

        issues = thresholds.validate()
        assert len(issues) == 0

        # Test invalid boundary values
        thresholds_invalid = QualityThresholds(
            min_accuracy_threshold=-0.1,  # Invalid: just below 0
            min_confidence_calibration=1.1,  # Invalid: just above 1
            max_missing_data_percent=100.1,  # Invalid: just above 100
        )

        issues_invalid = thresholds_invalid.validate()
        assert len(issues_invalid) >= 3


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_optimization_config_initialization(self):
        """Test OptimizationConfig initialization with defaults."""
        opt_config = OptimizationConfig()

        assert opt_config.enabled is True
        assert opt_config.level == OptimizationLevel.STANDARD
        assert opt_config.max_optimization_time_minutes == 30
        assert opt_config.n_trials == 50
        assert opt_config.optimization_metric == "mae"
        assert opt_config.early_stopping_rounds == 10
        assert opt_config.parallel_trials == 2

        # Check search spaces are populated
        assert len(opt_config.ensemble_search_space) > 0
        assert len(opt_config.lstm_search_space) > 0
        assert len(opt_config.xgboost_search_space) > 0

        assert "meta_learner" in opt_config.ensemble_search_space
        assert "hidden_units" in opt_config.lstm_search_space
        assert "n_estimators" in opt_config.xgboost_search_space

    def test_optimization_config_custom_values(self):
        """Test OptimizationConfig with custom values."""
        custom_ensemble_space = {"meta_learner": ["random_forest"], "cv_folds": [5]}

        opt_config = OptimizationConfig(
            enabled=False,
            level=OptimizationLevel.INTENSIVE,
            max_optimization_time_minutes=60,
            n_trials=100,
            optimization_metric="r2",
            ensemble_search_space=custom_ensemble_space,
        )

        assert opt_config.enabled is False
        assert opt_config.level == OptimizationLevel.INTENSIVE
        assert opt_config.max_optimization_time_minutes == 60
        assert opt_config.n_trials == 100
        assert opt_config.optimization_metric == "r2"
        assert opt_config.ensemble_search_space == custom_ensemble_space

    def test_optimization_levels(self):
        """Test optimization level enum values."""
        assert OptimizationLevel.NONE.value == "none"
        assert OptimizationLevel.BASIC.value == "basic"
        assert OptimizationLevel.STANDARD.value == "standard"
        assert OptimizationLevel.INTENSIVE.value == "intensive"


class TestTrainingEnvironmentConfig:
    """Test training environment configuration."""

    def test_environment_config_initialization(self):
        """Test TrainingEnvironmentConfig initialization."""
        env_config = TrainingEnvironmentConfig()

        assert env_config.profile == TrainingProfile.PRODUCTION
        assert isinstance(env_config.resource_limits, ResourceLimits)
        assert isinstance(env_config.quality_thresholds, QualityThresholds)
        assert isinstance(env_config.optimization_config, OptimizationConfig)

        # Check boolean flags
        assert env_config.experiment_tracking_enabled is True
        assert env_config.model_versioning_enabled is True
        assert env_config.training_monitoring_enabled is True
        assert env_config.auto_register_with_tracking_manager is True
        assert env_config.auto_deploy_best_models is True
        assert env_config.enable_model_comparison is True

    def test_environment_config_custom_components(self):
        """Test TrainingEnvironmentConfig with custom components."""
        custom_limits = ResourceLimits(max_memory_gb=32.0, max_cpu_cores=16)
        custom_thresholds = QualityThresholds(min_accuracy_threshold=0.9)
        custom_optimization = OptimizationConfig(enabled=False)

        env_config = TrainingEnvironmentConfig(
            profile=TrainingProfile.RESEARCH,
            resource_limits=custom_limits,
            quality_thresholds=custom_thresholds,
            optimization_config=custom_optimization,
            experiment_tracking_enabled=False,
        )

        assert env_config.profile == TrainingProfile.RESEARCH
        assert env_config.resource_limits == custom_limits
        assert env_config.quality_thresholds == custom_thresholds
        assert env_config.optimization_config == custom_optimization
        assert env_config.experiment_tracking_enabled is False

    def test_environment_config_validation(self):
        """Test TrainingEnvironmentConfig validation."""
        # Valid configuration
        env_config = TrainingEnvironmentConfig()
        issues = env_config.validate()
        assert len(issues) == 0

        # Configuration with invalid nested components
        invalid_limits = ResourceLimits(max_memory_gb=-1.0)  # Invalid
        invalid_thresholds = QualityThresholds(min_accuracy_threshold=2.0)  # Invalid

        invalid_env_config = TrainingEnvironmentConfig(
            resource_limits=invalid_limits, quality_thresholds=invalid_thresholds
        )

        issues = invalid_env_config.validate()
        assert len(issues) >= 2  # Should have issues from both nested components

    def test_environment_config_path_validation(self):
        """Test path validation in TrainingEnvironmentConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "models"

            env_config = TrainingEnvironmentConfig(model_artifacts_base_path=valid_path)

            issues = env_config.validate()
            assert len(issues) == 0
            assert isinstance(env_config.model_artifacts_base_path, Path)

        # Test invalid path handling (string conversion)
        env_config_str = TrainingEnvironmentConfig(
            model_artifacts_base_path="/valid/path"
        )

        issues = env_config_str.validate()
        # Should convert string to Path without issues
        assert len(issues) == 0
        assert isinstance(env_config_str.model_artifacts_base_path, Path)


class TestTrainingProfiles:
    """Test training profile enum and values."""

    def test_training_profile_values(self):
        """Test TrainingProfile enum values."""
        assert TrainingProfile.DEVELOPMENT.value == "development"
        assert TrainingProfile.PRODUCTION.value == "production"
        assert TrainingProfile.TESTING.value == "testing"
        assert TrainingProfile.RESEARCH.value == "research"
        assert TrainingProfile.QUICK.value == "quick"
        assert TrainingProfile.COMPREHENSIVE.value == "comprehensive"

    def test_training_profile_iteration(self):
        """Test that all training profiles are accessible."""
        all_profiles = list(TrainingProfile)
        assert len(all_profiles) == 6

        profile_values = [p.value for p in all_profiles]
        expected_values = [
            "development",
            "production",
            "testing",
            "research",
            "quick",
            "comprehensive",
        ]
        assert set(profile_values) == set(expected_values)


class TestTrainingConfigManager:
    """Test training configuration manager."""

    def test_config_manager_initialization(self):
        """Test TrainingConfigManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "training_config.yaml"

            with patch("src.models.training_config.get_config") as mock_get_config:
                mock_get_config.return_value = MagicMock()

                config_manager = TrainingConfigManager(config_path)

                assert config_manager.config_path == config_path
                assert config_manager._current_profile == TrainingProfile.PRODUCTION

                # Should have all default profiles
                assert len(config_manager._environment_configs) == 6
                assert (
                    TrainingProfile.DEVELOPMENT.value
                    in config_manager._environment_configs
                )
                assert (
                    TrainingProfile.PRODUCTION.value
                    in config_manager._environment_configs
                )
                assert (
                    TrainingProfile.TESTING.value in config_manager._environment_configs
                )
                assert (
                    TrainingProfile.RESEARCH.value
                    in config_manager._environment_configs
                )
                assert (
                    TrainingProfile.QUICK.value in config_manager._environment_configs
                )
                assert (
                    TrainingProfile.COMPREHENSIVE.value
                    in config_manager._environment_configs
                )

    def test_default_profile_characteristics(self):
        """Test characteristics of default training profiles."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Test development profile
            dev_config = config_manager.get_environment_config(
                TrainingProfile.DEVELOPMENT
            )
            assert dev_config.profile == TrainingProfile.DEVELOPMENT
            assert dev_config.resource_limits.max_training_time_minutes == 15
            assert dev_config.optimization_config.enabled is False
            assert dev_config.quality_thresholds.min_accuracy_threshold == 0.4

            # Test production profile
            prod_config = config_manager.get_environment_config(
                TrainingProfile.PRODUCTION
            )
            assert prod_config.profile == TrainingProfile.PRODUCTION
            assert prod_config.resource_limits.max_training_time_minutes == 120
            assert prod_config.optimization_config.enabled is True
            assert prod_config.quality_thresholds.min_accuracy_threshold == 0.7

            # Test testing profile
            test_config = config_manager.get_environment_config(TrainingProfile.TESTING)
            assert test_config.profile == TrainingProfile.TESTING
            assert test_config.resource_limits.max_training_time_minutes == 5
            assert test_config.resource_limits.max_memory_gb == 2.0
            assert test_config.quality_thresholds.min_samples_per_room == 20

            # Test research profile
            research_config = config_manager.get_environment_config(
                TrainingProfile.RESEARCH
            )
            assert research_config.profile == TrainingProfile.RESEARCH
            assert research_config.resource_limits.max_training_time_minutes == 300
            assert (
                research_config.optimization_config.level == OptimizationLevel.INTENSIVE
            )
            assert research_config.quality_thresholds.min_accuracy_threshold == 0.8

    def test_profile_management(self):
        """Test setting and getting current profile."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Initial profile should be production
            assert config_manager.get_current_profile() == TrainingProfile.PRODUCTION

            # Set to development profile
            config_manager.set_current_profile(TrainingProfile.DEVELOPMENT)
            assert config_manager.get_current_profile() == TrainingProfile.DEVELOPMENT

            # Test invalid profile
            with pytest.raises(
                ValueError, match="Training profile invalid_profile not available"
            ):
                config_manager.set_current_profile(TrainingProfile("invalid_profile"))

    def test_training_config_generation(self):
        """Test training config generation from environment config."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Test development profile config generation
            training_config = config_manager.get_training_config(
                TrainingProfile.DEVELOPMENT
            )

            assert isinstance(training_config, TrainingConfig)
            assert training_config.lookback_days == 30  # Development lookback
            assert training_config.max_training_time_minutes == 15  # Development limit
            assert training_config.min_samples_per_room == 50  # Development threshold
            assert (
                training_config.enable_hyperparameter_optimization is False
            )  # Development setting
            assert (
                training_config.save_intermediate_results is True
            )  # Not testing profile

            # Test testing profile config generation
            testing_config = config_manager.get_training_config(TrainingProfile.TESTING)

            assert testing_config.lookback_days == 7  # Testing lookback
            assert testing_config.max_training_time_minutes == 5  # Testing limit
            assert testing_config.save_intermediate_results is False  # Testing profile

    def test_lookback_days_mapping(self):
        """Test lookback days mapping for different profiles."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Test all profile lookback days
            expected_lookback = {
                TrainingProfile.DEVELOPMENT: 30,
                TrainingProfile.PRODUCTION: 180,
                TrainingProfile.TESTING: 7,
                TrainingProfile.RESEARCH: 365,
                TrainingProfile.QUICK: 14,
                TrainingProfile.COMPREHENSIVE: 730,
            }

            for profile, expected_days in expected_lookback.items():
                actual_days = config_manager._get_lookback_days_for_profile(profile)
                assert actual_days == expected_days

    def test_configuration_validation(self):
        """Test configuration validation for profiles."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Valid profiles should have no validation issues
            for profile in TrainingProfile:
                issues = config_manager.validate_configuration(profile)
                assert (
                    len(issues) == 0
                ), f"Profile {profile.value} has validation issues: {issues}"

            # Test validation of current profile
            issues = config_manager.validate_configuration()
            assert len(issues) == 0

    def test_optimization_config_retrieval(self):
        """Test optimization configuration retrieval."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Test different profiles have different optimization configs
            dev_opt = config_manager.get_optimization_config(
                TrainingProfile.DEVELOPMENT
            )
            prod_opt = config_manager.get_optimization_config(
                TrainingProfile.PRODUCTION
            )
            research_opt = config_manager.get_optimization_config(
                TrainingProfile.RESEARCH
            )

            assert dev_opt.enabled is False
            assert prod_opt.enabled is True
            assert research_opt.enabled is True

            assert dev_opt.level == OptimizationLevel.NONE
            assert prod_opt.level == OptimizationLevel.STANDARD
            assert research_opt.level == OptimizationLevel.INTENSIVE

            assert (
                research_opt.max_optimization_time_minutes
                > prod_opt.max_optimization_time_minutes
            )

    def test_profile_updates(self):
        """Test updating profile configurations."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Get initial value
            initial_config = config_manager.get_environment_config(
                TrainingProfile.DEVELOPMENT
            )
            initial_monitoring = initial_config.training_monitoring_enabled

            # Update configuration
            config_manager.update_profile_config(
                TrainingProfile.DEVELOPMENT,
                training_monitoring_enabled=not initial_monitoring,
            )

            # Verify update
            updated_config = config_manager.get_environment_config(
                TrainingProfile.DEVELOPMENT
            )
            assert updated_config.training_monitoring_enabled != initial_monitoring

            # Test invalid profile update
            with pytest.raises(ValueError, match="Profile invalid_profile not found"):
                config_manager.update_profile_config(
                    TrainingProfile("invalid_profile"), training_monitoring_enabled=True
                )

    def test_profile_comparison(self):
        """Test profile comparison functionality."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            comparison = config_manager.get_profile_comparison()

            assert "profiles" in comparison
            assert "metrics" in comparison
            assert len(comparison["profiles"]) == 6  # All default profiles

            # Check that all profiles have required metrics
            required_metrics = [
                "max_training_time_minutes",
                "min_accuracy_threshold",
                "max_error_threshold_minutes",
                "optimization_enabled",
                "max_parallel_models",
            ]

            for profile_name, profile_data in comparison["profiles"].items():
                for metric in required_metrics:
                    assert metric in profile_data
                assert "description" in profile_data

    def test_use_case_recommendations(self):
        """Test use case based profile recommendations."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            config_manager = TrainingConfigManager()

            # Test use case recommendations
            test_cases = [
                ("development testing", TrainingProfile.DEVELOPMENT),
                ("unit testing", TrainingProfile.TESTING),
                ("research experiment", TrainingProfile.RESEARCH),
                ("quick iteration", TrainingProfile.QUICK),
                ("comprehensive analysis", TrainingProfile.COMPREHENSIVE),
                ("production deployment", TrainingProfile.PRODUCTION),
                ("unknown use case", TrainingProfile.PRODUCTION),  # Default
            ]

            for use_case, expected_profile in test_cases:
                recommended = config_manager.recommend_profile_for_use_case(use_case)
                assert recommended == expected_profile


class TestConfigurationFileSerialization:
    """Test configuration file loading and saving."""

    def test_config_file_saving(self):
        """Test saving configuration to YAML file."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "test_config.yaml"

                config_manager = TrainingConfigManager()
                config_manager.set_current_profile(TrainingProfile.DEVELOPMENT)

                # Save configuration
                config_manager.save_config_to_file(config_path)

                assert config_path.exists()

                # Verify file content
                with open(config_path, "r") as f:
                    saved_config = yaml.safe_load(f)

                assert "default_profile" in saved_config
                assert saved_config["default_profile"] == "development"
                assert "training_profiles" in saved_config
                assert len(saved_config["training_profiles"]) == 6

                # Check profile structure
                dev_profile = saved_config["training_profiles"]["development"]
                assert "profile" in dev_profile
                assert "resource_limits" in dev_profile
                assert "quality_thresholds" in dev_profile
                assert "optimization_config" in dev_profile

    def test_config_file_loading(self):
        """Test loading configuration from YAML file."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "test_config.yaml"

                # Create custom configuration file
                custom_config = {
                    "default_profile": "testing",
                    "training_profiles": {
                        "custom_profile": {
                            "profile": "development",
                            "resource_limits": {
                                "max_training_time_minutes": 45,
                                "max_parallel_models": 3,
                            },
                            "quality_thresholds": {
                                "min_accuracy_threshold": 0.65,
                                "max_error_threshold_minutes": 25.0,
                            },
                            "optimization_config": {"enabled": True, "level": "basic"},
                        }
                    },
                }

                with open(config_path, "w") as f:
                    yaml.dump(custom_config, f)

                # Load configuration
                config_manager = TrainingConfigManager(config_path)

                # Verify loading
                assert config_manager._current_profile == TrainingProfile.TESTING
                assert "custom_profile" in config_manager._environment_configs

                custom_env = config_manager._environment_configs["custom_profile"]
                assert custom_env.resource_limits.max_training_time_minutes == 45
                assert custom_env.resource_limits.max_parallel_models == 3
                assert custom_env.quality_thresholds.min_accuracy_threshold == 0.65
                assert custom_env.optimization_config.enabled is True
                assert custom_env.optimization_config.level == OptimizationLevel.BASIC

    def test_config_file_loading_error_handling(self):
        """Test error handling during config file loading."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "invalid_config.yaml"

                # Create invalid YAML file
                with open(config_path, "w") as f:
                    f.write("invalid: yaml: content: [unclosed")

                # Should not raise exception, but should log error
                config_manager = TrainingConfigManager(config_path)

                # Should still have default profiles despite invalid file
                assert len(config_manager._environment_configs) == 6
                assert config_manager._current_profile == TrainingProfile.PRODUCTION


class TestGlobalConfigManager:
    """Test global configuration manager functions."""

    def test_global_config_manager_singleton(self):
        """Test global configuration manager singleton behavior."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            # Clear global instance for test
            import src.models.training_config

            src.models.training_config._global_config_manager = None

            # Get manager instances
            manager1 = get_training_config_manager()
            manager2 = get_training_config_manager()

            # Should be the same instance
            assert manager1 is manager2
            assert isinstance(manager1, TrainingConfigManager)

    def test_get_training_config_convenience_function(self):
        """Test convenience function for getting training config."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            # Clear global instance for test
            import src.models.training_config

            src.models.training_config._global_config_manager = None

            # Get training config using convenience function
            config = get_training_config(TrainingProfile.DEVELOPMENT)

            assert isinstance(config, TrainingConfig)
            assert config.lookback_days == 30  # Development profile
            assert config.max_training_time_minutes == 15  # Development profile

            # Test with default profile (None)
            default_config = get_training_config(None)
            assert isinstance(default_config, TrainingConfig)
            # Should use current profile (production by default)
            assert default_config.lookback_days == 180  # Production profile


class TestConfigurationIntegration:
    """Test configuration integration with training pipeline."""

    def test_training_config_to_pipeline_integration(self):
        """Test that training config integrates properly with pipeline config."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            config_manager = TrainingConfigManager()

            # Test different profiles produce different configurations
            dev_config = config_manager.get_training_config(TrainingProfile.DEVELOPMENT)
            prod_config = config_manager.get_training_config(TrainingProfile.PRODUCTION)
            test_config = config_manager.get_training_config(TrainingProfile.TESTING)

            # Verify key differences between profiles
            assert (
                dev_config.max_training_time_minutes
                < prod_config.max_training_time_minutes
            )
            assert test_config.min_samples_per_room < dev_config.min_samples_per_room
            assert not dev_config.enable_hyperparameter_optimization
            assert prod_config.enable_hyperparameter_optimization

            # Verify common settings
            for config in [dev_config, prod_config, test_config]:
                assert config.validation_split == 0.2
                assert config.test_split == 0.1
                assert config.include_temporal_features is True
                assert config.include_sequential_features is True
                assert config.include_contextual_features is True
                assert config.ensemble_enabled is True
                assert config.cv_folds == 5
                assert (
                    config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT
                )

    def test_profile_resource_mapping(self):
        """Test that profile resource limits map correctly to training config."""
        with patch("src.models.training_config.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            config_manager = TrainingConfigManager()

            # Test resource mapping for production profile
            prod_env = config_manager.get_environment_config(TrainingProfile.PRODUCTION)
            prod_training = config_manager.get_training_config(
                TrainingProfile.PRODUCTION
            )

            assert (
                prod_training.max_training_time_minutes
                == prod_env.resource_limits.max_training_time_minutes
            )
            assert (
                prod_training.max_parallel_models
                == prod_env.resource_limits.max_parallel_models
            )
            assert (
                prod_training.memory_limit_gb == prod_env.resource_limits.max_memory_gb
            )
            assert prod_training.cpu_cores == prod_env.resource_limits.max_cpu_cores

            # Test quality threshold mapping
            assert (
                prod_training.min_accuracy_threshold
                == prod_env.quality_thresholds.min_accuracy_threshold
            )
            assert (
                prod_training.max_error_threshold_minutes
                == prod_env.quality_thresholds.max_error_threshold_minutes
            )
            assert (
                prod_training.min_samples_per_room
                == prod_env.quality_thresholds.min_samples_per_room
            )

            # Test optimization mapping
            assert (
                prod_training.enable_hyperparameter_optimization
                == prod_env.optimization_config.enabled
            )
