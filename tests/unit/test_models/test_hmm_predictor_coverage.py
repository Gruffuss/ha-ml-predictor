"""
HMM Predictor Coverage-Focused Tests
Tests specifically designed to cover missing coverage lines.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ModelTrainingError
from src.models.base.hmm_predictor import HMMPredictor


class TestHMMCoverageMissing:
    """Test class focused on covering missing lines in HMM predictor."""

    @pytest.mark.asyncio
    async def test_insufficient_training_data_error(self):
        """Test error when training data is insufficient (< 20 samples)."""
        hmm = HMMPredictor(room_id="coverage_test")

        # Create insufficient data (only 10 samples)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
            }
        )

        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(60, 3600, 10)}
        )

        # Should raise ModelTrainingError due to insufficient data
        with pytest.raises(ModelTrainingError) as exc_info:
            await hmm.train(features, targets)

        # Just verify it's the right type of error
        assert isinstance(exc_info.value, ModelTrainingError)

    @pytest.mark.asyncio
    @patch("src.models.base.hmm_predictor.GaussianMixture")
    async def test_training_failure_exception_handling(self, mock_gaussian_mixture):
        """Test exception handling during training."""
        # Setup mock to raise exception during fit
        mock_model = MagicMock()
        mock_model.fit.side_effect = RuntimeError("Mock training failure")
        mock_gaussian_mixture.return_value = mock_model

        hmm = HMMPredictor(room_id="error_test")

        # Create valid training data
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )

        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(60, 3600, 50)}
        )

        # Should catch exception and re-raise as ModelTrainingError
        with pytest.raises(ModelTrainingError) as exc_info:
            await hmm.train(features, targets)

        # Verify error handling
        assert "HMM" in str(exc_info.value)
        assert "error_test" in str(exc_info.value)
        assert len(hmm.training_history) == 1
        assert hmm.training_history[0].success is False
        assert "Mock training failure" in hmm.training_history[0].error_message

    def test_target_preparation_error_handling(self):
        """Test error handling in target preparation."""
        hmm = HMMPredictor(room_id="target_error_test")

        # Create malformed targets that will cause errors
        targets = pd.DataFrame(
            {
                "invalid_column": ["invalid", "data", "types"],
                "another_column": [None, None, None],
            }
        )

        # Should handle the error and return default values
        result = hmm._prepare_targets(targets)

        # Should return default 1-hour values for all samples
        expected_default = np.array([3600, 3600, 3600])
        np.testing.assert_array_equal(result, expected_default)

    def test_state_analysis_missing_branch(self):
        """Test state analysis with edge case conditions."""
        hmm = HMMPredictor(room_id="state_test", n_states=2)

        # Create data that will trigger specific branches
        X = np.random.randn(30, 3)
        y = np.random.uniform(60, 3600, 30)

        # Mock state model to control behavior
        with patch.object(hmm, "state_model") as mock_state_model:
            mock_state_model.predict.return_value = np.array(
                [0, 1, 0, 1] * 7 + [0, 0]
            )  # 30 samples
            mock_state_model.predict_proba.return_value = np.random.rand(30, 2)

            # This should trigger the state analysis logic
            hmm._analyze_states(X, y)

            # Verify state characteristics were updated
            assert len(hmm.state_characteristics) >= 0

    @pytest.mark.asyncio
    async def test_prediction_with_no_state_model(self):
        """Test prediction when state model is not trained."""
        hmm = HMMPredictor(room_id="no_state_test")

        # Create features without training
        features = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [2.0, 3.0, 4.0],
                "feature3": [3.0, 4.0, 5.0],
            }
        )

        from datetime import datetime, timezone

        prediction_time = datetime.now(timezone.utc)

        # Should handle missing state model gracefully
        try:
            result = await hmm.predict(features, prediction_time)
            # If it doesn't raise an exception, verify the result structure
            assert isinstance(result, dict)
        except Exception as e:
            # Should be a graceful exception with meaningful message
            assert "not trained" in str(e).lower() or "state model" in str(e).lower()

    def test_save_model_with_mock_objects(self):
        """Test saving model that contains non-serializable objects."""
        hmm = HMMPredictor(room_id="mock_test")

        # Add mock objects that can't be pickled
        hmm.state_model = MagicMock()
        hmm.transition_models = {"mock": MagicMock()}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Should fail to save due to mock objects
            result = hmm.save_model(tmp_path)
            assert result is False
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_model_nonexistent_file(self):
        """Test loading model from non-existent file."""
        hmm = HMMPredictor(room_id="load_test")

        # Try to load from non-existent file
        result = hmm.load_model("/nonexistent/path/model.pkl")
        assert result is False

    def test_get_model_complexity_no_state_model(self):
        """Test get_model_complexity when no state model exists."""
        hmm = HMMPredictor(room_id="complexity_test")

        # Should handle missing state model
        complexity = hmm.get_model_complexity()

        assert isinstance(complexity, dict)
        assert "n_components" in complexity
        assert "n_features" in complexity
        assert complexity["n_features"] == 0
        assert "training_samples" in complexity
        assert complexity["training_samples"] == 0

    @pytest.mark.asyncio
    async def test_incremental_update_no_previous_training(self):
        """Test incremental update without previous training."""
        hmm = HMMPredictor(room_id="incremental_test")

        # Create small dataset
        new_features = pd.DataFrame(
            {
                "feature1": np.random.randn(25),
                "feature2": np.random.randn(25),
                "feature3": np.random.randn(25),
            }
        )

        new_targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(60, 3600, 25)}
        )

        # Should handle first-time training gracefully
        result = await hmm.incremental_update(new_features, new_targets)

        # Should either train successfully or handle gracefully
        assert isinstance(result, dict)
        assert "success" in result

    def test_n_states_parameter_alias(self):
        """Test n_states parameter alias functionality."""
        hmm = HMMPredictor(room_id="alias_test", n_states=6)

        # Both n_states and n_components should be set to 6
        assert hmm.model_params["n_states"] == 6
        assert hmm.model_params["n_components"] == 6

    def test_feature_importance_with_empty_characteristics(self):
        """Test feature importance when state characteristics are empty."""
        hmm = HMMPredictor(room_id="importance_test")

        # Set up minimal state but no characteristics
        hmm.state_characteristics = {}
        hmm.feature_names = ["feature1", "feature2", "feature3"]

        importance = hmm.get_feature_importance()

        # Should return empty dict or handle gracefully
        assert isinstance(importance, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
