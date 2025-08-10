#!/usr/bin/env python3
"""
Sprint 3 Validation Script

Simple validation script to test Sprint 3 model components without external dependencies.
This verifies that all the model architecture is properly implemented.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all Sprint 3 components can be imported."""
    print("Testing Sprint 3 imports...")

    try:
        # Core model components
        from models.base.hmm_predictor import HMMPredictor
        from models.base.lstm_predictor import LSTMPredictor
        from models.base.predictor import (
            BasePredictor,
            PredictionResult,
            TrainingResult,
        )
        from models.base.xgboost_predictor import XGBoostPredictor
        from models.ensemble import OccupancyEnsemble

        print("[PASS] All Sprint 3 imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_structure():
    """Test basic model structure and initialization."""
    print("\nTesting basic model structure...")

    try:
        from core.constants import ModelType
        from models.base.hmm_predictor import HMMPredictor
        from models.base.lstm_predictor import LSTMPredictor
        from models.base.xgboost_predictor import XGBoostPredictor
        from models.ensemble import OccupancyEnsemble

        # Test model initialization
        lstm = LSTMPredictor(room_id="test_room")
        xgb = XGBoostPredictor(room_id="test_room")
        hmm = HMMPredictor(room_id="test_room")
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test basic attributes
        assert lstm.model_type == ModelType.LSTM
        assert xgb.model_type == ModelType.XGBOOST
        assert hmm.model_type == ModelType.HMM
        assert ensemble.model_type == ModelType.ENSEMBLE

        assert all(pred.room_id == "test_room" for pred in [lstm, xgb, hmm, ensemble])
        assert all(pred.is_trained == False for pred in [lstm, xgb, hmm, ensemble])

        print("[PASS] Basic model structure test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Structure test failed: {e}")
        traceback.print_exc()
        return False


def test_prediction_result():
    """Test PredictionResult dataclass."""
    print("\nTesting PredictionResult structure...")

    try:
        from models.base.predictor import PredictionResult

        # Test basic prediction result
        pred_time = datetime.utcnow() + timedelta(minutes=30)
        result = PredictionResult(
            predicted_time=pred_time,
            transition_type="vacant_to_occupied",
            confidence_score=0.85,
        )

        assert result.predicted_time == pred_time
        assert result.transition_type == "vacant_to_occupied"
        assert result.confidence_score == 0.85

        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "predicted_time" in result_dict
        assert "confidence_score" in result_dict

        print("[PASS] PredictionResult test passed")
        return True
    except Exception as e:
        print(f"[FAIL] PredictionResult test failed: {e}")
        traceback.print_exc()
        return False


def test_training_result():
    """Test TrainingResult dataclass."""
    print("\nTesting TrainingResult structure...")

    try:
        from models.base.predictor import TrainingResult

        # Test training result
        result = TrainingResult(
            success=True,
            training_time_seconds=120.5,
            model_version="v1.0",
            training_samples=1000,
            validation_score=0.85,
        )

        assert result.success == True
        assert result.training_time_seconds == 120.5
        assert result.model_version == "v1.0"
        assert result.training_samples == 1000
        assert result.validation_score == 0.85

        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "training_time_seconds" in result_dict

        print("[PASS] TrainingResult test passed")
        return True
    except Exception as e:
        print(f"[FAIL] TrainingResult test failed: {e}")
        traceback.print_exc()
        return False


def test_model_methods():
    """Test that all required methods exist on models."""
    print("\nTesting model method interfaces...")

    try:
        from models.base.xgboost_predictor import XGBoostPredictor

        predictor = XGBoostPredictor(room_id="test_room")

        # Test required methods exist
        assert hasattr(predictor, "train")
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "get_model_info")
        assert hasattr(predictor, "save_model")
        assert hasattr(predictor, "load_model")
        assert hasattr(predictor, "get_feature_importance")
        assert hasattr(predictor, "validate_features")

        # Test model info
        info = predictor.get_model_info()
        assert isinstance(info, dict)
        assert info["room_id"] == "test_room"
        assert info["is_trained"] == False

        print("[PASS] Model method interfaces test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Model methods test failed: {e}")
        traceback.print_exc()
        return False


def test_ensemble_structure():
    """Test ensemble-specific functionality."""
    print("\nTesting ensemble structure...")

    try:
        from models.ensemble import OccupancyEnsemble

        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test ensemble-specific attributes
        assert hasattr(ensemble, "base_models")
        assert hasattr(ensemble, "model_weights")
        assert hasattr(ensemble, "base_models_trained")
        assert hasattr(ensemble, "meta_learner_trained")

        # Test base models are initialized
        assert "lstm" in ensemble.base_models
        assert "xgboost" in ensemble.base_models
        assert "hmm" in ensemble.base_models

        # Test ensemble methods
        assert hasattr(ensemble, "get_ensemble_info")
        info = ensemble.get_ensemble_info()
        assert isinstance(info, dict)
        assert "base_models" in info
        assert "ensemble_type" in info

        print("[PASS] Ensemble structure test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Ensemble structure test failed: {e}")
        traceback.print_exc()
        return False


def test_model_constants():
    """Test that model constants are properly defined."""
    print("\nTesting model constants...")

    try:
        from core.constants import ModelType, PredictionType

        # Test ModelType enum
        assert hasattr(ModelType, "LSTM")
        assert hasattr(ModelType, "XGBOOST")
        assert hasattr(ModelType, "HMM")
        assert hasattr(ModelType, "ENSEMBLE")

        # Test PredictionType enum (use existing values)
        assert hasattr(PredictionType, "NEXT_OCCUPIED")
        assert hasattr(PredictionType, "NEXT_VACANT")

        print("[PASS] Model constants test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Model constants test failed: {e}")
        traceback.print_exc()
        return False


def test_model_serialization():
    """Test model serialization capabilities."""
    print("\nTesting model serialization...")

    try:
        import os
        import tempfile

        from models.base.xgboost_predictor import XGBoostPredictor

        predictor = XGBoostPredictor(room_id="test_room")

        # Test save without training (should work)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.pkl")
            success = predictor.save_model(save_path)
            assert success, "Save operation should succeed"

            # Test load
            new_predictor = XGBoostPredictor(room_id="different_room")
            load_success = new_predictor.load_model(save_path)
            assert load_success, "Load operation should succeed"
            assert new_predictor.room_id == "test_room", "Room ID should be restored"

        print("[PASS] Model serialization test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Model serialization test failed: {e}")
        traceback.print_exc()
        return False


def test_mock_training_interface():
    """Test training interface without actual ML dependencies."""
    print("\nTesting mock training interface...")

    try:
        from models.base.xgboost_predictor import XGBoostPredictor

        predictor = XGBoostPredictor(room_id="test_room")

        # Create mock features and targets (simple dict structures)
        class MockDataFrame:
            def __init__(self, data=None):
                self.data = data or {}
                self.columns = list(self.data.keys()) if self.data else []
                self.shape = (1, len(self.columns))

            def __len__(self):
                return 1

        mock_features = MockDataFrame(
            {"time_since_last_change": [300], "hour_sin": [0.5], "hour_cos": [0.866]}
        )

        mock_targets = MockDataFrame({"next_transition_minutes": [45]})

        # Test that training method exists and can be called
        # (will fail due to missing ML libs, but interface should exist)
        assert hasattr(predictor, "train")
        assert callable(predictor.train)

        print("[PASS] Mock training interface test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Mock training interface test failed: {e}")
        traceback.print_exc()
        return False


def test_ensemble_base_models():
    """Test that ensemble properly initializes base models."""
    print("\nTesting ensemble base model initialization...")

    try:
        from models.ensemble import OccupancyEnsemble

        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test that all base models are present
        expected_models = ["lstm", "xgboost", "hmm"]
        for model_name in expected_models:
            assert (
                model_name in ensemble.base_models
            ), f"Missing base model: {model_name}"
            base_model = ensemble.base_models[model_name]
            assert (
                base_model.room_id == "test_room"
            ), f"Base model {model_name} has wrong room_id"
            assert (
                not base_model.is_trained
            ), f"Base model {model_name} should not be trained initially"

        # Test ensemble-specific methods
        assert hasattr(ensemble, "get_ensemble_info")
        assert hasattr(ensemble, "_combine_predictions")

        print("[PASS] Ensemble base models test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Ensemble base models test failed: {e}")
        traceback.print_exc()
        return False


def test_feature_validation_interface():
    """Test feature validation interface."""
    print("\nTesting feature validation interface...")

    try:
        from models.base.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor(room_id="test_room")

        # Create mock DataFrame
        class MockDataFrame:
            def __init__(self, columns):
                self.columns = columns

        # Test with untrained model (should return False with warning)
        mock_features = MockDataFrame(["feature1", "feature2"])
        result = predictor.validate_features(mock_features)
        assert result == False, "Untrained model should fail feature validation"

        # Test that method exists
        assert hasattr(predictor, "validate_features")
        assert callable(predictor.validate_features)

        print("[PASS] Feature validation interface test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Feature validation interface test failed: {e}")
        traceback.print_exc()
        return False


def test_model_info_completeness():
    """Test that model info contains all required fields."""
    print("\nTesting model info completeness...")

    try:
        from models.base.hmm_predictor import HMMPredictor

        predictor = HMMPredictor(room_id="test_room")
        info = predictor.get_model_info()

        # Required fields in model info
        required_fields = [
            "model_type",
            "room_id",
            "model_version",
            "is_trained",
            "training_date",
            "feature_count",
            "feature_names",
            "model_params",
            "training_sessions",
            "predictions_made",
        ]

        for field in required_fields:
            assert field in info, f"Missing field in model info: {field}"

        # Test specific values
        assert info["room_id"] == "test_room"
        assert info["is_trained"] == False
        assert info["training_sessions"] == 0
        assert info["predictions_made"] == 0

        print("[PASS] Model info completeness test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Model info completeness test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")

    try:
        base_path = Path(__file__).parent

        # Required model files
        required_files = [
            "src/models/__init__.py",
            "src/models/base/__init__.py",
            "src/models/base/predictor.py",
            "src/models/base/lstm_predictor.py",
            "src/models/base/xgboost_predictor.py",
            "src/models/base/hmm_predictor.py",
            "src/models/ensemble.py",
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Missing file: {file_path}"

        print("[PASS] File structure test passed")
        return True
    except Exception as e:
        print(f"[FAIL] File structure test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Sprint 3 validation tests."""
    print("=" * 60)
    print("Sprint 3 Model Development Validation")
    print("=" * 60)
    print("Validating Sprint 3 components:")
    print(
        "  - Base predictor interface (BasePredictor, PredictionResult, TrainingResult)"
    )
    print("  - LSTM predictor implementation")
    print("  - XGBoost predictor implementation")
    print("  - HMM predictor implementation")
    print("  - Ensemble architecture with meta-learning")
    print("  - Model serialization (save/load)")
    print("  - Feature validation interfaces")
    print("  - Model constants and enums")
    print("  - File structure completeness")
    print("-" * 60)

    tests = [
        test_imports,
        test_basic_structure,
        test_prediction_result,
        test_training_result,
        test_model_methods,
        test_ensemble_structure,
        test_model_constants,
        test_model_serialization,
        test_mock_training_interface,
        test_ensemble_base_models,
        test_feature_validation_interface,
        test_model_info_completeness,
        test_file_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Sprint 3 Validation Results:")
    print(f"   [PASS] Passed: {passed}")
    print(f"   [FAIL] Failed: {failed}")
    print(f"   Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n[SUCCESS] All Sprint 3 validation tests PASSED!")
        print("   Ready to proceed with Sprint 4: Integration & Deployment")
        return True
    else:
        print(f"\n[ERROR] {failed} Sprint 3 tests FAILED!")
        print("   Fix failing tests before proceeding to Sprint 4")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
