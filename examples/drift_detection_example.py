"""
Example demonstration of the Concept Drift Detection System.

This example shows how to use the drift detection capabilities to monitor
occupancy patterns and detect when models need retraining.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adaptation.drift_detector import ConceptDriftDetector, FeatureDriftDetector
from src.adaptation.validator import PredictionValidator
from src.adaptation.tracker import AccuracyTracker


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demonstrate_concept_drift_detection():
    """Demonstrate comprehensive concept drift detection."""
    logger.info("Starting Concept Drift Detection Demonstration")

    # Initialize drift detector with custom parameters
    drift_detector = ConceptDriftDetector(
        baseline_days=30,  # 30 days of baseline data
        current_days=7,  # 7 days of current data
        min_samples=50,  # Minimum 50 samples for reliable detection
        alpha=0.05,  # 5% significance level
        ph_threshold=50.0,  # Page-Hinkley threshold
        psi_threshold=0.25,  # PSI threshold (>0.25 indicates drift)
    )

    # Initialize prediction validator for performance drift analysis
    prediction_validator = PredictionValidator(
        accuracy_threshold_minutes=15,
        max_records_in_memory=10000,
        cleanup_interval_hours=24,
    )

    try:
        # Start the prediction validator
        await prediction_validator.start_background_tasks()

        # Example room to analyze
        room_id = "living_room"

        logger.info(f"Analyzing drift for room: {room_id}")

        # Perform comprehensive drift detection
        drift_metrics = await drift_detector.detect_drift(
            room_id=room_id,
            prediction_validator=prediction_validator,
            feature_engineering_engine=None,  # Would integrate with feature engine
        )

        # Display drift analysis results
        print_drift_analysis(drift_metrics)

        # Check if retraining is recommended
        if drift_metrics.retraining_recommended:
            logger.warning(f"Retraining recommended for {room_id}")

            if drift_metrics.immediate_attention_required:
                logger.error(
                    f"Immediate attention required for {room_id} - critical drift detected!"
                )

        # Export drift metrics for further analysis
        drift_data = drift_metrics.to_dict()
        logger.info(
            f"Drift analysis completed. Overall score: {drift_metrics.overall_drift_score:.3f}"
        )

    except Exception as e:
        logger.error(f"Error in drift detection demonstration: {e}")
    finally:
        # Cleanup
        await prediction_validator.stop_background_tasks()


async def demonstrate_feature_drift_monitoring():
    """Demonstrate continuous feature drift monitoring."""
    logger.info("Starting Feature Drift Monitoring Demonstration")

    # Initialize feature drift detector
    feature_detector = FeatureDriftDetector(
        monitor_window_hours=168,  # 1 week monitoring window
        comparison_window_hours=336,  # 2 weeks comparison window
        min_samples_per_window=50,  # Minimum samples per window
        significance_threshold=0.05,  # 5% significance threshold
    )

    # Add drift notification callback
    async def drift_notification_callback(room_id: str, drift_result):
        """Handle drift detection notifications."""
        logger.warning(
            f"Feature drift detected in {room_id}: "
            f"feature={drift_result.feature_name}, "
            f"score={drift_result.drift_score:.3f}, "
            f"p_value={drift_result.p_value:.6f}"
        )

        # In production, this could trigger alerts or retraining
        if drift_result.drift_score > 0.5:
            logger.error(
                f"High drift score for {drift_result.feature_name} - immediate attention needed"
            )

    feature_detector.add_drift_callback(drift_notification_callback)

    try:
        # Start continuous monitoring for example rooms
        room_ids = ["living_room", "bedroom", "kitchen"]
        await feature_detector.start_monitoring(room_ids)

        logger.info(f"Started continuous feature drift monitoring for: {room_ids}")
        logger.info("Monitoring will check for drift every hour...")

        # Let it run for a short demo period
        await asyncio.sleep(5)  # In production, this would run continuously

    except Exception as e:
        logger.error(f"Error in feature drift monitoring: {e}")
    finally:
        # Stop monitoring
        await feature_detector.stop_monitoring()
        logger.info("Stopped feature drift monitoring")


def print_drift_analysis(drift_metrics):
    """Print detailed drift analysis results."""
    print("\n" + "=" * 80)
    print("DRIFT DETECTION ANALYSIS RESULTS")
    print("=" * 80)

    print(f"Room ID: {drift_metrics.room_id}")
    print(f"Detection Time: {drift_metrics.detection_time}")
    print(f"Overall Drift Score: {drift_metrics.overall_drift_score:.3f}")
    print(f"Drift Severity: {drift_metrics.drift_severity.value.upper()}")
    print(f"Statistical Confidence: {drift_metrics.statistical_confidence:.3f}")

    print(
        f"\nBaseline Period: {drift_metrics.baseline_period[0]} to {drift_metrics.baseline_period[1]}"
    )
    print(
        f"Current Period: {drift_metrics.current_period[0]} to {drift_metrics.current_period[1]}"
    )

    # Statistical test results
    print(f"\nSTATISTICAL TEST RESULTS:")
    print(
        f"Kolmogorov-Smirnov: statistic={drift_metrics.ks_statistic:.4f}, p={drift_metrics.ks_p_value:.6f}"
    )
    print(
        f"Mann-Whitney U: statistic={drift_metrics.mw_statistic:.4f}, p={drift_metrics.mw_p_value:.6f}"
    )
    print(f"Population Stability Index: {drift_metrics.psi_score:.4f}")
    print(
        f"Page-Hinkley: statistic={drift_metrics.ph_statistic:.2f}, threshold={drift_metrics.ph_threshold}"
    )

    # Drift types detected
    if drift_metrics.drift_types:
        print(f"\nDRIFT TYPES DETECTED:")
        for drift_type in drift_metrics.drift_types:
            print(f"  - {drift_type.value}")
    else:
        print(f"\nNo significant drift detected")

    # Feature drift analysis
    if drift_metrics.drifting_features:
        print(f"\nDRIFTING FEATURES:")
        for feature in drift_metrics.drifting_features:
            score = drift_metrics.feature_drift_scores.get(feature, 0)
            print(f"  - {feature}: drift_score={score:.3f}")

    # Performance analysis
    print(f"\nPREDICTION PERFORMANCE ANALYSIS:")
    print(f"Accuracy Degradation: {drift_metrics.accuracy_degradation:.1f} minutes")
    print(f"Error Distribution Change: {drift_metrics.error_distribution_change:.4f}")
    print(
        f"Confidence Calibration Drift: {drift_metrics.confidence_calibration_drift:.4f}"
    )

    # Pattern analysis
    print(f"\nPATTERN ANALYSIS:")
    print(f"Temporal Pattern Drift: {drift_metrics.temporal_pattern_drift:.4f}")
    print(f"Frequency Pattern Drift: {drift_metrics.frequency_pattern_drift:.4f}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(
        f"Retraining Recommended: {'YES' if drift_metrics.retraining_recommended else 'NO'}"
    )
    print(
        f"Immediate Attention Required: {'YES' if drift_metrics.immediate_attention_required else 'NO'}"
    )

    if drift_metrics.affected_models:
        print(f"Affected Models: {', '.join(drift_metrics.affected_models)}")

    print("=" * 80 + "\n")


async def demonstrate_integrated_monitoring():
    """Demonstrate integrated monitoring with accuracy tracking and drift detection."""
    logger.info("Starting Integrated Monitoring Demonstration")

    # Initialize all components
    prediction_validator = PredictionValidator()
    accuracy_tracker = AccuracyTracker(prediction_validator)
    drift_detector = ConceptDriftDetector()

    try:
        # Start background monitoring
        await prediction_validator.start_background_tasks()
        await accuracy_tracker.start_monitoring()

        # Example workflow: Check accuracy and trigger drift detection if needed
        room_id = "living_room"

        # Get current accuracy metrics
        real_time_metrics = await accuracy_tracker.get_real_time_metrics(
            room_id=room_id
        )

        if real_time_metrics and not real_time_metrics.is_healthy:
            logger.warning(
                f"Poor accuracy detected for {room_id} - triggering drift analysis"
            )

            # Perform drift detection due to poor accuracy
            drift_metrics = await drift_detector.detect_drift(
                room_id=room_id, prediction_validator=prediction_validator
            )

            # Check if drift explains the poor accuracy
            if drift_metrics.drift_severity.value in ["major", "critical"]:
                logger.error(
                    f"Significant drift detected - this explains the accuracy degradation"
                )

                # In production, this would trigger model retraining
                logger.info(f"Would trigger automatic retraining for {room_id}")
            else:
                logger.info(
                    f"No significant drift detected - accuracy issues may be due to other factors"
                )

        else:
            logger.info(
                f"Accuracy is healthy for {room_id} - no drift detection needed"
            )

    except Exception as e:
        logger.error(f"Error in integrated monitoring: {e}")
    finally:
        # Cleanup
        await accuracy_tracker.stop_monitoring()
        await prediction_validator.stop_background_tasks()


async def main():
    """Main demonstration function."""
    print("Concept Drift Detection System Demonstration")
    print("=" * 50)

    try:
        # Demonstrate concept drift detection
        await demonstrate_concept_drift_detection()

        print("\n" + "-" * 50 + "\n")

        # Demonstrate feature drift monitoring
        await demonstrate_feature_drift_monitoring()

        print("\n" + "-" * 50 + "\n")

        # Demonstrate integrated monitoring
        await demonstrate_integrated_monitoring()

    except Exception as e:
        logger.error(f"Error in main demonstration: {e}")

    print("\nDemonstration completed!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
