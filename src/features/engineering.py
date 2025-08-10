"""
Feature Engineering Engine for occupancy prediction.

This module provides a unified interface for extracting comprehensive features
from sensor data, coordinating temporal, sequential, and contextual extractors.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from ..core.config import RoomConfig, SystemConfig, get_config
from ..core.exceptions import ConfigurationError, FeatureExtractionError
from ..data.storage.models import RoomState, SensorEvent
from .contextual import ContextualFeatureExtractor
from .sequential import SequentialFeatureExtractor
from .temporal import TemporalFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureEngineeringEngine:
    """
    Unified feature engineering engine that coordinates all feature extractors.

    This engine orchestrates temporal, sequential, and contextual feature extraction
    to provide comprehensive feature sets for machine learning models.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        enable_parallel: bool = True,
        max_workers: int = 3,
    ):
        """
        Initialize the feature engineering engine.

        Args:
            config: System configuration
            enable_parallel: Whether to enable parallel feature extraction
            max_workers: Maximum number of parallel workers
        """
        self.config = config or get_config()
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        # Initialize feature extractors
        self.temporal_extractor = TemporalFeatureExtractor()
        self.sequential_extractor = SequentialFeatureExtractor(self.config)
        self.contextual_extractor = ContextualFeatureExtractor(self.config)

        # Feature extraction statistics
        self.stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_extraction_time": 0.0,
            "feature_counts": {
                "temporal": 0,
                "sequential": 0,
                "contextual": 0,
            },
        }

        # Thread pool for parallel processing
        if self.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None

    async def extract_features(
        self,
        room_id: str,
        target_time: datetime,
        events: Optional[List[SensorEvent]] = None,
        room_states: Optional[List[RoomState]] = None,
        lookback_hours: int = 24,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Extract comprehensive features for a specific room and time.

        Args:
            room_id: Room identifier
            target_time: Time for which to extract features
            events: Pre-filtered sensor events (if None, will query database)
            room_states: Pre-filtered room states (if None, will query database)
            lookback_hours: How far back to look for patterns
            feature_types: Which feature types to extract ('temporal', 'sequential', 'contextual')

        Returns:
            Dictionary containing all extracted features

        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        start_time = datetime.utcnow()
        self.stats["total_extractions"] += 1

        try:
            # Validate inputs
            if not room_id:
                raise FeatureExtractionError("Room ID is required")

            # Get room configuration
            room_config = self.config.rooms.get(room_id)
            if not room_config:
                logger.warning(f"No configuration found for room {room_id}")
                room_config = None

            # Determine which feature types to extract
            if feature_types is None:
                feature_types = ["temporal", "sequential", "contextual"]

            # If events or room_states are None, we would query the database here
            # For now, we'll work with the provided data
            if events is None:
                events = []
            if room_states is None:
                room_states = []

            # Filter events for the specific room and time window
            cutoff_time = target_time - timedelta(hours=lookback_hours)
            room_events = [
                e for e in events if e.room_id == room_id and e.timestamp >= cutoff_time
            ]
            room_room_states = [
                r
                for r in room_states
                if r.room_id == room_id and r.timestamp >= cutoff_time
            ]

            # Extract features in parallel or sequentially
            if self.enable_parallel and self.executor:
                features = await self._extract_features_parallel(
                    room_events,
                    room_room_states,
                    target_time,
                    room_config,
                    feature_types,
                    lookback_hours,
                )
            else:
                features = await self._extract_features_sequential(
                    room_events,
                    room_room_states,
                    target_time,
                    room_config,
                    feature_types,
                    lookback_hours,
                )

            # Add metadata features
            features.update(
                self._add_metadata_features(
                    room_id,
                    target_time,
                    len(room_events),
                    len(room_room_states),
                )
            )

            # Update statistics
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats["successful_extractions"] += 1
            self.stats["avg_extraction_time"] = (
                self.stats["avg_extraction_time"]
                * (self.stats["successful_extractions"] - 1)
                + extraction_time
            ) / self.stats["successful_extractions"]

            logger.debug(
                f"Extracted {len(features)} features for room {room_id} in {extraction_time:.3f}s"
            )
            return features

        except Exception as e:
            self.stats["failed_extractions"] += 1
            logger.error(f"Feature extraction failed for room {room_id}: {e}")
            raise FeatureExtractionError(
                f"Failed to extract features for room {room_id}: {e}"
            )

    async def extract_batch_features(
        self,
        extraction_requests: List[
            Tuple[str, datetime, List[SensorEvent], List[RoomState]]
        ],
        lookback_hours: int = 24,
        feature_types: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Extract features for multiple rooms and times in batch.

        Args:
            extraction_requests: List of (room_id, target_time, events, room_states) tuples
            lookback_hours: How far back to look for patterns
            feature_types: Which feature types to extract

        Returns:
            List of feature dictionaries corresponding to each request
        """
        results = []

        # Process requests in parallel if enabled
        if self.enable_parallel and len(extraction_requests) > 1:
            tasks = []
            for (
                room_id,
                target_time,
                events,
                room_states,
            ) in extraction_requests:
                task = self.extract_features(
                    room_id,
                    target_time,
                    events,
                    room_states,
                    lookback_hours,
                    feature_types,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction failed for request {i}: {result}")
                    results[i] = self._get_default_features()
        else:
            # Process sequentially
            for (
                room_id,
                target_time,
                events,
                room_states,
            ) in extraction_requests:
                try:
                    features = await self.extract_features(
                        room_id,
                        target_time,
                        events,
                        room_states,
                        lookback_hours,
                        feature_types,
                    )
                    results.append(features)
                except Exception as e:
                    logger.error(f"Batch extraction failed for room {room_id}: {e}")
                    results.append(self._get_default_features())

        return results

    async def _extract_features_parallel(
        self,
        events: List[SensorEvent],
        room_states: List[RoomState],
        target_time: datetime,
        room_config: Optional[RoomConfig],
        feature_types: List[str],
        lookback_hours: int,
    ) -> Dict[str, float]:
        """Extract features using parallel processing."""
        loop = asyncio.get_event_loop()

        # Create extraction tasks
        tasks = []

        if "temporal" in feature_types:
            task = loop.run_in_executor(
                self.executor,
                self.temporal_extractor.extract_features,
                events,
                target_time,
                room_states,
            )
            tasks.append(("temporal", task))

        if "sequential" in feature_types:
            room_configs = {room_config.room_id: room_config} if room_config else {}
            task = loop.run_in_executor(
                self.executor,
                self.sequential_extractor.extract_features,
                events,
                target_time,
                room_configs,
                lookback_hours,
            )
            tasks.append(("sequential", task))

        if "contextual" in feature_types:
            room_configs = {room_config.room_id: room_config} if room_config else {}
            task = loop.run_in_executor(
                self.executor,
                self.contextual_extractor.extract_features,
                events,
                room_states,
                target_time,
                room_configs,
                lookback_hours,
            )
            tasks.append(("contextual", task))

        # Wait for all tasks to complete
        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Combine results
        combined_features = {}
        for i, (feature_type, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to extract {feature_type} features: {result}")
                result = {}

            # Add prefix to feature names to avoid conflicts
            prefixed_features = {f"{feature_type}_{k}": v for k, v in result.items()}
            combined_features.update(prefixed_features)

            self.stats["feature_counts"][feature_type] += len(result)

        return combined_features

    async def _extract_features_sequential(
        self,
        events: List[SensorEvent],
        room_states: List[RoomState],
        target_time: datetime,
        room_config: Optional[RoomConfig],
        feature_types: List[str],
        lookback_hours: int,
    ) -> Dict[str, float]:
        """Extract features using sequential processing."""
        combined_features = {}

        try:
            if "temporal" in feature_types:
                temporal_features = self.temporal_extractor.extract_features(
                    events, target_time, room_states
                )
                prefixed_temporal = {
                    f"temporal_{k}": v for k, v in temporal_features.items()
                }
                combined_features.update(prefixed_temporal)
                self.stats["feature_counts"]["temporal"] += len(temporal_features)
        except Exception as e:
            logger.error(f"Failed to extract temporal features: {e}")

        try:
            if "sequential" in feature_types:
                room_configs = {room_config.room_id: room_config} if room_config else {}
                sequential_features = self.sequential_extractor.extract_features(
                    events, target_time, room_configs, lookback_hours
                )
                prefixed_sequential = {
                    f"sequential_{k}": v for k, v in sequential_features.items()
                }
                combined_features.update(prefixed_sequential)
                self.stats["feature_counts"]["sequential"] += len(sequential_features)
        except Exception as e:
            logger.error(f"Failed to extract sequential features: {e}")

        try:
            if "contextual" in feature_types:
                room_configs = {room_config.room_id: room_config} if room_config else {}
                contextual_features = self.contextual_extractor.extract_features(
                    events,
                    room_states,
                    target_time,
                    room_configs,
                    lookback_hours,
                )
                prefixed_contextual = {
                    f"contextual_{k}": v for k, v in contextual_features.items()
                }
                combined_features.update(prefixed_contextual)
                self.stats["feature_counts"]["contextual"] += len(contextual_features)
        except Exception as e:
            logger.error(f"Failed to extract contextual features: {e}")

        return combined_features

    def _add_metadata_features(
        self,
        room_id: str,
        target_time: datetime,
        event_count: int,
        room_state_count: int,
    ) -> Dict[str, float]:
        """Add metadata features about the extraction context."""
        # Use numpy for efficient normalization and feature processing
        feature_values = np.array(
            [event_count, room_state_count, target_time.hour, target_time.weekday()],
            dtype=np.float64,
        )

        # Normalize using numpy operations
        normalized_values = np.clip(
            feature_values / np.array([100.0, 50.0, 24.0, 7.0]), 0.0, 1.0
        )

        return {
            "meta_event_count": float(
                normalized_values[0] * 100.0
            ),  # Denormalize for interpretability
            "meta_room_state_count": float(normalized_values[1] * 50.0),
            "meta_extraction_hour": float(feature_values[2]),
            "meta_extraction_day_of_week": float(feature_values[3]),
            "meta_data_quality_score": float(
                normalized_values[0]
            ),  # Keep normalized for quality score
            "meta_feature_vector_norm": float(
                np.linalg.norm(normalized_values)
            ),  # Add vector norm
        }

    def get_feature_names(self, feature_types: Optional[List[str]] = None) -> List[str]:
        """
        Get list of all feature names that will be extracted.

        Args:
            feature_types: Which feature types to include

        Returns:
            List of feature names with prefixes
        """
        if feature_types is None:
            feature_types = ["temporal", "sequential", "contextual"]

        feature_names = []

        if "temporal" in feature_types:
            temporal_names = self.temporal_extractor.get_feature_names()
            feature_names.extend([f"temporal_{name}" for name in temporal_names])

        if "sequential" in feature_types:
            sequential_names = self.sequential_extractor.get_feature_names()
            feature_names.extend([f"sequential_{name}" for name in sequential_names])

        if "contextual" in feature_types:
            contextual_names = self.contextual_extractor.get_feature_names()
            feature_names.extend([f"contextual_{name}" for name in contextual_names])

        # Add metadata features
        feature_names.extend(
            [
                "meta_event_count",
                "meta_room_state_count",
                "meta_extraction_hour",
                "meta_extraction_day_of_week",
                "meta_data_quality_score",
            ]
        )

        return feature_names

    def create_feature_dataframe(
        self,
        feature_dicts: List[Dict[str, float]],
        feature_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from feature dictionaries.

        Args:
            feature_dicts: List of feature dictionaries
            feature_types: Which feature types were included

        Returns:
            DataFrame with features as columns
        """
        if not feature_dicts:
            return pd.DataFrame()

        # Get all possible feature names
        all_feature_names = self.get_feature_names(feature_types)

        # Create DataFrame with consistent columns
        df_data = []
        for feature_dict in feature_dicts:
            row = {}
            for feature_name in all_feature_names:
                row[feature_name] = feature_dict.get(feature_name, 0.0)
            df_data.append(row)

        return pd.DataFrame(df_data)

    def _get_default_features(self) -> Dict[str, float]:
        """Get default features for error cases."""
        # Combine default features from all extractors
        temporal_defaults = self.temporal_extractor._get_default_features()
        sequential_defaults = self.sequential_extractor._get_default_features()
        contextual_defaults = self.contextual_extractor._get_default_features()

        defaults = {}
        defaults.update({f"temporal_{k}": v for k, v in temporal_defaults.items()})
        defaults.update({f"sequential_{k}": v for k, v in sequential_defaults.items()})
        defaults.update({f"contextual_{k}": v for k, v in contextual_defaults.items()})

        # Add metadata defaults
        defaults.update(self._add_metadata_features("unknown", datetime.utcnow(), 0, 0))

        return defaults

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get feature extraction statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset extraction statistics."""
        self.stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_extraction_time": 0.0,
            "feature_counts": {
                "temporal": 0,
                "sequential": 0,
                "contextual": 0,
            },
        }

    def clear_caches(self):
        """Clear all feature extractor caches."""
        self.temporal_extractor.clear_cache()
        self.sequential_extractor.clear_cache()
        self.contextual_extractor.clear_cache()

    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the feature engineering configuration.

        Returns:
            Dictionary with validation results
        """
        validation_results = {"valid": True, "warnings": [], "errors": []}

        # Check if configuration is available
        if not self.config:
            validation_results["errors"].append("No system configuration available")
            validation_results["valid"] = False

        # Check room configurations
        if not self.config.rooms:
            validation_results["warnings"].append("No room configurations available")

        # Check feature extractor initialization
        extractors = {
            "temporal": self.temporal_extractor,
            "sequential": self.sequential_extractor,
            "contextual": self.contextual_extractor,
        }

        for extractor_name, extractor in extractors.items():
            if extractor is None:
                validation_results["errors"].append(
                    f"{extractor_name} extractor not initialized"
                )
                validation_results["valid"] = False

        # Check parallel processing setup
        if self.enable_parallel and self.executor is None:
            validation_results["warnings"].append(
                "Parallel processing enabled but executor not available"
            )

        return validation_results

    def __del__(self):
        """Cleanup when the engine is destroyed."""
        if self.executor:
            self.executor.shutdown(wait=False)
