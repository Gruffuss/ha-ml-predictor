"""
Bulk historical data importer for Home Assistant sensor events.

This module handles the efficient import of historical sensor data from Home Assistant,
with support for batch processing, progress tracking, resume capability, and validation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pickle
import traceback
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import SystemConfig, get_config
from ...core.exceptions import (
    DatabaseError,
    DataValidationError,
    HomeAssistantError,
    InsufficientTrainingDataError,
)
from ..storage.database import get_db_session
from ..storage.models import SensorEvent, get_bulk_insert_query
from .event_processor import EventProcessor
from .ha_client import HAEvent, HomeAssistantClient

logger = logging.getLogger(__name__)


@dataclass
class ImportProgress:
    """Tracks progress of bulk import operation."""

    total_entities: int = 0
    processed_entities: int = 0
    total_events: int = 0
    processed_events: int = 0
    valid_events: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    current_entity: str = ""
    current_date_range: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get total duration of import process."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def entity_progress_percent(self) -> float:
        """Get entity processing progress percentage."""
        if self.total_entities == 0:
            return 0.0
        return (self.processed_entities / self.total_entities) * 100

    @property
    def event_progress_percent(self) -> float:
        """Get event processing progress percentage."""
        if self.total_events == 0:
            return 0.0
        return (self.processed_events / self.total_events) * 100

    @property
    def events_per_second(self) -> float:
        """Get average events processed per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.processed_events / self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary for serialization."""
        return {
            "total_entities": self.total_entities,
            "processed_entities": self.processed_entities,
            "total_events": self.total_events,
            "processed_events": self.processed_events,
            "valid_events": self.valid_events,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "current_entity": self.current_entity,
            "current_date_range": self.current_date_range,
            "duration_seconds": self.duration_seconds,
            "entity_progress_percent": self.entity_progress_percent,
            "event_progress_percent": self.event_progress_percent,
            "events_per_second": self.events_per_second,
            "errors": self.errors[-10:],  # Only last 10 errors
        }


@dataclass
class ImportConfig:
    """Configuration for bulk import operation."""

    months_to_import: int = 6
    batch_size: int = 1000
    entity_batch_size: int = 10
    max_concurrent_entities: int = 3
    chunk_days: int = 7
    resume_file: Optional[str] = None
    skip_existing: bool = True
    validate_events: bool = True
    store_raw_data: bool = False
    progress_callback: Optional[callable] = None


class BulkImporter:
    """
    Bulk historical data importer for Home Assistant sensor events.

    Features:
    - Efficient batch processing with configurable chunk sizes
    - Progress tracking and resume capability
    - Concurrent processing of multiple entities
    - Data validation and error handling
    - Memory-efficient streaming for large datasets
    - Duplicate detection and skipping
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        import_config: Optional[ImportConfig] = None,
    ):
        self.config = config or get_config()
        self.import_config = import_config or ImportConfig()
        self.ha_client: Optional[HomeAssistantClient] = None
        self.event_processor: Optional[EventProcessor] = None
        self.progress = ImportProgress()

        # Resume capability
        self._resume_data: Dict[str, Any] = {}
        self._completed_entities: set = set()

        # Statistics
        self.stats = {
            "entities_processed": 0,
            "events_imported": 0,
            "events_skipped": 0,
            "validation_errors": 0,
            "database_errors": 0,
            "api_errors": 0,
        }

    async def import_historical_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> ImportProgress:
        """
        Import historical sensor data from Home Assistant.

        Args:
            start_date: Start date for import (defaults to X months ago)
            end_date: End date for import (defaults to now)
            entity_ids: Specific entities to import (defaults to all configured)

        Returns:
            ImportProgress with final import statistics
        """
        # Set default date range
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(
                days=30 * self.import_config.months_to_import
            )

        # Get entity list
        if entity_ids is None:
            entity_ids = self.config.get_all_entity_ids()

        logger.info(
            f"Starting bulk import of {len(entity_ids)} entities from {start_date} to {end_date}"
        )

        # Initialize components
        await self._initialize_components()

        # Load resume data if available
        await self._load_resume_data()

        # Update progress tracking
        self.progress.total_entities = len(entity_ids)
        self.progress.current_date_range = f"{start_date.date()} to {end_date.date()}"

        try:
            # Estimate total events for progress tracking
            await self._estimate_total_events(entity_ids, start_date, end_date)

            # Process entities in batches
            await self._process_entities_batch(entity_ids, start_date, end_date)

            # Generate final report
            await self._generate_import_report()

        except Exception as e:
            logger.error(f"Bulk import failed: {e}")
            self.progress.errors.append(f"Import failed: {str(e)}")
            raise
        finally:
            await self._cleanup_components()
            await self._save_resume_data()

        logger.info(
            f"Bulk import completed. Processed {self.progress.processed_events} events"
        )
        return self.progress

    async def _initialize_components(self):
        """Initialize HA client and event processor."""
        self.ha_client = HomeAssistantClient(self.config)
        await self.ha_client.connect()

        self.event_processor = EventProcessor(self.config)

        logger.info("Initialized bulk import components")

    async def _cleanup_components(self):
        """Clean up components and connections."""
        if self.ha_client:
            await self.ha_client.disconnect()

        logger.info("Cleaned up bulk import components")

    async def _load_resume_data(self):
        """Load resume data from previous import attempt."""
        if not self.import_config.resume_file:
            return

        resume_path = Path(self.import_config.resume_file)
        if not resume_path.exists():
            return

        try:
            with open(resume_path, "rb") as f:
                self._resume_data = pickle.load(f)

            self._completed_entities = set(
                self._resume_data.get("completed_entities", [])
            )

            logger.info(
                f"Loaded resume data: {len(self._completed_entities)} entities already processed"
            )

        except Exception as e:
            logger.warning(f"Failed to load resume data: {e}")

    async def _save_resume_data(self):
        """Save resume data for potential restart."""
        if not self.import_config.resume_file:
            return

        resume_data = {
            "completed_entities": list(self._completed_entities),
            "progress": self.progress.to_dict(),
            "stats": self.stats,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            resume_path = Path(self.import_config.resume_file)
            resume_path.parent.mkdir(parents=True, exist_ok=True)

            with open(resume_path, "wb") as f:
                pickle.dump(resume_data, f)

            logger.info(f"Saved resume data to {resume_path}")

        except Exception as e:
            logger.warning(f"Failed to save resume data: {e}")

    async def _estimate_total_events(
        self, entity_ids: List[str], start_date: datetime, end_date: datetime
    ):
        """Estimate total number of events for progress tracking."""
        # Sample a few entities to estimate event density
        sample_size = min(5, len(entity_ids))
        sample_entities = entity_ids[:sample_size]

        sample_events = 0
        sample_days = (end_date - start_date).days

        for entity_id in sample_entities:
            try:
                # Get one day of data as sample
                sample_start = start_date
                sample_end = min(start_date + timedelta(days=1), end_date)

                history = await self.ha_client.get_entity_history(
                    entity_id, sample_start, sample_end
                )
                sample_events += len(history)

            except HomeAssistantError as e:
                logger.warning(f"HA API error sampling entity {entity_id}: {e}")
                self.stats["api_errors"] += 1
            except Exception as e:
                logger.warning(f"Failed to sample entity {entity_id}: {e}")
                # Convert to HomeAssistantError for consistency
                raise HomeAssistantError(
                    f"Failed to sample entity {entity_id}: {str(e)}", cause=e
                )

        if sample_events > 0:
            # Estimate total events based on sample
            avg_events_per_entity_per_day = sample_events / (
                sample_size * 1
            )  # 1 day sample
            estimated_total = int(
                avg_events_per_entity_per_day * len(entity_ids) * sample_days
            )
            self.progress.total_events = estimated_total

            logger.info(f"Estimated {estimated_total} total events to process")

    async def _process_entities_batch(
        self, entity_ids: List[str], start_date: datetime, end_date: datetime
    ):
        """Process entities in batches with concurrency control."""
        # Filter out already completed entities
        remaining_entities = [
            e for e in entity_ids if e not in self._completed_entities
        ]

        if not remaining_entities:
            logger.info("All entities already processed")
            return

        # Process in batches
        for i in range(
            0, len(remaining_entities), self.import_config.entity_batch_size
        ):
            batch = remaining_entities[i : i + self.import_config.entity_batch_size]

            # Limit concurrency
            semaphore = asyncio.Semaphore(self.import_config.max_concurrent_entities)

            # Process batch concurrently
            tasks = [
                self._process_entity_with_semaphore(
                    semaphore, entity_id, start_date, end_date
                )
                for entity_id in batch
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Update progress
            self.progress.processed_entities += len(batch)
            await self._update_progress()

    async def _process_entity_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        entity_id: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """Process single entity with concurrency control."""
        async with semaphore:
            await self._process_single_entity(entity_id, start_date, end_date)

    async def _process_single_entity(
        self, entity_id: str, start_date: datetime, end_date: datetime
    ):
        """Process historical data for a single entity."""
        self.progress.current_entity = entity_id

        try:
            # Process in time chunks to manage memory
            current_date = start_date
            entity_events = 0

            while current_date < end_date:
                chunk_end = min(
                    current_date + timedelta(days=self.import_config.chunk_days),
                    end_date,
                )

                # Fetch historical data for chunk
                try:
                    history_data = await self.ha_client.get_entity_history(
                        entity_id, current_date, chunk_end
                    )

                    if history_data:
                        processed_count = await self._process_history_chunk(
                            entity_id, history_data
                        )
                        entity_events += processed_count

                        # Update progress
                        self.progress.processed_events += processed_count

                except Exception as e:
                    error_msg = (
                        f"Error processing {entity_id} chunk {current_date.date()}: {e}"
                    )
                    logger.error(error_msg)
                    self.progress.errors.append(error_msg)
                    self.stats["api_errors"] += 1

                current_date = chunk_end

                # Yield control periodically
                await asyncio.sleep(0.01)

            # Mark entity as completed
            self._completed_entities.add(entity_id)
            self.stats["entities_processed"] += 1

            logger.info(f"Completed entity {entity_id}: {entity_events} events")

        except Exception as e:
            error_msg = f"Failed to process entity {entity_id}: {e}"
            logger.error(error_msg)
            self.progress.errors.append(error_msg)

    async def _process_history_chunk(
        self, entity_id: str, history_data: List[Dict[str, Any]]
    ) -> int:
        """Process a chunk of historical data."""
        if not history_data:
            return 0

        try:
            # Convert to HAEvents
            ha_events = []
            for record in history_data:
                try:
                    ha_event = self._convert_history_record_to_ha_event(record)
                    if ha_event:
                        ha_events.append(ha_event)
                except DataValidationError as e:
                    logger.debug(
                        f"Skipping invalid history record due to validation error: {e}"
                    )
                    self.stats["validation_errors"] += 1
                except Exception as e:
                    logger.debug(f"Skipping invalid history record: {e}")
                    # Convert to validation error for consistency
                    raise DataValidationError(
                        data_source="history_record",
                        validation_errors=[str(e)],
                        sample_data=record,
                    )

            if not ha_events:
                return 0

            # Process events through event processor
            if self.import_config.validate_events:
                processed_events = await self.event_processor.process_event_batch(
                    ha_events
                )
            else:
                # Skip validation for faster processing
                processed_events = await self._convert_ha_events_to_sensor_events(
                    ha_events
                )

            # Bulk insert to database
            if processed_events:
                inserted_count = await self._bulk_insert_events(processed_events)
                self.stats["events_imported"] += inserted_count
                return inserted_count

            return 0

        except DataValidationError as e:
            logger.error(
                f"Data validation error processing history chunk for {entity_id}: {e}"
            )
            logger.debug(f"Validation error traceback: {traceback.format_exc()}")
            self.stats["validation_errors"] += 1
            return 0
        except DatabaseError as e:
            logger.error(
                f"Database error processing history chunk for {entity_id}: {e}"
            )
            logger.debug(f"Database error traceback: {traceback.format_exc()}")
            self.stats["database_errors"] += 1
            return 0
        except HomeAssistantError as e:
            logger.error(
                f"Home Assistant error processing history chunk for {entity_id}: {e}"
            )
            logger.debug(f"HA error traceback: {traceback.format_exc()}")
            self.stats["api_errors"] += 1
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected error processing history chunk for {entity_id}: {e}"
            )
            logger.debug(f"Unexpected error traceback: {traceback.format_exc()}")
            self.stats["database_errors"] += 1
            return 0

    def _convert_history_record_to_ha_event(
        self, record: Dict[str, Any]
    ) -> Optional[HAEvent]:
        """Convert Home Assistant history record to HAEvent."""
        try:
            timestamp_str = record.get("last_changed", record.get("last_updated", ""))
            if not timestamp_str:
                return None

            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            return HAEvent(
                entity_id=record.get("entity_id", ""),
                state=record.get("state", ""),
                previous_state=None,  # Not available in history API
                timestamp=timestamp,
                attributes=record.get("attributes", {}),
            )
        except Exception as e:
            logger.debug(f"Failed to convert history record: {e}")
            return None

    async def _convert_ha_events_to_sensor_events(
        self, ha_events: List[HAEvent]
    ) -> List[SensorEvent]:
        """Convert HAEvents to SensorEvents without full validation."""
        sensor_events = []

        for ha_event in ha_events:
            room_config = self.config.get_room_by_entity_id(ha_event.entity_id)
            if not room_config:
                continue

            sensor_type = self._determine_sensor_type(ha_event.entity_id, room_config)

            sensor_event = SensorEvent(
                room_id=room_config.room_id,
                sensor_id=ha_event.entity_id,
                sensor_type=sensor_type,
                state=ha_event.state,
                previous_state=ha_event.previous_state,
                timestamp=ha_event.timestamp,
                attributes=ha_event.attributes,
                is_human_triggered=True,  # Default for historical data
                created_at=datetime.utcnow(),
            )

            sensor_events.append(sensor_event)

        return sensor_events

    def _determine_sensor_type(self, entity_id: str, room_config) -> str:
        """Determine sensor type from entity ID and configuration."""
        # Check room configuration first
        for sensor_type, sensors in room_config.sensors.items():
            if isinstance(sensors, dict):
                if entity_id in sensors.values():
                    return sensor_type
            elif isinstance(sensors, str) and entity_id == sensors:
                return sensor_type

        # Fallback to entity ID analysis
        if "presence" in entity_id or "motion" in entity_id:
            return "presence"
        elif "door" in entity_id:
            return "door"
        elif "temperature" in entity_id:
            return "climate"
        elif "light" in entity_id:
            return "light"
        else:
            return "motion"

    async def _bulk_insert_events(self, events: List[SensorEvent]) -> int:
        """Bulk insert events into database efficiently."""
        if not events:
            return 0

        try:
            async with get_db_session() as session:
                # Type hint for better IDE support
                session: AsyncSession
                # Prepare bulk insert data
                insert_data = []
                for event in events:
                    insert_data.append(
                        {
                            "timestamp": event.timestamp,
                            "room_id": event.room_id,
                            "sensor_id": event.sensor_id,
                            "sensor_type": event.sensor_type,
                            "state": event.state,
                            "previous_state": event.previous_state,
                            "attributes": event.attributes,
                            "is_human_triggered": event.is_human_triggered,
                            "confidence_score": getattr(
                                event, "confidence_score", None
                            ),
                            "created_at": event.created_at,
                        }
                    )

                # Use optimized bulk insert query
                if len(insert_data) > self.import_config.batch_size:
                    # For very large batches, use raw SQL for better performance
                    bulk_query = get_bulk_insert_query()
                    result = await session.execute(text(bulk_query), insert_data)
                else:
                    # Use PostgreSQL INSERT ... ON CONFLICT for efficiency
                    stmt = insert(SensorEvent.__table__).values(insert_data)
                    if self.import_config.skip_existing:
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["timestamp", "room_id", "sensor_id"]
                        )
                    result = await session.execute(stmt)
                await session.commit()

                return result.rowcount if result.rowcount else len(insert_data)

        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            self.stats["database_errors"] += 1
            raise DatabaseError(f"Bulk insert failed: {str(e)}", cause=e)

    async def _update_progress(self):
        """Update progress and call progress callback if configured."""
        self.progress.last_update = datetime.utcnow()

        if self.import_config.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.import_config.progress_callback):
                    await self.import_config.progress_callback(self.progress)
                else:
                    self.import_config.progress_callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        # Log progress periodically
        if self.progress.processed_entities % 10 == 0:
            logger.info(
                f"Progress: {self.progress.entity_progress_percent:.1f}% entities, "
                f"{self.progress.processed_events} events, "
                f"{self.progress.events_per_second:.1f} events/sec"
            )

    async def _generate_import_report(self):
        """Generate comprehensive import report."""
        duration = self.progress.duration_seconds

        report = {
            "import_summary": {
                "total_duration_seconds": duration,
                "total_entities": self.progress.total_entities,
                "entities_processed": self.stats["entities_processed"],
                "total_events_processed": self.progress.processed_events,
                "valid_events": self.progress.valid_events,
                "events_imported": self.stats["events_imported"],
                "events_skipped": self.stats["events_skipped"],
                "average_events_per_second": self.progress.events_per_second,
            },
            "error_summary": {
                "validation_errors": self.stats["validation_errors"],
                "database_errors": self.stats["database_errors"],
                "api_errors": self.stats["api_errors"],
                "total_errors": len(self.progress.errors),
            },
            "data_quality": {
                "success_rate": (
                    self.stats["events_imported"]
                    / max(self.progress.processed_events, 1)
                )
                * 100,
                "error_rate": (
                    len(self.progress.errors) / max(self.progress.processed_events, 1)
                )
                * 100,
            },
        }

        logger.info(f"Import report: {json.dumps(report, indent=2)}")

        # Save detailed report if configured
        if (
            hasattr(self.import_config, "report_file")
            and self.import_config.report_file
        ):
            try:
                with open(self.import_config.report_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save report: {e}")

    def get_import_stats(self) -> Dict[str, Any]:
        """Get current import statistics."""
        return {
            "progress": self.progress.to_dict(),
            "stats": self.stats.copy(),
        }

    async def validate_data_sufficiency(
        self,
        room_id: str,
        minimum_days: int = 30,
        minimum_events_per_day: int = 10,
    ) -> Dict[str, Any]:
        """
        Validate that imported data is sufficient for model training.

        Args:
            room_id: Room to validate
            minimum_days: Minimum days of data required
            minimum_events_per_day: Minimum events per day required

        Returns:
            Validation results with recommendations
        """
        try:
            async with get_db_session() as session:
                # Count events by day for the room
                query = text(
                    """
                    SELECT
                        DATE(timestamp) as event_date,
                        COUNT(*) as event_count
                    FROM sensor_events
                    WHERE room_id =:room_id
                        AND timestamp >= NOW() - INTERVAL '90 days'
                    GROUP BY DATE(timestamp)
                    ORDER BY event_date DESC
                """
                )

                result = await session.execute(query, {"room_id": room_id})
                daily_counts = result.fetchall()

                if not daily_counts:
                    # Raise specific exception for insufficient data
                    raise InsufficientTrainingDataError(
                        room_id=room_id,
                        data_points=0,
                        minimum_required=minimum_days * minimum_events_per_day,
                        time_span_days=0,
                    )

                # Analyze data sufficiency
                total_days = len(daily_counts)
                avg_events_per_day = (
                    sum(row.event_count for row in daily_counts) / total_days
                )

                sufficient_days = total_days >= minimum_days
                sufficient_events = avg_events_per_day >= minimum_events_per_day

                # Raise specific exception if data is insufficient
                if not (sufficient_days and sufficient_events):
                    total_events = sum(row.event_count for row in daily_counts)
                    minimum_required = minimum_days * minimum_events_per_day

                    raise InsufficientTrainingDataError(
                        room_id=room_id,
                        data_points=total_events,
                        minimum_required=minimum_required,
                        time_span_days=total_days,
                    )

                return {
                    "sufficient": sufficient_days and sufficient_events,
                    "total_days": total_days,
                    "average_events_per_day": avg_events_per_day,
                    "minimum_days_required": minimum_days,
                    "minimum_events_per_day_required": minimum_events_per_day,
                    "meets_day_requirement": sufficient_days,
                    "meets_event_requirement": sufficient_events,
                    "recommendation": self._generate_sufficiency_recommendation(
                        sufficient_days,
                        sufficient_events,
                        total_days,
                        avg_events_per_day,
                    ),
                }

        except InsufficientTrainingDataError:
            raise  # Re-raise specific data insufficiency errors
        except DatabaseError as e:
            logger.error(f"Database error during data sufficiency validation: {e}")
            logger.debug(f"Database error traceback: {traceback.format_exc()}")
            raise
        except Exception as e:
            logger.error(f"Data sufficiency validation failed: {e}")
            logger.debug(f"Validation error traceback: {traceback.format_exc()}")
            raise DatabaseError(
                f"Data sufficiency validation failed: {str(e)}", cause=e
            )

    def _generate_sufficiency_recommendation(
        self,
        sufficient_days: bool,
        sufficient_events: bool,
        total_days: int,
        avg_events_per_day: float,
    ) -> str:
        """Generate recommendation based on data sufficiency analysis."""
        if sufficient_days and sufficient_events:
            return "Data is sufficient for model training"
        elif not sufficient_days:
            return f"Need more historical data (only {total_days} days available)"
        elif not sufficient_events:
            return f"Low event frequency ({avg_events_per_day:.1f} events/day), check sensor configuration"
        else:
            return "Increase historical data range and verify sensor operation"
