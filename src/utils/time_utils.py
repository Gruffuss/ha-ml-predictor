"""
Time utilities for Home Assistant ML Predictor.
Provides timezone-aware time operations, duration calculations, and edge case handling.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from typing import Dict, List, Optional, Union

import pytz


class TimeFrame(Enum):
    """Standard time frames for occupancy prediction."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TimeRange:
    """Represents a time range with timezone awareness."""

    def __init__(
        self, start: datetime, end: datetime, timezone_str: Optional[str] = None
    ):
        """
        Initialize time range.

        Args:
            start: Start datetime
            end: End datetime
            timezone_str: Optional timezone string (e.g., 'America/New_York')
        """
        self.start = self._ensure_timezone(start, timezone_str)
        self.end = self._ensure_timezone(end, timezone_str)

        if self.start >= self.end:
            raise ValueError("Start time must be before end time")

    def _ensure_timezone(self, dt: datetime, timezone_str: Optional[str]) -> datetime:
        """Ensure datetime has timezone information."""
        if dt.tzinfo is None:
            if timezone_str:
                tz = pytz.timezone(timezone_str)
                return tz.localize(dt)
            else:
                return dt.replace(tzinfo=timezone.utc)
        return dt

    @property
    def duration(self) -> timedelta:
        """Get the duration of this time range."""
        return self.end - self.start

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within this range."""
        dt = self._ensure_timezone(dt, None)
        return self.start <= dt <= self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and self.end > other.start

    def intersection(self, other: "TimeRange") -> Optional["TimeRange"]:
        """Get the intersection with another time range."""
        if not self.overlaps(other):
            return None

        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return TimeRange(start, end)


class TimeUtils:
    """Comprehensive time utility functions."""

    # Standard timezone objects for common cases
    UTC = timezone.utc
    LOCAL = None  # Will be set to system timezone

    @classmethod
    def setup_local_timezone(cls, timezone_str: Optional[str] = None):
        """Set up the local timezone."""
        if timezone_str:
            cls.LOCAL = pytz.timezone(timezone_str)
        else:
            # Try to detect system timezone
            try:
                import time

                cls.LOCAL = pytz.timezone(time.tzname[0])
            except Exception:
                cls.LOCAL = pytz.UTC

    @staticmethod
    def now(tz: Optional[timezone] = None) -> datetime:
        """Get current time with optional timezone."""
        if tz is None:
            tz = TimeUtils.UTC
        return datetime.now(tz)

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC time."""
        return datetime.now(timezone.utc)

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC."""
        if dt.tzinfo is None:
            # Assume local timezone
            if TimeUtils.LOCAL:
                dt = TimeUtils.LOCAL.localize(dt)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def to_timezone(dt: datetime, tz: Union[str, timezone]) -> datetime:
        """Convert datetime to specified timezone."""
        if isinstance(tz, str):
            tz = pytz.timezone(tz)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(tz)

    @staticmethod
    def parse_datetime(
        dt_str: str,
        formats: Optional[List[str]] = None,
        timezone_str: Optional[str] = None,
    ) -> datetime:
        """
        Parse datetime string with multiple format attempts.

        Args:
            dt_str: Datetime string to parse
            formats: List of format strings to try
            timezone_str: Timezone to apply if parsed datetime is naive

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If datetime cannot be parsed
        """
        if formats is None:
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with microseconds + Z
                "%Y-%m-%dT%H:%M:%SZ",  # ISO + Z
                "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with microseconds + timezone
                "%Y-%m-%dT%H:%M:%S%z",  # ISO + timezone
                "%Y-%m-%dT%H:%M:%S.%f",  # ISO with microseconds
                "%Y-%m-%dT%H:%M:%S",  # ISO basic
                "%Y-%m-%d %H:%M:%S.%f",  # SQL timestamp with microseconds
                "%Y-%m-%d %H:%M:%S",  # SQL timestamp
                "%Y-%m-%d",  # Date only
            ]

        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)

                # Handle timezone
                if dt.tzinfo is None and timezone_str:
                    tz = pytz.timezone(timezone_str)
                    dt = tz.localize(dt)
                elif dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return dt
            except ValueError:
                continue

        raise ValueError(f"Unable to parse datetime string: {dt_str}")

    @staticmethod
    def format_duration(duration: timedelta, precision: int = 2) -> str:
        """
        Format duration as human-readable string.

        Args:
            duration: Duration to format
            precision: Number of units to include (e.g., "2 hours 30 minutes")

        Returns:
            Formatted duration string
        """
        if duration.total_seconds() == 0:
            return "0 seconds"

        total_seconds = int(abs(duration.total_seconds()))

        units = [
            ("day", 86400),
            ("hour", 3600),
            ("minute", 60),
            ("second", 1),
        ]

        parts = []
        for unit_name, unit_seconds in units:
            if total_seconds >= unit_seconds:
                count = total_seconds // unit_seconds
                total_seconds %= unit_seconds

                unit_str = unit_name if count == 1 else f"{unit_name}s"
                parts.append(f"{count} {unit_str}")

                if len(parts) >= precision:
                    break

        if not parts:
            return "less than 1 second"

        result = ", ".join(parts)
        return f"-{result}" if duration.total_seconds() < 0 else result

    @staticmethod
    def time_until(
        target_time: datetime, from_time: Optional[datetime] = None
    ) -> timedelta:
        """
        Calculate time until target time.

        Args:
            target_time: Target datetime
            from_time: Reference time (defaults to now)

        Returns:
            Time delta until target time
        """
        if from_time is None:
            from_time = TimeUtils.utc_now()

        # Ensure both times have timezone info
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)

        return target_time - from_time

    @staticmethod
    def time_since(
        reference_time: datetime, from_time: Optional[datetime] = None
    ) -> timedelta:
        """
        Calculate time since reference time.

        Args:
            reference_time: Reference datetime
            from_time: Current time (defaults to now)

        Returns:
            Time delta since reference time
        """
        if from_time is None:
            from_time = TimeUtils.utc_now()

        # Ensure both times have timezone info
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)

        return from_time - reference_time

    @staticmethod
    def round_to_interval(
        dt: datetime, interval: timedelta, direction: str = "nearest"
    ) -> datetime:
        """
        Round datetime to specified interval.

        Args:
            dt: Datetime to round
            interval: Interval to round to
            direction: 'up', 'down', or 'nearest'

        Returns:
            Rounded datetime
        """
        # Calculate seconds from epoch
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        seconds_since_epoch = (dt - epoch).total_seconds()
        interval_seconds = interval.total_seconds()

        if direction == "down":
            rounded_seconds = (
                math.floor(seconds_since_epoch / interval_seconds) * interval_seconds
            )
        elif direction == "up":
            rounded_seconds = (
                math.ceil(seconds_since_epoch / interval_seconds) * interval_seconds
            )
        else:  # nearest
            rounded_seconds = (
                round(seconds_since_epoch / interval_seconds) * interval_seconds
            )

        return epoch + timedelta(seconds=rounded_seconds)

    @staticmethod
    def get_time_buckets(
        start: datetime, end: datetime, interval: timedelta
    ) -> List[datetime]:
        """
        Generate time buckets between start and end with specified interval.

        Args:
            start: Start datetime
            end: End datetime
            interval: Bucket interval

        Returns:
            List of bucket start times
        """
        buckets = []
        current = start

        while current < end:
            buckets.append(current)
            current += interval

        return buckets

    @staticmethod
    def is_business_hours(
        dt: datetime,
        start_hour: int = 9,
        end_hour: int = 17,
        weekdays_only: bool = True,
    ) -> bool:
        """
        Check if datetime falls within business hours.

        Args:
            dt: Datetime to check
            start_hour: Business day start hour (24-hour format)
            end_hour: Business day end hour (24-hour format)
            weekdays_only: Whether to consider weekends as non-business days

        Returns:
            True if within business hours
        """
        if weekdays_only and dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        return start_hour <= dt.hour < end_hour

    @staticmethod
    def get_cyclical_time_features(dt: datetime) -> Dict[str, float]:
        """
        Extract cyclical time features for ML models.

        Args:
            dt: Datetime to extract features from

        Returns:
            Dictionary of cyclical features (sin/cos pairs)
        """
        # Extract time components
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()  # 0 = Monday
        day_of_month = dt.day
        month = dt.month

        # Convert to cyclical features using sin/cos
        features = {
            # Hour of day (0-23)
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            # Minute of hour (0-59)
            "minute_sin": math.sin(2 * math.pi * minute / 60),
            "minute_cos": math.cos(2 * math.pi * minute / 60),
            # Day of week (0-6)
            "day_of_week_sin": math.sin(2 * math.pi * day_of_week / 7),
            "day_of_week_cos": math.cos(2 * math.pi * day_of_week / 7),
            # Day of month (1-31, approximate)
            "day_of_month_sin": math.sin(2 * math.pi * day_of_month / 31),
            "day_of_month_cos": math.cos(2 * math.pi * day_of_month / 31),
            # Month of year (1-12)
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
        }

        return features

    @staticmethod
    def validate_timezone(timezone_str: str) -> bool:
        """
        Validate timezone string.

        Args:
            timezone_str: Timezone string to validate

        Returns:
            True if valid timezone
        """
        try:
            pytz.timezone(timezone_str)
            return True
        except pytz.UnknownTimeZoneError:
            return False


class AsyncTimeUtils:
    """Async utilities for time-based operations."""

    @staticmethod
    async def wait_until(target_time: datetime, check_interval: float = 1.0):
        """
        Wait until a specific time.

        Args:
            target_time: Target datetime to wait until
            check_interval: How often to check time (seconds)
        """
        while True:
            now = TimeUtils.utc_now()
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)

            if now >= target_time:
                break

            time_diff = (target_time - now).total_seconds()
            sleep_time = min(time_diff, check_interval)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    @staticmethod
    async def periodic_task(
        interval: timedelta,
        max_iterations: Optional[int] = None,
        align_to_interval: bool = False,
    ):
        """
        Async generator for periodic task execution.

        Args:
            interval: Time interval between iterations
            max_iterations: Maximum number of iterations (None for infinite)
            align_to_interval: Whether to align to interval boundaries

        Yields:
            Current iteration number (starting from 0)
        """
        iteration = 0
        start_time = TimeUtils.utc_now()

        if align_to_interval:
            # Round start time to next interval boundary
            start_time = TimeUtils.round_to_interval(
                start_time, interval, direction="up"
            )
            await AsyncTimeUtils.wait_until(start_time)

        while max_iterations is None or iteration < max_iterations:
            yield iteration

            # Calculate next execution time
            next_time = start_time + (interval * (iteration + 1))
            await AsyncTimeUtils.wait_until(next_time)

            iteration += 1


class TimeProfiler:
    """Context manager and decorator for timing operations."""

    def __init__(self, operation_name: str = "operation"):
        """
        Initialize profiler.

        Args:
            operation_name: Name of the operation being profiled
        """
        self.operation_name = operation_name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self._duration: Optional[timedelta] = None

    def __enter__(self) -> "TimeProfiler":
        """Start timing."""
        self.start_time = TimeUtils.utc_now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.end_time = TimeUtils.utc_now()
        if self.start_time:
            self._duration = self.end_time - self.start_time

    @property
    def duration(self) -> Optional[timedelta]:
        """Get the measured duration."""
        return self._duration

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get the measured duration in seconds."""
        if self._duration:
            return self._duration.total_seconds()
        elif self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        else:
            return None

    def __call__(self, func):
        """Decorator usage."""

        def wrapper(*args, **kwargs):
            with TimeProfiler(f"{func.__name__}") as profiler:
                result = func(*args, **kwargs)

            # Store timing info in result if it's a dict
            if isinstance(result, dict):
                result["_timing"] = {
                    "duration_seconds": profiler.duration_seconds,
                    "operation": profiler.operation_name,
                }

            return result

        return wrapper


# Convenience functions
def format_duration(duration: timedelta, precision: int = 2) -> str:
    """Convenience function for formatting durations."""
    return TimeUtils.format_duration(duration, precision)


def time_until(target_time: datetime) -> timedelta:
    """Convenience function for calculating time until target."""
    return TimeUtils.time_until(target_time)


def time_since(reference_time: datetime) -> timedelta:
    """Convenience function for calculating time since reference."""
    return TimeUtils.time_since(reference_time)


def cyclical_time_features(dt: datetime) -> Dict[str, float]:
    """Convenience function for extracting cyclical time features."""
    return TimeUtils.get_cyclical_time_features(dt)


# Initialize local timezone on import
TimeUtils.setup_local_timezone()
