"""
Comprehensive unit tests for time_utils.py.
Tests timezone handling, edge cases, cyclical features, and performance.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import math
from unittest.mock import Mock, patch

import pytest
import pytz

from src.utils.time_utils import (
    AsyncTimeUtils,
    TimeFrame,
    TimeProfiler,
    TimeRange,
    TimeUtils,
    cyclical_time_features,
    format_duration,
    time_since,
    time_until,
)


class TestTimeRange:
    """Test TimeRange class."""

    def test_time_range_creation(self):
        """Test basic TimeRange creation."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        time_range = TimeRange(start, end)
        assert time_range.start == start
        assert time_range.end == end
        assert time_range.duration == timedelta(hours=2)

    def test_time_range_with_timezone_string(self):
        """Test TimeRange creation with timezone string."""
        # Naive datetimes
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0)

        time_range = TimeRange(start, end, "America/New_York")

        # Should have timezone applied
        assert time_range.start.tzinfo is not None
        assert time_range.end.tzinfo is not None
        assert str(time_range.start.tzinfo) == "America/New_York"

    def test_time_range_invalid_order(self):
        """Test TimeRange with invalid time order."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Start time must be before end time"):
            TimeRange(start, end)

    def test_time_range_contains(self):
        """Test TimeRange contains method."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        time_range = TimeRange(start, end)

        # Within range
        middle = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        assert time_range.contains(middle)

        # At boundaries
        assert time_range.contains(start)
        assert time_range.contains(end)

        # Outside range
        before = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        after = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        assert not time_range.contains(before)
        assert not time_range.contains(after)

    def test_time_range_overlaps(self):
        """Test TimeRange overlaps method."""
        range1 = TimeRange(
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        # Overlapping range
        range2 = TimeRange(
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
        )
        assert range1.overlaps(range2)
        assert range2.overlaps(range1)

        # Non-overlapping range
        range3 = TimeRange(
            datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
        )
        assert not range1.overlaps(range3)
        assert not range3.overlaps(range1)

        # Adjacent ranges (touching endpoints)
        range4 = TimeRange(
            datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
        )
        assert not range1.overlaps(range4)  # Touching endpoints don't overlap

    def test_time_range_intersection(self):
        """Test TimeRange intersection method."""
        range1 = TimeRange(
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        # Overlapping range
        range2 = TimeRange(
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
        )

        intersection = range1.intersection(range2)
        assert intersection is not None
        assert intersection.start == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
        assert intersection.end == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

        # Non-overlapping range
        range3 = TimeRange(
            datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
        )

        intersection = range1.intersection(range3)
        assert intersection is None


class TestTimeUtils:
    """Test TimeUtils class."""

    def test_now_with_timezone(self):
        """Test TimeUtils.now with different timezones."""
        utc_now = TimeUtils.now(timezone.utc)
        assert utc_now.tzinfo == timezone.utc

        # Test with specific timezone
        ny_tz = pytz.timezone("America/New_York")
        ny_now = TimeUtils.now(ny_tz)
        assert ny_now.tzinfo.zone == "America/New_York"

    def test_utc_now(self):
        """Test TimeUtils.utc_now."""
        utc_now = TimeUtils.utc_now()
        assert utc_now.tzinfo == timezone.utc
        assert isinstance(utc_now, datetime)

    def test_to_utc_conversion(self):
        """Test TimeUtils.to_utc with various inputs."""
        # Naive datetime (should assume local/UTC)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        utc_dt = TimeUtils.to_utc(naive_dt)
        assert utc_dt.tzinfo == timezone.utc

        # Already UTC datetime
        utc_original = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        utc_converted = TimeUtils.to_utc(utc_original)
        assert utc_converted == utc_original
        assert utc_converted.tzinfo == timezone.utc

        # Different timezone
        ny_tz = pytz.timezone("America/New_York")
        ny_dt = ny_tz.localize(datetime(2024, 1, 1, 12, 0, 0))
        utc_from_ny = TimeUtils.to_utc(ny_dt)
        assert utc_from_ny.tzinfo == timezone.utc
        # Should be 5 hours ahead (EST offset)
        assert utc_from_ny.hour == 17  # 12 PM EST = 5 PM UTC

    def test_to_timezone_conversion(self):
        """Test TimeUtils.to_timezone."""
        utc_dt = datetime(2024, 1, 1, 17, 0, 0, tzinfo=timezone.utc)

        # Convert to New York time
        ny_dt = TimeUtils.to_timezone(utc_dt, "America/New_York")
        assert ny_dt.hour == 12  # 5 PM UTC = 12 PM EST

        # Test with timezone object
        ny_tz = pytz.timezone("America/New_York")
        ny_dt2 = TimeUtils.to_timezone(utc_dt, ny_tz)
        assert ny_dt2.hour == 12

    def test_parse_datetime_formats(self):
        """Test TimeUtils.parse_datetime with various formats."""
        # ISO format with microseconds and Z
        dt1 = TimeUtils.parse_datetime("2024-01-01T12:00:00.123456Z")
        assert dt1.year == 2024
        assert dt1.hour == 12
        assert dt1.microsecond == 123456
        assert dt1.tzinfo == timezone.utc

        # ISO format without timezone
        dt2 = TimeUtils.parse_datetime("2024-01-01T12:00:00")
        assert dt2.year == 2024
        assert dt2.hour == 12
        assert dt2.tzinfo == timezone.utc

        # SQL timestamp format
        dt3 = TimeUtils.parse_datetime("2024-01-01 12:00:00")
        assert dt3.year == 2024
        assert dt3.hour == 12

        # Date only
        dt4 = TimeUtils.parse_datetime("2024-01-01")
        assert dt4.year == 2024
        assert dt4.hour == 0

        # Custom timezone
        dt5 = TimeUtils.parse_datetime(
            "2024-01-01T12:00:00", timezone_str="America/New_York"
        )
        assert dt5.tzinfo.zone == "America/New_York"

    def test_parse_datetime_invalid(self):
        """Test TimeUtils.parse_datetime with invalid input."""
        with pytest.raises(ValueError, match="Unable to parse datetime string"):
            TimeUtils.parse_datetime("invalid-datetime")

    def test_format_duration(self):
        """Test TimeUtils.format_duration."""
        # Zero duration
        assert TimeUtils.format_duration(timedelta(0)) == "0 seconds"

        # Seconds only
        assert TimeUtils.format_duration(timedelta(seconds=30)) == "30 seconds"
        assert TimeUtils.format_duration(timedelta(seconds=1)) == "1 second"

        # Minutes and seconds
        assert (
            TimeUtils.format_duration(timedelta(minutes=2, seconds=30))
            == "2 minutes, 30 seconds"
        )

        # Hours, minutes
        duration = timedelta(hours=1, minutes=30, seconds=45)
        result = TimeUtils.format_duration(duration, precision=2)
        assert result == "1 hour, 30 minutes"

        # Days, hours, minutes with higher precision
        duration = timedelta(days=2, hours=3, minutes=15, seconds=30)
        result = TimeUtils.format_duration(duration, precision=3)
        assert result == "2 days, 3 hours, 15 minutes"

        # Negative duration
        negative_duration = timedelta(hours=-2, minutes=-30)
        result = TimeUtils.format_duration(negative_duration)
        assert result.startswith("-")
        assert "2 hours" in result

        # Very small duration
        tiny_duration = timedelta(milliseconds=500)
        result = TimeUtils.format_duration(tiny_duration)
        assert result == "less than 1 second"

    def test_time_until_and_since(self):
        """Test TimeUtils.time_until and time_since."""
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        future = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        past = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Time until future
        until = TimeUtils.time_until(future, now)
        assert until == timedelta(hours=2)

        # Time since past
        since = TimeUtils.time_since(past, now)
        assert since == timedelta(hours=2)

        # Test with naive datetimes (should work)
        naive_now = datetime(2024, 1, 1, 12, 0, 0)
        naive_future = datetime(2024, 1, 1, 14, 0, 0)
        until_naive = TimeUtils.time_until(naive_future, naive_now)
        assert until_naive == timedelta(hours=2)

    def test_round_to_interval(self):
        """Test TimeUtils.round_to_interval."""
        dt = datetime(2024, 1, 1, 12, 37, 30, tzinfo=timezone.utc)

        # Round to nearest hour
        hour_interval = timedelta(hours=1)

        # Round down
        rounded_down = TimeUtils.round_to_interval(dt, hour_interval, "down")
        assert rounded_down.hour == 12
        assert rounded_down.minute == 0

        # Round up
        rounded_up = TimeUtils.round_to_interval(dt, hour_interval, "up")
        assert rounded_up.hour == 13
        assert rounded_up.minute == 0

        # Round to nearest (should round up since 37 minutes > 30)
        rounded_nearest = TimeUtils.round_to_interval(dt, hour_interval, "nearest")
        assert rounded_nearest.hour == 13
        assert rounded_nearest.minute == 0

        # Test with 15-minute intervals
        quarter_hour = timedelta(minutes=15)
        rounded_quarter = TimeUtils.round_to_interval(dt, quarter_hour, "nearest")
        # 37 minutes is closest to 30 minutes (2nd quarter) - 37 is closer to 30 than 45
        assert rounded_quarter.minute == 30

    def test_get_time_buckets(self):
        """Test TimeUtils.get_time_buckets."""
        start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        interval = timedelta(hours=1)

        buckets = TimeUtils.get_time_buckets(start, end, interval)

        assert len(buckets) == 2
        assert buckets[0] == start
        assert buckets[1] == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)

        # Test with smaller intervals
        interval_30min = timedelta(minutes=30)
        buckets_30 = TimeUtils.get_time_buckets(start, end, interval_30min)
        assert len(buckets_30) == 4  # 10:00, 10:30, 11:00, 11:30

    def test_is_business_hours(self):
        """Test TimeUtils.is_business_hours."""
        # Tuesday 2 PM (business hours)
        business_time = datetime(2024, 1, 2, 14, 0)  # Tuesday
        assert TimeUtils.is_business_hours(business_time)

        # Tuesday 8 AM (before business hours)
        early_time = datetime(2024, 1, 2, 8, 0)
        assert not TimeUtils.is_business_hours(early_time)

        # Tuesday 6 PM (after business hours)
        late_time = datetime(2024, 1, 2, 18, 0)
        assert not TimeUtils.is_business_hours(late_time)

        # Saturday 2 PM (weekend)
        weekend_time = datetime(2024, 1, 6, 14, 0)  # Saturday
        assert not TimeUtils.is_business_hours(weekend_time)

        # Saturday 2 PM (weekend, but weekdays_only=False)
        assert TimeUtils.is_business_hours(weekend_time, weekdays_only=False)

        # Custom business hours
        assert TimeUtils.is_business_hours(
            datetime(2024, 1, 2, 8, 0), start_hour=8, end_hour=18  # 8 AM Tuesday
        )

    def test_get_cyclical_time_features(self):
        """Test TimeUtils.get_cyclical_time_features."""
        # Test specific time: January 1st, 2024, 6:30 AM, Monday
        dt = datetime(2024, 1, 1, 6, 30)  # Monday
        features = TimeUtils.get_cyclical_time_features(dt)

        # Check that all expected features are present
        expected_features = [
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "month_sin",
            "month_cos",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert -1.0 <= features[feature] <= 1.0

        # Test mathematical correctness for hour=6 (quarter of day)
        # 6/24 = 0.25, so 2*pi*0.25 = pi/2
        # sin(pi/2) = 1, cos(pi/2) = 0
        assert abs(features["hour_sin"] - 1.0) < 1e-10
        assert abs(features["hour_cos"]) < 1e-10

        # Test minute=30 (half of hour)
        # 30/60 = 0.5, so 2*pi*0.5 = pi
        # sin(pi) = 0, cos(pi) = -1
        assert abs(features["minute_sin"]) < 1e-10
        assert abs(features["minute_cos"] - (-1.0)) < 1e-10

        # Test Monday (day_of_week = 0)
        # 0/7 = 0, so sin(0) = 0, cos(0) = 1
        assert abs(features["day_of_week_sin"]) < 1e-10
        assert abs(features["day_of_week_cos"] - 1.0) < 1e-10

    def test_validate_timezone(self):
        """Test TimeUtils.validate_timezone."""
        # Valid timezones
        assert TimeUtils.validate_timezone("UTC")
        assert TimeUtils.validate_timezone("America/New_York")
        assert TimeUtils.validate_timezone("Europe/London")
        assert TimeUtils.validate_timezone("Asia/Tokyo")

        # Invalid timezones
        assert not TimeUtils.validate_timezone("Invalid/Timezone")
        assert not TimeUtils.validate_timezone("Not_A_Timezone")
        assert not TimeUtils.validate_timezone("")

    def test_edge_cases_timezone_handling(self):
        """Test edge cases in timezone handling."""
        # Daylight Saving Time transitions
        ny_tz = pytz.timezone("America/New_York")

        # Spring forward (2 AM becomes 3 AM)
        with pytest.raises(pytz.NonExistentTimeError):
            ny_tz.localize(datetime(2024, 3, 10, 2, 30), is_dst=None)

        # Fall back (2 AM happens twice)
        # This should work with is_dst specified
        fall_back_1 = ny_tz.localize(datetime(2024, 11, 3, 1, 30), is_dst=True)
        fall_back_2 = ny_tz.localize(datetime(2024, 11, 3, 1, 30), is_dst=False)
        assert fall_back_1 != fall_back_2

    def test_performance_large_intervals(self):
        """Test performance with large time intervals."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # This should complete quickly even with large ranges
        with TimeProfiler("large_interval_test") as profiler:
            buckets = TimeUtils.get_time_buckets(start, end, timedelta(days=1))

        # Should have approximately 4 years * 365 days = ~1460 buckets
        assert 1460 <= len(buckets) <= 1465  # Account for leap years
        assert profiler.duration_seconds < 1.0  # Should be very fast


class TestAsyncTimeUtils:
    """Test AsyncTimeUtils class."""

    @pytest.mark.asyncio
    async def test_wait_until(self):
        """Test AsyncTimeUtils.wait_until."""
        # Test waiting for a short time in the future
        start_time = TimeUtils.utc_now()
        target_time = start_time + timedelta(milliseconds=100)

        await AsyncTimeUtils.wait_until(target_time, check_interval=0.01)

        end_time = TimeUtils.utc_now()
        elapsed = (end_time - start_time).total_seconds()

        # Should have waited at least 100ms
        assert elapsed >= 0.1
        assert elapsed < 0.5  # But not too long

    @pytest.mark.asyncio
    async def test_wait_until_past_time(self):
        """Test AsyncTimeUtils.wait_until with past time."""
        # Should return immediately for past times
        past_time = TimeUtils.utc_now() - timedelta(seconds=1)

        start_time = TimeUtils.utc_now()
        await AsyncTimeUtils.wait_until(past_time)
        end_time = TimeUtils.utc_now()

        elapsed = (end_time - start_time).total_seconds()
        assert elapsed < 0.1  # Should be very quick

    @pytest.mark.asyncio
    async def test_periodic_task(self):
        """Test AsyncTimeUtils.periodic_task."""
        iterations = []
        start_time = TimeUtils.utc_now()

        async for iteration in AsyncTimeUtils.periodic_task(
            interval=timedelta(milliseconds=50), max_iterations=3
        ):
            iterations.append((iteration, TimeUtils.utc_now()))

        end_time = TimeUtils.utc_now()

        # Should have 3 iterations
        assert len(iterations) == 3
        assert iterations[0][0] == 0
        assert iterations[1][0] == 1
        assert iterations[2][0] == 2

        # Total time should be approximately 150ms (3 * 50ms)
        total_time = (end_time - start_time).total_seconds()
        assert 0.1 <= total_time <= 0.4  # Allow some variance

    @pytest.mark.asyncio
    async def test_periodic_task_aligned(self):
        """Test AsyncTimeUtils.periodic_task with alignment."""
        # This test is more complex and mainly checks that alignment works
        iterations = []

        async for iteration in AsyncTimeUtils.periodic_task(
            interval=timedelta(seconds=1), max_iterations=2, align_to_interval=True
        ):
            iterations.append((iteration, TimeUtils.utc_now()))
            # Break early to avoid long test runtime
            if iteration >= 0:
                break

        # Should have at least one iteration
        assert len(iterations) >= 1
        assert iterations[0][0] == 0


class TestTimeProfiler:
    """Test TimeProfiler class."""

    def test_context_manager_usage(self):
        """Test TimeProfiler as context manager."""
        import time

        with TimeProfiler("test_operation") as profiler:
            time.sleep(0.1)

        assert profiler.duration is not None
        assert profiler.duration_seconds >= 0.1
        assert profiler.duration_seconds < 0.2
        assert profiler.operation_name == "test_operation"

    def test_decorator_usage(self):
        """Test TimeProfiler as decorator."""
        profiler = TimeProfiler("test_function")

        @profiler
        def slow_function():
            import time

            time.sleep(0.05)
            return {"result": "success"}

        result = slow_function()

        # Should have timing info added
        assert "_timing" in result
        assert result["_timing"]["duration_seconds"] >= 0.05
        assert result["_timing"]["operation"] == "slow_function"
        assert result["result"] == "success"

    def test_decorator_non_dict_return(self):
        """Test TimeProfiler decorator with non-dict return."""
        profiler = TimeProfiler("test_function")

        @profiler
        def simple_function():
            import time

            time.sleep(0.01)
            return "simple_result"

        result = simple_function()

        # Should return original result without modification
        assert result == "simple_result"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_format_duration_convenience(self):
        """Test format_duration convenience function."""
        duration = timedelta(hours=1, minutes=30)
        result = format_duration(duration)
        assert "1 hour" in result
        assert "30 minutes" in result

    def test_time_until_convenience(self):
        """Test time_until convenience function."""
        target = TimeUtils.utc_now() + timedelta(hours=2)
        result = time_until(target)

        # Should be close to 2 hours
        assert 7195 <= result.total_seconds() <= 7205  # ~2 hours ± 5 seconds

    def test_time_since_convenience(self):
        """Test time_since convenience function."""
        reference = TimeUtils.utc_now() - timedelta(hours=1)
        result = time_since(reference)

        # Should be close to 1 hour
        assert 3595 <= result.total_seconds() <= 3605  # ~1 hour ± 5 seconds

    def test_cyclical_time_features_convenience(self):
        """Test cyclical_time_features convenience function."""
        dt = datetime(2024, 1, 1, 12, 0)
        features = cyclical_time_features(dt)

        assert "hour_sin" in features
        assert "hour_cos" in features
        assert isinstance(features["hour_sin"], float)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_leap_year_handling(self):
        """Test handling of leap years."""
        # 2024 is a leap year
        leap_day = datetime(2024, 2, 29, 12, 0, tzinfo=timezone.utc)
        features = TimeUtils.get_cyclical_time_features(leap_day)

        # Should handle leap day correctly
        assert "day_of_month_sin" in features
        # Day 29 out of 31: 29/31 ≈ 0.935
        expected_angle = 2 * math.pi * 29 / 31
        expected_sin = math.sin(expected_angle)
        assert abs(features["day_of_month_sin"] - expected_sin) < 1e-10

    def test_year_boundary_handling(self):
        """Test handling of year boundaries."""
        # New Year's Eve
        nye = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        ny = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Should be 1 second apart
        diff = TimeUtils.time_until(ny, nye)
        assert diff.total_seconds() == 1

    def test_extreme_dates(self):
        """Test handling of extreme dates."""
        # Very far future
        far_future = datetime(2100, 1, 1, tzinfo=timezone.utc)
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)

        diff = TimeUtils.time_until(far_future, now)
        assert diff.days > 25000  # Many years in the future

        # Very far past
        far_past = datetime(1900, 1, 1, tzinfo=timezone.utc)
        diff = TimeUtils.time_since(far_past, now)
        assert diff.days > 40000  # Many years in the past

    def test_microsecond_precision(self):
        """Test microsecond precision handling."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 1, 12, 0, 0, 123457, tzinfo=timezone.utc)

        diff = dt2 - dt1
        assert diff.microseconds == 1

        # Test with TimeUtils
        time_diff = TimeUtils.time_until(dt2, dt1)
        assert time_diff.microseconds == 1

    def test_different_timezone_comparisons(self):
        """Test comparisons across different timezones."""
        utc_time = datetime(2024, 1, 1, 17, 0, tzinfo=timezone.utc)
        ny_time = TimeUtils.to_timezone(utc_time, "America/New_York")
        london_time = TimeUtils.to_timezone(utc_time, "Europe/London")

        # All should represent the same moment in time
        assert TimeUtils.to_utc(utc_time) == TimeUtils.to_utc(ny_time)
        assert TimeUtils.to_utc(utc_time) == TimeUtils.to_utc(london_time)

        # But have different local times
        assert utc_time.hour == 17
        assert ny_time.hour == 12  # EST
        assert london_time.hour == 17  # GMT (same as UTC in winter)


class TestPerformanceAndMemory:
    """Test performance and memory characteristics."""

    def test_large_time_bucket_generation(self):
        """Test memory efficiency with large time bucket generation."""
        import sys

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)  # 24 hours
        interval = timedelta(minutes=1)  # 1440 buckets

        with TimeProfiler("bucket_generation") as profiler:
            buckets = TimeUtils.get_time_buckets(start, end, interval)

        # Should be fast and memory-efficient
        assert len(buckets) == 1440  # 24 * 60 minutes
        assert profiler.duration_seconds < 0.1  # Should be very fast

        # Check memory usage (rough estimate)
        bucket_size = sys.getsizeof(buckets[0]) if buckets else 0
        total_size = sys.getsizeof(buckets) + (bucket_size * len(buckets))
        # Should be reasonable (less than 1MB for 1440 datetime objects)
        assert total_size < 1024 * 1024

    def test_cyclical_features_performance(self):
        """Test performance of cyclical feature extraction."""
        test_times = [
            datetime(2024, 1, 1, h, m) for h in range(24) for m in range(0, 60, 15)
        ]

        with TimeProfiler("cyclical_features") as profiler:
            all_features = [
                TimeUtils.get_cyclical_time_features(dt) for dt in test_times
            ]

        # Should process 96 timestamps quickly (24 hours * 4 quarters)
        assert len(all_features) == 96
        assert profiler.duration_seconds < 0.1

        # All features should have consistent structure
        for features in all_features:
            assert len(features) == 10  # 5 sin/cos pairs
            assert all(-1.0 <= v <= 1.0 for v in features.values())

    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async utilities performance."""
        # Test multiple concurrent wait_until operations
        import asyncio

        base_time = TimeUtils.utc_now()
        tasks = [
            AsyncTimeUtils.wait_until(base_time + timedelta(milliseconds=50 + i * 10))
            for i in range(5)
        ]

        start_time = TimeUtils.utc_now()
        await asyncio.gather(*tasks)
        end_time = TimeUtils.utc_now()

        # All tasks should complete in approximately the time of the longest wait
        # Longest wait is 50 + 4*10 = 90ms
        elapsed = (end_time - start_time).total_seconds()
        assert 0.09 <= elapsed <= 0.2  # Should be close to 90ms


class TestIntegrationWithOtherModules:
    """Test integration with other system components."""

    def test_logger_integration(self):
        """Test time utilities integration with logging."""
        from src.utils.logger import get_logger

        logger = get_logger("time_utils_test")

        # Test timing operations with logging
        with TimeProfiler("logged_operation") as profiler:
            # Simulate some work
            import time

            time.sleep(0.01)

            # Log with timing info
            logger.info(
                "Operation completed",
                extra={
                    "duration_seconds": profiler.duration_seconds,
                    "operation": profiler.operation_name,
                },
            )

        assert profiler.duration_seconds >= 0.01

    def test_metrics_integration(self):
        """Test time utilities integration with metrics."""
        try:
            from src.utils.metrics import get_metrics_collector

            collector = get_metrics_collector()

            # Test timing with metrics
            with TimeProfiler("metric_operation") as profiler:
                import time

                time.sleep(0.01)

            # Record timing metric
            if hasattr(collector, "record_prediction"):
                collector.record_prediction(
                    room_id="test_room",
                    prediction_type="occupancy",
                    model_type="test",
                    duration=profiler.duration_seconds,
                    status="success",
                )

            assert profiler.duration_seconds >= 0.01

        except ImportError:
            # Metrics module not available in test environment
            pytest.skip("Metrics module not available")

    def test_timezone_consistency_across_modules(self):
        """Test timezone consistency across different modules."""
        # Create timestamps that should be consistent
        utc_time = TimeUtils.utc_now()
        formatted_time = utc_time.isoformat()

        # Parse back and ensure consistency
        parsed_time = TimeUtils.parse_datetime(formatted_time)

        # Should be identical
        assert abs((parsed_time - utc_time).total_seconds()) < 0.001


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests for time utilities."""

    def test_datetime_parsing_performance(self):
        """Benchmark datetime parsing performance."""
        test_strings = [
            "2024-01-01T12:00:00.123456Z",
            "2024-01-01T12:00:00Z",
            "2024-01-01 12:00:00",
            "2024-01-01",
        ] * 100  # 400 total strings

        with TimeProfiler("datetime_parsing") as profiler:
            parsed_times = [TimeUtils.parse_datetime(dt_str) for dt_str in test_strings]

        assert len(parsed_times) == 400
        # Should parse 400 datetime strings in reasonable time
        assert profiler.duration_seconds < 1.0

        # Calculate average parsing time
        avg_time_per_parse = profiler.duration_seconds / 400
        assert avg_time_per_parse < 0.0025  # Less than 2.5ms per parse

    def test_timezone_conversion_performance(self):
        """Benchmark timezone conversion performance."""
        utc_times = [TimeUtils.utc_now() + timedelta(hours=i) for i in range(100)]

        timezones = ["America/New_York", "Europe/London", "Asia/Tokyo"]

        with TimeProfiler("timezone_conversions") as profiler:
            converted_times = [
                TimeUtils.to_timezone(dt, tz) for dt in utc_times for tz in timezones
            ]

        assert len(converted_times) == 300  # 100 times * 3 timezones
        # Should convert 300 timestamps in reasonable time
        assert profiler.duration_seconds < 0.5

    def test_cyclical_features_bulk_performance(self):
        """Benchmark bulk cyclical feature extraction."""
        # Generate a week's worth of hourly timestamps
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            start_time + timedelta(hours=i) for i in range(24 * 7)  # 168 hours
        ]

        with TimeProfiler("bulk_cyclical_features") as profiler:
            all_features = [
                TimeUtils.get_cyclical_time_features(dt) for dt in timestamps
            ]

        assert len(all_features) == 168
        # Should process a week of timestamps quickly
        assert profiler.duration_seconds < 0.1

        # Verify all features are valid
        for features in all_features:
            assert len(features) == 10
            assert all(isinstance(v, float) for v in features.values())
            assert all(-1.0 <= v <= 1.0 for v in features.values())
