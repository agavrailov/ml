"""Tests for src.live.poll_loop (connect-on-demand architecture).

Covers:
- Bar boundary calculation (core scheduling logic)
- Frequency parsing
- Edge cases: midnight rollover, just-past-boundary, DST-naive fallback
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.live.poll_loop import (
    _STALE_BAR_THRESHOLD_SECS,
    _frequency_to_bar_size_setting,
    _frequency_to_minutes,
    _is_market_hours,
    _is_trading_day,
    compute_next_bar_boundary,
)


# ---------------------------------------------------------------------------
# _frequency_to_minutes
# ---------------------------------------------------------------------------


class TestFrequencyToMinutes:
    def test_standard_frequencies(self):
        assert _frequency_to_minutes("1min") == 1
        assert _frequency_to_minutes("5min") == 5
        assert _frequency_to_minutes("15min") == 15
        assert _frequency_to_minutes("30min") == 30
        assert _frequency_to_minutes("60min") == 60
        assert _frequency_to_minutes("240min") == 240

    def test_hour_aliases(self):
        assert _frequency_to_minutes("1h") == 60
        assert _frequency_to_minutes("1hr") == 60
        assert _frequency_to_minutes("4h") == 240
        assert _frequency_to_minutes("4hr") == 240

    def test_case_insensitive(self):
        assert _frequency_to_minutes("60MIN") == 60
        assert _frequency_to_minutes("4H") == 240

    def test_whitespace_tolerant(self):
        assert _frequency_to_minutes(" 60 min ") == 60

    def test_unsupported_returns_none(self):
        assert _frequency_to_minutes("weekly") is None
        assert _frequency_to_minutes("") is None
        assert _frequency_to_minutes("abc") is None


# ---------------------------------------------------------------------------
# _frequency_to_bar_size_setting
# ---------------------------------------------------------------------------


class TestFrequencyToBarSizeSetting:
    def test_known_frequencies(self):
        assert _frequency_to_bar_size_setting("60min") == "1 hour"
        assert _frequency_to_bar_size_setting("240min") == "4 hours"
        assert _frequency_to_bar_size_setting("5min") == "5 mins"
        assert _frequency_to_bar_size_setting("1min") == "1 min"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported frequency"):
            _frequency_to_bar_size_setting("weekly")


# ---------------------------------------------------------------------------
# compute_next_bar_boundary
# ---------------------------------------------------------------------------


class TestComputeNextBarBoundary:
    """Test bar boundary alignment with explicit 'now' values.

    All times use a fixed UTC offset to avoid relying on system timezone.
    """

    # Use a fixed Eastern-like offset for deterministic tests
    _EST = timezone(timedelta(hours=-5))

    def test_60min_mid_hour(self):
        """At 14:35 ET, next 60min bar boundary = 15:00 + 2 min buffer = 15:02."""
        now = datetime(2026, 2, 26, 14, 35, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 15, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_60min_on_boundary(self):
        """Exactly at 15:00 ET should target 16:00 + buffer."""
        now = datetime(2026, 2, 26, 15, 0, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 16, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_60min_just_past_boundary(self):
        """At 15:00:01 ET, should still target 16:02."""
        now = datetime(2026, 2, 26, 15, 0, 1, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 16, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_240min_morning(self):
        """At 10:30 ET with 240min bars, next boundary = 12:00 + 2 min = 12:02."""
        now = datetime(2026, 2, 26, 10, 30, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("240min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 12, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_240min_late_afternoon(self):
        """At 15:30 ET with 240min bars, next boundary = 16:00 + buffer."""
        now = datetime(2026, 2, 26, 15, 30, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("240min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 16, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_midnight_rollover(self):
        """At 23:30 ET with 60min bars, next boundary = 00:00 next day + buffer."""
        now = datetime(2026, 2, 26, 23, 30, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 27, 0, 2, 0, tzinfo=self._EST)
        assert result == expected

    def test_buffer_zero(self):
        """Buffer=0 should return the exact boundary."""
        now = datetime(2026, 2, 26, 14, 35, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=0)
        expected = datetime(2026, 2, 26, 15, 0, 0, tzinfo=self._EST)
        assert result == expected

    def test_5min_bars(self):
        """At 14:33 ET with 5min bars, next = 14:35 + buffer."""
        now = datetime(2026, 2, 26, 14, 33, 0, tzinfo=self._EST)
        result = compute_next_bar_boundary("5min", now, buffer_minutes=2)
        expected = datetime(2026, 2, 26, 14, 37, 0, tzinfo=self._EST)
        assert result == expected

    def test_unsupported_frequency_raises(self):
        now = datetime(2026, 2, 26, 14, 0, 0, tzinfo=self._EST)
        with pytest.raises(ValueError, match="Unsupported frequency"):
            compute_next_bar_boundary("weekly", now)

    def test_result_is_always_in_future(self):
        """Result must always be strictly after 'now'."""
        now = datetime(2026, 2, 26, 14, 59, 59, tzinfo=self._EST)
        result = compute_next_bar_boundary("60min", now, buffer_minutes=0)
        assert result > now

    def test_naive_datetime_treated_as_eastern(self):
        """A naive datetime should be treated as Eastern time."""
        now_naive = datetime(2026, 2, 26, 14, 35, 0)  # no tzinfo
        result = compute_next_bar_boundary("60min", now_naive, buffer_minutes=2)
        # Should produce 15:02 in whatever tz it interprets as Eastern
        assert result.hour == 15
        assert result.minute == 2


# ---------------------------------------------------------------------------
# _is_market_hours / _is_trading_day with `at` parameter
# ---------------------------------------------------------------------------


class TestIsMarketHoursAt:
    _EST = timezone(timedelta(hours=-5))

    def test_during_regular_hours(self):
        at = datetime(2026, 2, 26, 10, 30, tzinfo=self._EST)  # Thu 10:30
        assert _is_market_hours(premarket=True, at=at) is True

    def test_after_close(self):
        at = datetime(2026, 2, 26, 17, 0, tzinfo=self._EST)  # Thu 17:00
        assert _is_market_hours(premarket=True, at=at) is False

    def test_premarket_4am(self):
        at = datetime(2026, 2, 26, 4, 0, tzinfo=self._EST)  # Thu 04:00
        assert _is_market_hours(premarket=True, at=at) is True

    def test_before_premarket(self):
        at = datetime(2026, 2, 26, 3, 59, tzinfo=self._EST)  # Thu 03:59
        assert _is_market_hours(premarket=True, at=at) is False

    def test_weekend_returns_false(self):
        at = datetime(2026, 2, 28, 10, 0, tzinfo=self._EST)  # Sat 10:00
        assert _is_market_hours(premarket=True, at=at) is False

    def test_no_at_uses_now(self):
        # Should not raise (backward compat)
        _is_market_hours(premarket=True)


class TestIsTradingDayAt:
    _EST = timezone(timedelta(hours=-5))

    def test_weekday(self):
        at = datetime(2026, 2, 26, 12, 0, tzinfo=self._EST)  # Thursday
        assert _is_trading_day(at=at) is True

    def test_saturday(self):
        at = datetime(2026, 2, 28, 12, 0, tzinfo=self._EST)  # Saturday
        assert _is_trading_day(at=at) is False

    def test_sunday(self):
        at = datetime(2026, 3, 1, 12, 0, tzinfo=self._EST)  # Sunday
        assert _is_trading_day(at=at) is False

    def test_no_at_uses_now(self):
        # Should not raise (backward compat)
        _is_trading_day()


class TestSleepWakeBarProcessing:
    """Regression: bars from market hours must NOT be skipped after PC wake."""

    _EST = timezone(timedelta(hours=-5))

    def test_bar_at_1302_wake_at_1800_should_not_skip(self):
        """Target 13:02 EST (during market), PC wakes at 18:00 EST.

        Old code checked current time -> skipped.  New code checks target -> runs.
        """
        target = datetime(2026, 2, 26, 13, 2, tzinfo=self._EST)  # Thu 13:02
        assert _is_trading_day(at=target) is True
        assert _is_market_hours(premarket=True, at=target) is True

    def test_bar_at_2202_should_skip(self):
        """Target 22:02 EST — well outside market hours, should skip."""
        target = datetime(2026, 2, 26, 22, 2, tzinfo=self._EST)
        assert _is_market_hours(premarket=True, at=target) is False

    def test_bar_on_saturday_should_skip(self):
        target = datetime(2026, 2, 28, 13, 2, tzinfo=self._EST)  # Saturday
        assert _is_trading_day(at=target) is False

    def test_staleness_suppresses_trading_not_cycle(self):
        """Overshoot > threshold -> cycle runs but trading suppressed."""
        target = datetime(2026, 2, 26, 13, 2, tzinfo=self._EST)  # Thu 13:02
        wake = datetime(2026, 2, 26, 18, 0, tzinfo=self._EST)    # Thu 18:00
        overshoot = (wake - target).total_seconds()
        # Bar was during market hours -> cycle should NOT be skipped
        assert _is_trading_day(at=target) is True
        assert _is_market_hours(premarket=True, at=target) is True
        # But overshoot exceeds threshold -> trading should be suppressed
        assert overshoot > _STALE_BAR_THRESHOLD_SECS

    def test_small_overshoot_allows_trading(self):
        """Overshoot < threshold -> trading proceeds normally."""
        target = datetime(2026, 2, 26, 13, 2, tzinfo=self._EST)
        wake = datetime(2026, 2, 26, 13, 4, tzinfo=self._EST)  # 2 min late
        overshoot = (wake - target).total_seconds()
        assert overshoot < _STALE_BAR_THRESHOLD_SECS


# ---------------------------------------------------------------------------
# State enum extensions (smoke test)
# ---------------------------------------------------------------------------


class TestStateEnumExtensions:
    """Verify the new poll-loop states exist in SystemState."""

    def test_sleeping_state_exists(self):
        from src.live.state import SystemState
        assert SystemState.SLEEPING.value == "SLEEPING"

    def test_processing_state_exists(self):
        from src.live.state import SystemState
        assert SystemState.PROCESSING.value == "PROCESSING"

    def test_error_state_exists(self):
        from src.live.state import SystemState
        assert SystemState.ERROR.value == "ERROR"

    def test_original_states_still_exist(self):
        from src.live.state import SystemState
        assert SystemState.TRADING.value == "TRADING"
        assert SystemState.RECONNECTING.value == "RECONNECTING"
        assert SystemState.STOPPED.value == "STOPPED"
