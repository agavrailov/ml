"""Regression tests for the IBKR bootstrap cursor-stepping logic.

Guards against the 2026-04 incident where ``scripts/ibkr_bootstrap_history.py``
stepped the cursor by ``timedelta(days=32)`` followed by ``replace(day=1)``.
Starting from a mid-month end date this produced a 2-month stride, so every
odd calendar month was never requested — the raw NVDA minute CSV was missing
~half of its history and the backtest equity curve cratered.

Each IBKR request is made with ``durationStr="1 M"``, which returns bars in
roughly ``[endDateTime - 30 days, endDateTime]``. The chunk-ending cursors
must therefore step by < 31 days so consecutive chunks cover every day.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from scripts.ibkr_bootstrap_history import (
    _MAX_CHUNK_COVERAGE,
    iter_chunk_endings,
    step_cursor_back,
)


# A single chunk covers at most this many days (IBKR "1 M" = 1 calendar month,
# up to 31 days).
_CHUNK_COVERAGE = timedelta(days=31)


def _assert_full_coverage(endings: list[datetime], start: datetime, end: datetime) -> None:
    """Every day in [start, end] must be covered by some chunk."""
    assert endings, "expected at least one chunk"
    assert endings[0] == end, "newest chunk must end at `end`"

    # Walk from newest to oldest. Each chunk covers (cursor - 30d, cursor].
    # Consecutive chunks must overlap (next cursor >= prior cursor - 30d).
    for prev, nxt in zip(endings, endings[1:]):
        gap = prev - nxt
        assert gap <= _CHUNK_COVERAGE, (
            f"stride {gap} between chunks {nxt} and {prev} exceeds "
            f"chunk coverage {_CHUNK_COVERAGE} — days would be skipped"
        )

    # Oldest chunk must reach back to `start`.
    oldest_chunk_start = endings[-1] - _CHUNK_COVERAGE
    assert oldest_chunk_start <= start, (
        f"oldest chunk start {oldest_chunk_start} does not reach requested "
        f"start {start}"
    )


@pytest.mark.parametrize(
    "end",
    [
        datetime(2026, 4, 20),   # the incident date — mid-month end
        datetime(2026, 4, 1),    # first-of-month end
        datetime(2026, 4, 15),
        datetime(2026, 2, 28),   # end of short month
        datetime(2025, 12, 31),  # end of year
    ],
)
def test_iter_chunk_endings_covers_every_day(end: datetime) -> None:
    start = datetime(2022, 1, 1)
    endings = iter_chunk_endings(start, end)
    _assert_full_coverage(endings, start, end)


def test_no_calendar_month_is_skipped_from_mid_month_start() -> None:
    """The regression case: end=2026-04-20, start=2022-01-01.

    Before the fix, iter_chunk_endings produced the sequence
    2026-04-20, 2026-03-01, 2026-01-01, 2025-11-01, ... — every odd month
    between 2022-01 and 2026-04 was never touched. This test asserts every
    (year, month) in that range is covered by at least one chunk.
    """
    end = datetime(2026, 4, 20)
    start = datetime(2022, 1, 1)
    endings = iter_chunk_endings(start, end)

    covered_months: set[tuple[int, int]] = set()
    for cursor in endings:
        chunk_start = cursor - _CHUNK_COVERAGE
        day = chunk_start
        while day <= cursor:
            covered_months.add((day.year, day.month))
            day += timedelta(days=1)

    expected: set[tuple[int, int]] = set()
    day = start
    while day <= end:
        expected.add((day.year, day.month))
        day += timedelta(days=1)

    missing = sorted(expected - covered_months)
    assert not missing, f"months never requested from IBKR: {missing}"


def test_stride_stays_under_chunk_coverage() -> None:
    """A single ``step_cursor_back`` must not skip any days."""
    cursor = datetime(2026, 4, 20)
    for _ in range(60):
        nxt = step_cursor_back(cursor)
        assert cursor - nxt <= _CHUNK_COVERAGE, (
            f"stride {cursor - nxt} from {cursor} to {nxt} would skip days"
        )
        cursor = nxt


def test_max_chunk_coverage_constant_matches_ibkr_duration() -> None:
    """Guard the constant so it cannot silently regress above 31 days."""
    assert _MAX_CHUNK_COVERAGE <= _CHUNK_COVERAGE, (
        f"_MAX_CHUNK_COVERAGE={_MAX_CHUNK_COVERAGE} must be <= IBKR '1 M' "
        f"chunk coverage {_CHUNK_COVERAGE} to avoid skipping days"
    )


def test_step_snaps_to_previous_month_end() -> None:
    """Subsequent cursor values must be last-day-of-previous-month."""
    cases = [
        (datetime(2026, 4, 21), datetime(2026, 3, 31)),  # mid-month start
        (datetime(2026, 3, 31), datetime(2026, 2, 28)),  # month-end -> prev end
        (datetime(2026, 2, 28), datetime(2026, 1, 31)),  # non-leap Feb -> Jan
        (datetime(2024, 3, 1),  datetime(2024, 2, 29)),  # leap Feb
        (datetime(2026, 1, 15), datetime(2025, 12, 31)), # crosses year boundary
    ]
    for cursor, expected in cases:
        assert step_cursor_back(cursor) == expected, (
            f"step_cursor_back({cursor}) -> {step_cursor_back(cursor)}, "
            f"expected {expected}"
        )
