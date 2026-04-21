"""Tests for analyze_gaps.py trading-hours overlap logic."""
import json
import pandas as pd
import pytest

from src.analyze_gaps import _gap_overlaps_trading_hours, analyze_gaps


# ---------------------------------------------------------------------------
# _gap_overlaps_trading_hours unit tests
# ---------------------------------------------------------------------------

def test_overnight_gap_before_market_open_does_not_overlap():
    # 03:00 → 12:30 UTC same day — entirely pre-market, should not overlap
    start = pd.Timestamp("2022-04-07 03:00:00")
    end = pd.Timestamp("2022-04-07 12:30:00")
    assert not _gap_overlaps_trading_hours(start, end)


def test_gap_ending_just_before_market_open_does_not_overlap():
    # ends at exactly 13:00 UTC (= session_open boundary, exclusive)
    start = pd.Timestamp("2022-04-07 03:00:00")
    end = pd.Timestamp("2022-04-07 13:00:00")
    # overlap_end = min(13:00, 21:00) = 13:00, overlap_start = max(03:00, 13:00) = 13:00
    # overlap_end == overlap_start → not strictly greater → no overlap
    assert not _gap_overlaps_trading_hours(start, end)


def test_gap_ending_during_trading_hours_overlaps():
    # ends at 15:00 UTC (11 AM ET), well within trading hours
    start = pd.Timestamp("2022-04-07 03:00:00")
    end = pd.Timestamp("2022-04-07 15:00:00")
    assert _gap_overlaps_trading_hours(start, end)


def test_intraday_gap_during_trading_hours_overlaps():
    # Pure intraday gap: 15:00 → 16:30 UTC, entirely within trading window
    start = pd.Timestamp("2022-04-07 15:00:00")
    end = pd.Timestamp("2022-04-07 16:30:00")
    assert _gap_overlaps_trading_hours(start, end)


def test_holiday_gap_spanning_full_trading_day_overlaps():
    # Juneteenth 2024: data stops ~03:00 Jun-19 and resumes ~11:00 Jun-20.
    # Jun-19 trading window (13:00-21:00 UTC) is entirely inside this gap.
    start = pd.Timestamp("2024-06-19 03:00:00")
    end = pd.Timestamp("2024-06-20 11:00:00")
    assert _gap_overlaps_trading_hours(start, end)


def test_multi_day_gap_covers_trading_session_overlaps():
    # Gap from Mon close to Wed open — covers all of Tuesday's session
    start = pd.Timestamp("2023-01-09 21:00:00")  # Mon after close
    end = pd.Timestamp("2023-01-11 14:00:00")    # Wed pre-open
    assert _gap_overlaps_trading_hours(start, end)


# ---------------------------------------------------------------------------
# analyze_gaps integration tests (JSON output)
# ---------------------------------------------------------------------------

def _make_csv_with_gap(tmp_path, gap_start: str, gap_end: str) -> str:
    """Build a minimal minute CSV with one gap."""
    rows = []
    # 5 bars before the gap
    t = pd.Timestamp("2022-04-07 02:55:00")
    for _ in range(5):
        rows.append({"DateTime": t.isoformat(), "Open": 100, "High": 101, "Low": 99, "Close": 100})
        t += pd.Timedelta(minutes=1)
    # gap: skip to gap_end
    t = pd.Timestamp(gap_end)
    for _ in range(5):
        rows.append({"DateTime": t.isoformat(), "Open": 100, "High": 101, "Low": 99, "Close": 100})
        t += pd.Timedelta(minutes=1)
    df = pd.DataFrame(rows)
    path = str(tmp_path / "test_minute.csv")
    df.to_csv(path, index=False)
    return path


def test_analyze_gaps_overnight_gap_marked_not_overlapping(tmp_path):
    # Gap from 02:59 → 12:30 on same weekday — pre-market only
    csv_path = _make_csv_with_gap(tmp_path, "2022-04-07 02:59:00", "2022-04-07 12:30:00")
    out_json = str(tmp_path / "gaps.json")
    analyze_gaps(csv_path, out_json)
    with open(out_json) as f:
        gaps = json.load(f)
    assert len(gaps) == 1
    assert gaps[0]["overlaps_trading_hours"] is False


def test_analyze_gaps_holiday_gap_marked_overlapping(tmp_path):
    # Multi-day gap spanning Juneteenth — overlaps trading hours on Jun-19
    rows = []
    t = pd.Timestamp("2024-06-18 20:00:00")
    for _ in range(5):
        rows.append({"DateTime": t.isoformat(), "Open": 100, "High": 101, "Low": 99, "Close": 100})
        t += pd.Timedelta(minutes=1)
    t = pd.Timestamp("2024-06-20 14:00:00")
    for _ in range(5):
        rows.append({"DateTime": t.isoformat(), "Open": 100, "High": 101, "Low": 99, "Close": 100})
        t += pd.Timedelta(minutes=1)
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "holiday.csv")
    df.to_csv(csv_path, index=False)
    out_json = str(tmp_path / "gaps.json")
    analyze_gaps(csv_path, out_json)
    with open(out_json) as f:
        gaps = json.load(f)
    assert len(gaps) == 1
    assert gaps[0]["overlaps_trading_hours"] is True
