from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    """Half-open time window [start, end) in calendar dates.

    The ``start`` and ``end`` fields are ISO date strings (``YYYY-MM-DD``) or
    ``None`` to indicate an open bound. All comparisons are done on the
    ``Time`` column in dataframes after conversion to pandas ``Timestamp``.
    """

    start: Optional[str]
    end: Optional[str]


def _to_timestamp(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if date_str is None:
        return None
    return pd.to_datetime(date_str).normalize()


def slice_df_by_window(df: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    """Return rows with ``Time`` in [start, end).

    If ``window.start`` / ``window.end`` are ``None``, the corresponding bound
    is left open.
    """

    if "Time" not in df.columns:
        raise KeyError("slice_df_by_window expects a 'Time' column in the DataFrame")

    time_series = pd.to_datetime(df["Time"])
    mask = pd.Series(True, index=df.index)

    start_ts = _to_timestamp(window.start)
    end_ts = _to_timestamp(window.end)

    if start_ts is not None:
        mask &= time_series >= start_ts
    if end_ts is not None:
        # half-open interval [start, end)
        mask &= time_series < end_ts

    return df.loc[mask].copy()


def generate_walkforward_windows(
    t_start: str,
    t_end: str,
    *,
    test_span_months: int = 3,
    train_lookback_months: int = 24,
    min_lookback_months: int = 18,
    first_test_start: Optional[str] = None,
) -> List[Tuple[TimeWindow, TimeWindow]]:
    """Generate rolling (train_window, test_window) pairs.

    Parameters
    ----------
    t_start, t_end:
        Inclusive overall data horizon. These are typically derived from the
        earliest and latest timestamps available in the underlying OHLC data.
    test_span_months:
        Length of each test window in months (default 3).
    train_lookback_months:
        Target lookback for training data before each test window (default 24).
    min_lookback_months:
        Minimum desired lookback before the first test window (default 18).
        If there is not enough history, we fall back to using whatever is
        available from ``t_start``.
    first_test_start:
        Optional explicit first test-start date (``YYYY-MM-DD``). When not
        provided, the first test start is chosen as ``t_start + min_lookback``
        months (clipped to be < ``t_end``).

    Returns
    -------
    List of (train_window, test_window) pairs. Each window is a ``TimeWindow``
    with ISO date strings.
    """

    if test_span_months <= 0:
        raise ValueError("test_span_months must be positive")
    if train_lookback_months <= 0:
        raise ValueError("train_lookback_months must be positive")

    t_start_ts = pd.to_datetime(t_start).normalize()
    t_end_ts = pd.to_datetime(t_end).normalize()
    if t_end_ts <= t_start_ts:
        return []

    # Determine the first test window start.
    if first_test_start is not None:
        test_start_ts = pd.to_datetime(first_test_start).normalize()
        if test_start_ts <= t_start_ts:
            # Ensure we start strictly after the global start.
            test_start_ts = t_start_ts + pd.DateOffset(days=1)
    else:
        # Default: require at least ~min_lookback_months of history before the
        # first test window when possible.
        candidate = t_start_ts + pd.DateOffset(months=min_lookback_months)
        test_start_ts = candidate if candidate < t_end_ts else t_end_ts

    windows: List[Tuple[TimeWindow, TimeWindow]] = []

    while test_start_ts < t_end_ts:
        test_end_ts = test_start_ts + pd.DateOffset(months=test_span_months)
        if test_end_ts > t_end_ts:
            test_end_ts = t_end_ts

        # Require a minimally useful test window (e.g. > 10 days).
        if (test_end_ts - test_start_ts).days <= 10:
            break

        train_end_ts = test_start_ts
        train_start_ts = train_end_ts - pd.DateOffset(months=train_lookback_months)
        if train_start_ts < t_start_ts:
            train_start_ts = t_start_ts

        train_w = TimeWindow(
            start=train_start_ts.strftime("%Y-%m-%d"),
            end=train_end_ts.strftime("%Y-%m-%d"),
        )
        test_w = TimeWindow(
            start=test_start_ts.strftime("%Y-%m-%d"),
            end=test_end_ts.strftime("%Y-%m-%d"),
        )
        windows.append((train_w, test_w))

        # Next fold starts immediately after this test window.
        test_start_ts = test_end_ts

    return windows


def infer_data_horizon(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (t_start, t_end) ISO dates from the Time column of df.

    ``t_start`` is the first date, ``t_end`` is the last date.
    """

    if "Time" not in df.columns or df.empty:
        raise ValueError("infer_data_horizon expects non-empty df with 'Time' column")

    times = pd.to_datetime(df["Time"])
    t_start = times.min().normalize().strftime("%Y-%m-%d")
    t_end = times.max().normalize().strftime("%Y-%m-%d")
    return t_start, t_end
