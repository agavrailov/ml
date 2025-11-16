import pandas as pd
from datetime import datetime

from src.data_processing import fill_gaps


def _make_df(times_and_prices):
    rows = []
    for ts, price in times_and_prices:
        rows.append(
            {
                "DateTime": ts,
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price + 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_fill_gaps_small_gap_filled_forward():
    base = datetime(2024, 1, 1, 9, 30)
    # Data at 09:30 and 09:33; 09:31-09:32 should be filled.
    df = _make_df(
        [
            (base, 100.0),
            (base.replace(minute=33), 103.0),
        ]
    )

    gaps = [
        {
            "start": "2024-01-01 09:31:00",
            "end": "2024-01-01 09:32:00",
            "duration": "0 days 00:02:00",
        }
    ]

    df_filled = fill_gaps(df, gaps)

    # We expect continuous minutes from 09:30 to 09:33 inclusive.
    ts_list = list(df_filled["DateTime"])
    assert len(ts_list) == 4
    assert ts_list[0] == base
    assert ts_list[-1] == base.replace(minute=33)

    # The filled minutes (31, 32) should be forward-filled from 09:30.
    row_31 = df_filled[df_filled["DateTime"] == base.replace(minute=31)].iloc[0]
    row_32 = df_filled[df_filled["DateTime"] == base.replace(minute=32)].iloc[0]
    assert row_31["Open"] == 100.0
    assert row_32["Open"] == 100.0


def test_fill_gaps_large_gap_skipped():
    base = datetime(2024, 1, 1, 9, 30)
    df = _make_df([(base, 100.0), (base.replace(minute=40), 110.0)])

    gaps = [
        {
            "start": "2024-01-01 09:31:00",
            "end": "2024-01-01 13:31:00",
            "duration": "0 days 04:00:00",
        }
    ]

    # Use a small max_fill_hours so this gap is considered "too large".
    df_filled = fill_gaps(df, gaps, max_fill_hours=1)

    # No new rows should be added.
    assert len(df_filled) == len(df)
    assert list(df_filled["DateTime"]) == list(df["DateTime"])