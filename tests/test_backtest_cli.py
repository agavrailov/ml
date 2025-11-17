"""Tests for src.backtest helper functions (not the CLI itself).

We validate that `run_backtest_on_dataframe` runs end-to-end on a small
synthetic dataset and returns a BacktestResult.
"""
from __future__ import annotations

import pandas as pd

from src.backtest import run_backtest_on_dataframe


def test_run_backtest_on_dataframe_smoke() -> None:
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.5, 101.0, 101.5],
            "High": [100.8, 101.3, 101.8, 102.1],
            "Low": [99.8, 100.2, 100.6, 101.0],
            "Close": [100.4, 100.9, 101.4, 101.9],
        }
    )

    result = run_backtest_on_dataframe(
        data,
        initial_equity=10_000.0,
        frequency="60min",
        prediction_mode="naive",
        commission_per_unit_per_leg=0.0,
        min_commission_per_order=0.0,
    )

    # Basic shape checks
    assert len(result.equity_curve) == len(data)
    # Should return a BacktestResult without raising; trades may or may not be present
    assert result.final_equity > 0
