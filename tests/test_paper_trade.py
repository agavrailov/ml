"""Tests for src.paper_trade.

We validate that `run_paper_trading_over_dataframe` can run end-to-end on a
small synthetic dataset using CSV-based predictions and returns a
BacktestResult with a sane equity curve.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paper_trade import PaperTradingConfig, run_paper_trading_over_dataframe


def test_run_paper_trading_over_dataframe_with_csv_predictions(tmp_path: Path) -> None:
    # Synthetic 4-bar OHLC dataset.
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.5, 101.0, 101.5],
            "High": [100.8, 101.3, 101.8, 102.1],
            "Low": [99.8, 100.2, 100.6, 101.0],
            "Close": [100.4, 100.9, 101.4, 101.9],
        }
    )

    # Simple predictions: always Close + 1.0 to encourage long entries.
    preds = pd.DataFrame({"predicted_price": data["Close"] + 1.0})
    preds_path = tmp_path / "preds.csv"
    preds.to_csv(preds_path, index=False)

    cfg = PaperTradingConfig(initial_equity=10_000.0)

    result = run_paper_trading_over_dataframe(
        data,
        cfg=cfg,
        predictions_csv=str(preds_path),
    )

    # Basic sanity checks.
    assert len(result.equity_curve) == len(data)
    # Should not raise and should return a positive equity value.
    assert result.final_equity > 0.0


def test_run_paper_trading_requires_predictions_csv(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.5],
            "High": [100.8, 101.3],
            "Low": [99.8, 100.2],
            "Close": [100.4, 100.9],
        }
    )

    cfg = PaperTradingConfig(initial_equity=10_000.0)

    # Ensure we clearly fail if predictions_csv is omitted.
    try:
        run_paper_trading_over_dataframe(data, cfg=cfg)
    except ValueError as exc:
        assert "predictions_csv" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError when predictions_csv is not provided")
