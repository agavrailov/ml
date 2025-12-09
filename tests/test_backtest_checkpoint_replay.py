from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as backtest_mod
from src.data import load_hourly_ohlc


def test_model_checkpoint_csv_replay_matches_model_mode(tmp_path: Path) -> None:
    """Replaying the model prediction checkpoint via CSV-mode should match
    model-mode when using the same OHLC slice.

    This is an integration-level invariant: given the same data, prediction
    prices, and per-bar sigma series, model-mode and CSV-mode replays of the
    checkpoint must produce identical equity curves (up to float noise).
    """

    freq = "60min"

    # Load full hourly OHLC data and restrict to a manageable tail slice to
    # keep the test runtime reasonable while still exercising realistic data.
    data_full = load_hourly_ohlc(freq)
    assert not data_full.empty
    data = data_full.tail(2000).reset_index(drop=True)

    # Ensure we do not accidentally reuse an old checkpoint from a different
    # slice. The backtest code will recreate it on the first model-mode run.
    checkpoint_path = Path("backtests") / f"nvda_{freq}_model_predictions_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # 1) Run model-mode backtest, which will also write the checkpoint with
    # per-bar predictions and model_error_sigma to disk.
    result_model = backtest_mod.run_backtest_on_dataframe(
        data,
        initial_equity=10_000.0,
        frequency=freq,
        prediction_mode="model",
        commission_per_unit_per_leg=0.0,
        min_commission_per_order=0.0,
    )

    assert checkpoint_path.exists(), "Model backtest did not write checkpoint CSV as expected."

    # 2) Replay the same checkpoint via CSV-mode on the *same* data slice.
    result_csv = backtest_mod.run_backtest_on_dataframe(
        data,
        initial_equity=10_000.0,
        frequency=freq,
        prediction_mode="csv",
        predictions_csv=str(checkpoint_path),
        commission_per_unit_per_leg=0.0,
        min_commission_per_order=0.0,
    )

    eq_model = np.asarray(result_model.equity_curve, dtype=float)
    eq_csv = np.asarray(result_csv.equity_curve, dtype=float)

    # Basic length sanity.
    assert len(eq_model) == len(eq_csv) > 0

    # Core invariant: replaying the checkpoint via CSV-mode must reproduce the
    # same equity curve as model-mode, up to tight float tolerances.
    assert np.allclose(eq_model, eq_csv, rtol=1e-9, atol=1e-9)

    # Trade counts should also match exactly.
    assert len(result_model.trades) == len(result_csv.trades)
