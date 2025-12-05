from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_dummy_ohlc_csv(base_dir: Path, frequency: str = "60min") -> None:
    """Create a small OHLC CSV under base_dir/data/processed matching config expectations."""
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    times = pd.date_range("2023-01-01", periods=10, freq="H")
    df = pd.DataFrame(
        {
            "Time": times,
            "Open": np.linspace(100.0, 101.0, len(times)),
            "High": np.linspace(100.5, 101.5, len(times)),
            "Low": np.linspace(99.5, 100.5, len(times)),
            "Close": np.linspace(100.2, 101.2, len(times)),
        }
    )

    csv_path = processed_dir / f"nvda_{frequency}.csv"
    df.to_csv(csv_path, index=False)


def test_backtest_cli_model_mode_smoke(tmp_path: Path) -> None:
    """Smoke test: run src.backtest CLI in model mode against synthetic data."""
    # 1) Create an isolated fake BASE_DIR with synthetic OHLC data.
    fake_base_dir = tmp_path / "ml_lstm_base"
    fake_base_dir.mkdir()
    _write_dummy_ohlc_csv(fake_base_dir, frequency="60min")

    # 2) Compute the real project root (directory that contains src/ and tests/).
    project_root = Path(__file__).resolve().parents[1]

    # 3) Run the backtest CLI in a subprocess, pointing ML_LSTM_BASE_DIR at fake_base_dir.
    env = os.environ.copy()
    env["ML_LSTM_BASE_DIR"] = str(fake_base_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.backtest",
        "--frequency",
        "60min",
        "--prediction-mode",
        "model",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    # If the CLI fails solely because no trained model is available, skip the test
    # rather than treating it as a hard failure. This keeps the smoke test usable
    # in fresh environments before any training has been run.
    if "No best model found for frequency" in (result.stderr or ""):
        pytest.skip("Skipping CLI smoke test: no trained model available for model mode")

    # If the CLI fails for any other reason, surface stdout/stderr in the assertion message.
    assert result.returncode == 0, (
        f"src.backtest CLI failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
