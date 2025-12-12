import json
from pathlib import Path

import pandas as pd
import pytest

import src.app as app


def test_load_json_history_missing_file_returns_empty(tmp_path: Path, monkeypatch) -> None:
    """_load_json_history should return [] when the file does not exist."""

    # Point UI_STATE_DIR at a temporary directory with no history files.
    monkeypatch.setattr(app, "UI_STATE_DIR", tmp_path)

    history = app._load_json_history("backtests_history.json")
    assert history == []


def test_save_and_load_json_history_round_trip(tmp_path: Path, monkeypatch) -> None:
    """_save_json_history writes JSON that _load_json_history can read back."""

    monkeypatch.setattr(app, "UI_STATE_DIR", tmp_path)

    rows = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "frequency": "60min",
            "trade_side": "Long only",
            "total_return": 0.5,
            "cagr": 0.4,
            "max_drawdown": -0.2,
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "n_trades": 42,
            "final_equity": 150_000.0,
        }
    ]

    app._save_json_history("backtests_history.json", rows)

    path = tmp_path / "backtests_history.json"
    assert path.exists()

    loaded = app._load_json_history("backtests_history.json")
    assert loaded == rows


def test_save_json_history_truncates_to_max_history_rows(tmp_path: Path, monkeypatch) -> None:
    """_save_json_history should keep only the most recent MAX_HISTORY_ROWS entries."""

    monkeypatch.setattr(app, "UI_STATE_DIR", tmp_path)

    # Force a small cap so the test is cheap.
    monkeypatch.setattr(app, "MAX_HISTORY_ROWS", 3)

    rows = [{"idx": i} for i in range(10)]  # 0..9

    app._save_json_history("backtests_history.json", rows)

    loaded = app._load_json_history("backtests_history.json")
    # Expect only the last 3 rows (indices 7, 8, 9).
    assert [r["idx"] for r in loaded] == [7, 8, 9]


def test_filter_backtest_history_by_frequency_sorts_desc() -> None:
    rows = [
        {"timestamp": "2025-01-01T00:00:00", "frequency": "1min", "n_trades": 1},
        {"timestamp": "2025-01-02T00:00:00", "frequency": "60min", "n_trades": 2},
        {"timestamp": "2025-01-03T00:00:00", "frequency": "1min", "n_trades": 3},
    ]

    filtered = app._filter_backtest_history(rows, frequency="1min")
    assert [r["timestamp"] for r in filtered] == ["2025-01-03T00:00:00", "2025-01-01T00:00:00"]


def test_run_backtest_raises_when_predictions_csv_missing(monkeypatch) -> None:
    # Force config.get_predictions_csv_path to return a known path.
    import src.config as cfg

    monkeypatch.setattr(cfg, "get_predictions_csv_path", lambda symbol, frequency: "C:/tmp/missing_predictions.csv")
    monkeypatch.setattr(app.os.path, "exists", lambda p: False)

    with pytest.raises(FileNotFoundError):
        app._run_backtest(
            frequency="1min",
            start_date=None,
            end_date=None,
            risk_per_trade_pct=0.01,
            reward_risk_ratio=2.0,
            k_sigma_long=1.0,
            k_sigma_short=1.0,
            k_atr_long=1.0,
            k_atr_short=1.0,
            enable_longs=True,
            allow_shorts=False,
        )
