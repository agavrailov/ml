import json
from pathlib import Path

import pandas as pd

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
