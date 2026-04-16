from __future__ import annotations

from pathlib import Path

from src import config
from src.config import get_hourly_data_csv_path, get_scaler_params_json_path


def test_hourly_csv_default_symbol_is_nvda() -> None:
    path = get_hourly_data_csv_path("60min")
    assert "nvda_60min" in path.lower()


def test_hourly_csv_custom_symbol() -> None:
    path = get_hourly_data_csv_path("60min", symbol="MSFT")
    assert "msft_60min" in path.lower()
    assert "nvda" not in path.lower()


def test_scaler_params_custom_symbol() -> None:
    path = get_scaler_params_json_path("60min", symbol="MSFT")
    assert "msft" in path.lower()


def test_backtest_path_helpers_use_backtests_dir_and_lowercase() -> None:
    """Ensure the backtest CSV helpers resolve into BASE_DIR/backtests with lowercase symbol.

    The assertions are written in a path-separator-agnostic way so they work on
    both POSIX and Windows.
    """
    base = Path(config.BASE_DIR)

    pred = Path(config.get_predictions_csv_path("NVDA", "15min"))
    trades = Path(config.get_trades_csv_path("NVDA", "15min"))
    equity = Path(config.get_equity_csv_path("NVDA", "15min"))

    # Filenames should be correct and lowercase.
    assert pred.name == "nvda_15min_predictions.csv"
    assert trades.name == "nvda_15min_trades.csv"
    assert equity.name == "nvda_15min_equity.csv"

    # The parent directory should be a "backtests" directory under BASE_DIR.
    assert pred.parent.name == "backtests"
    assert trades.parent.name == "backtests"
    assert equity.parent.name == "backtests"

    assert base in pred.parents
    assert base in trades.parents
    assert base in equity.parents
