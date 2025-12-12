import json
from pathlib import Path

import pandas as pd

from src import config as cfg
import src.app as app


def test_load_strategy_defaults_matches_config_module() -> None:
    """_load_strategy_defaults should reflect current values from src.config.

    This validates that the helper reloads the module and returns both the
    shared and long/short-specific strategy parameters we care about.
    """

    defaults = app._load_strategy_defaults()

    expected_keys = {
        "k_sigma_err",
        "k_atr_min_tp",
        "k_sigma_long",
        "k_sigma_short",
        "k_atr_long",
        "k_atr_short",
        "risk_per_trade_pct",
        "reward_risk_ratio",
    }
    assert set(defaults.keys()) == expected_keys

    # Shared aliases
    assert defaults["k_sigma_err"] == float(cfg.K_SIGMA_ERR)
    assert defaults["k_atr_min_tp"] == float(cfg.K_ATR_MIN_TP)
    assert defaults["risk_per_trade_pct"] == float(cfg.RISK_PER_TRADE_PCT)
    assert defaults["reward_risk_ratio"] == float(cfg.REWARD_RISK_RATIO)

    # Long/short-specific defaults (fall back to shared when dedicated names are
    # not present in config).
    exp_k_sigma_long = float(getattr(cfg, "K_SIGMA_LONG", cfg.K_SIGMA_ERR))
    exp_k_sigma_short = float(getattr(cfg, "K_SIGMA_SHORT", cfg.K_SIGMA_ERR))
    exp_k_atr_long = float(getattr(cfg, "K_ATR_LONG", cfg.K_ATR_MIN_TP))
    exp_k_atr_short = float(getattr(cfg, "K_ATR_SHORT", cfg.K_ATR_MIN_TP))

    assert defaults["k_sigma_long"] == exp_k_sigma_long
    assert defaults["k_sigma_short"] == exp_k_sigma_short
    assert defaults["k_atr_long"] == exp_k_atr_long
    assert defaults["k_atr_short"] == exp_k_atr_short


def test_build_default_params_df_structure() -> None:
    """Default parameter grid has one row per parameter with expected columns."""

    defaults = {
        "k_sigma_long": 1.1,
        "k_sigma_short": 1.2,
        "k_atr_long": 2.1,
        "k_atr_short": 2.2,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.3,
    }

    df = app._build_default_params_df(defaults)

    assert list(df.columns) == [
        "Parameter",
        "Value",
        "Start",
        "Step",
        "Stop",
        "Optimize",
    ]

    # Preserve ordering and defaults
    assert list(df["Parameter"]) == [
        "k_sigma_long",
        "k_sigma_short",
        "k_atr_long",
        "k_atr_short",
        "risk_per_trade_pct",
        "reward_risk_ratio",
    ]

    row_sigma_long = df[df["Parameter"] == "k_sigma_long"].iloc[0]
    assert row_sigma_long["Value"] == defaults["k_sigma_long"]
    assert row_sigma_long["Start"] == defaults["k_sigma_long"]
    assert row_sigma_long["Stop"] == defaults["k_sigma_long"]
    # Pandas may store this as a numpy.bool_, so coerce to Python bool.
    assert bool(row_sigma_long["Optimize"]) is True

    row_risk = df[df["Parameter"] == "risk_per_trade_pct"].iloc[0]
    assert row_risk["Step"] == 0.001
    # Coerce to Python bool to avoid numpy.bool_ identity issues.
    assert bool(row_risk["Optimize"]) is False


def test_load_params_grid_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    """When the JSON sidecar is missing, _load_params_grid returns defaults."""

    # Point PARAMS_STATE_PATH at a temp location that does not exist.
    monkeypatch.setattr(app, "PARAMS_STATE_PATH", tmp_path / "ui_strategy_params.json")

    defaults = {
        "k_sigma_long": 1.0,
        "k_sigma_short": 1.1,
        "k_atr_long": 2.0,
        "k_atr_short": 2.1,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.0,
    }

    df = app._load_params_grid(defaults)
    expected = app._build_default_params_df(defaults)

    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


def test_load_params_grid_merges_missing_columns(tmp_path: Path, monkeypatch) -> None:
    """Old state files with missing columns are merged with defaults safely."""

    state_path = tmp_path / "ui_strategy_params.json"
    monkeypatch.setattr(app, "PARAMS_STATE_PATH", state_path)

    # Simulate an older file that only had Parameter and Value for one row.
    records = [
        {"Parameter": "k_sigma_long", "Value": 9.9},
        # Other parameters will be picked up from defaults only.
    ]
    state_path.write_text(json.dumps(records), encoding="utf-8")

    defaults = {
        "k_sigma_long": 1.0,
        "k_sigma_short": 1.1,
        "k_atr_long": 2.0,
        "k_atr_short": 2.1,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.0,
    }

    df = app._load_params_grid(defaults)

    # All required columns present
    assert set(df.columns) == {"Parameter", "Value", "Start", "Step", "Stop", "Optimize"}

    # k_sigma_long value comes from saved file; other fields from defaults.
    row_sigma = df[df["Parameter"] == "k_sigma_long"].iloc[0]
    assert row_sigma["Value"] == 9.9
    assert row_sigma["Start"] == defaults["k_sigma_long"]
    assert row_sigma["Stop"] == defaults["k_sigma_long"]

    # A parameter that was not in the file falls back entirely to defaults.
    row_atr = df[df["Parameter"] == "k_atr_short"].iloc[0]
    assert row_atr["Value"] == defaults["k_atr_short"]


def test_save_and_reload_params_grid_round_trip(tmp_path: Path, monkeypatch) -> None:
    """_save_params_grid writes JSON that _load_params_grid can read back."""

    state_path = tmp_path / "ui_strategy_params.json"
    monkeypatch.setattr(app, "PARAMS_STATE_PATH", state_path)

    defaults = {
        "k_sigma_long": 1.0,
        "k_sigma_short": 1.1,
        "k_atr_long": 2.0,
        "k_atr_short": 2.1,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.0,
    }
    df = app._build_default_params_df(defaults)

    # Modify one value to ensure we are not just seeing defaults.
    df.loc[df["Parameter"] == "k_sigma_long", "Value"] = 4.2

    app._save_params_grid(df)

    # Now load via the public loader and ensure we get the same grid back.
    reloaded = app._load_params_grid(defaults)
    pd.testing.assert_frame_equal(reloaded.reset_index(drop=True), df.reset_index(drop=True))


def test_save_strategy_defaults_to_config_overwrites_literals(tmp_path: Path, monkeypatch) -> None:
    """_save_strategy_defaults_to_config replaces only numeric literals in CONFIG_PATH."""

    config_text = """
class StrategyDefaultsConfig:
    risk_per_trade_pct: float = 0.01
    reward_risk_ratio: float = 2.0
    k_sigma_long: float = 1.5
    k_sigma_short: float = 1.6
    k_atr_long: float = 3.0
    k_atr_short: float = 3.5
"""
    cfg_path = tmp_path / "config.py"
    cfg_path.write_text(config_text, encoding="utf-8")

    monkeypatch.setattr(app, "CONFIG_PATH", cfg_path)

    app._save_strategy_defaults_to_config(
        risk_per_trade_pct=0.02,
        reward_risk_ratio=3.5,
        k_sigma_long=2.5,
        k_sigma_short=2.6,
        k_atr_long=4.0,
        k_atr_short=4.5,
    )

    updated = cfg_path.read_text(encoding="utf-8")

    assert "risk_per_trade_pct: float = 0.02" in updated
    assert "reward_risk_ratio: float = 3.5" in updated
    assert "k_sigma_long: float = 2.5" in updated
    assert "k_sigma_short: float = 2.6" in updated
    assert "k_atr_long: float = 4.0" in updated
    assert "k_atr_short: float = 4.5" in updated


def test_run_backtest_uses_csv_predictions(monkeypatch, tmp_path: Path) -> None:
    """_run_backtest should call run_backtest_for_ui with CSV predictions path.

    This validates that the helper wires frequency, date range, risk/ATR/sigma
    parameters, and long/short flags into ``run_backtest_for_ui`` and that it
    always uses ``prediction_mode="csv"`` with the path from
    ``get_predictions_csv_path("nvda", frequency)``.
    """

    # Arrange: fake predictions CSV path and capture calls to the engine.
    fake_csv_path = tmp_path / "nvda_15min_predictions.csv"
    # Newer UI code validates that the predictions CSV exists.
    fake_csv_path.write_text("Time,predicted_price\n", encoding="utf-8")

    def fake_get_predictions_csv_path(symbol: str, frequency: str) -> Path:
        # Ensure the helper always uses the NVDA symbol and the provided freq.
        assert symbol == "nvda"
        assert frequency == "15min"
        return fake_csv_path

    calls: dict = {}

    def fake_run_backtest_for_ui(**kwargs):  # type: ignore[override]
        calls["kwargs"] = kwargs
        return "sentinel-result"

    monkeypatch.setattr(app, "run_backtest_for_ui", fake_run_backtest_for_ui)
    monkeypatch.setattr(
        cfg,
        "get_predictions_csv_path",
        fake_get_predictions_csv_path,
    )

    # Act: call the helper with a representative set of arguments.
    result = app._run_backtest(
        frequency="15min",
        start_date="2020-01-01",
        end_date="2020-12-31",
        risk_per_trade_pct=0.01,
        reward_risk_ratio=3.0,
        k_sigma_long=1.1,
        k_sigma_short=1.2,
        k_atr_long=2.1,
        k_atr_short=2.2,
        enable_longs=True,
        allow_shorts=False,
    )

    # Assert: return value is forwarded from the underlying engine.
    assert result == "sentinel-result"

    # And the engine was called exactly once with the expected wiring.
    kwargs = calls["kwargs"]
    assert kwargs["frequency"] == "15min"
    assert kwargs["prediction_mode"] == "csv"
    assert kwargs["predictions_csv"] == fake_csv_path
    assert kwargs["start_date"] == "2020-01-01"
    assert kwargs["end_date"] == "2020-12-31"
    assert kwargs["risk_per_trade_pct"] == 0.01
    assert kwargs["reward_risk_ratio"] == 3.0
    assert kwargs["k_sigma_long"] == 1.1
    assert kwargs["k_sigma_short"] == 1.2
    assert kwargs["k_atr_long"] == 2.1
    assert kwargs["k_atr_short"] == 2.2
    assert kwargs["enable_longs"] is True
    assert kwargs["allow_shorts"] is False

    # Legacy aggregate parameters are not used in the app helper.
    assert kwargs["k_sigma_err"] is None
    assert kwargs["k_atr_min_tp"] is None
