import json
from pathlib import Path

import pandas as pd

from src import config as cfg
import src.app as app


def test_load_strategy_defaults_matches_config_module() -> None:
    """_load_strategy_defaults should reflect current values from src.config.

    This indirectly validates that the helper reloads the module and returns
    the four strategy parameters we care about.
    """

    defaults = app._load_strategy_defaults()

    assert set(defaults.keys()) == {
        "k_sigma_err",
        "k_atr_min_tp",
        "risk_per_trade_pct",
        "reward_risk_ratio",
    }

    assert defaults["k_sigma_err"] == float(cfg.K_SIGMA_ERR)
    assert defaults["k_atr_min_tp"] == float(cfg.K_ATR_MIN_TP)
    assert defaults["risk_per_trade_pct"] == float(cfg.RISK_PER_TRADE_PCT)
    assert defaults["reward_risk_ratio"] == float(cfg.REWARD_RISK_RATIO)


def test_build_default_params_df_structure() -> None:
    """Default parameter grid has one row per parameter with expected columns."""

    defaults = {
        "k_sigma_err": 1.1,
        "k_atr_min_tp": 2.2,
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
        "k_sigma_err",
        "k_atr_min_tp",
        "risk_per_trade_pct",
        "reward_risk_ratio",
    ]

    row_sigma = df[df["Parameter"] == "k_sigma_err"].iloc[0]
    assert row_sigma["Value"] == defaults["k_sigma_err"]
    assert row_sigma["Start"] == defaults["k_sigma_err"]
    assert row_sigma["Stop"] == defaults["k_sigma_err"]
    # Pandas may store this as a numpy.bool_, so coerce to Python bool.
    assert bool(row_sigma["Optimize"]) is True

    row_risk = df[df["Parameter"] == "risk_per_trade_pct"].iloc[0]
    assert row_risk["Step"] == 0.001
    # Coerce to Python bool to avoid numpy.bool_ identity issues.
    assert bool(row_risk["Optimize"]) is False


def test_load_params_grid_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    """When the JSON sidecar is missing, _load_params_grid returns defaults."""

    # Point PARAMS_STATE_PATH at a temp location that does not exist.
    monkeypatch.setattr(app, "PARAMS_STATE_PATH", tmp_path / "ui_strategy_params.json")

    defaults = {
        "k_sigma_err": 1.0,
        "k_atr_min_tp": 2.0,
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
        {"Parameter": "k_sigma_err", "Value": 9.9},
        # Other parameters will be picked up from defaults only.
    ]
    state_path.write_text(json.dumps(records), encoding="utf-8")

    defaults = {
        "k_sigma_err": 1.0,
        "k_atr_min_tp": 2.0,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.0,
    }

    df = app._load_params_grid(defaults)

    # All required columns present
    assert set(df.columns) == {"Parameter", "Value", "Start", "Step", "Stop", "Optimize"}

    # k_sigma_err value comes from saved file; other fields from defaults.
    row_sigma = df[df["Parameter"] == "k_sigma_err"].iloc[0]
    assert row_sigma["Value"] == 9.9
    assert row_sigma["Start"] == defaults["k_sigma_err"]
    assert row_sigma["Stop"] == defaults["k_sigma_err"]

    # A parameter that was not in the file falls back entirely to defaults.
    row_atr = df[df["Parameter"] == "k_atr_min_tp"].iloc[0]
    assert row_atr["Value"] == defaults["k_atr_min_tp"]


def test_save_and_reload_params_grid_round_trip(tmp_path: Path, monkeypatch) -> None:
    """_save_params_grid writes JSON that _load_params_grid can read back."""

    state_path = tmp_path / "ui_strategy_params.json"
    monkeypatch.setattr(app, "PARAMS_STATE_PATH", state_path)

    defaults = {
        "k_sigma_err": 1.0,
        "k_atr_min_tp": 2.0,
        "risk_per_trade_pct": 0.01,
        "reward_risk_ratio": 3.0,
    }
    df = app._build_default_params_df(defaults)

    # Modify one value to ensure we are not just seeing defaults.
    df.loc[df["Parameter"] == "k_sigma_err", "Value"] = 4.2

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
    k_sigma_err: float = 1.5
    k_atr_min_tp: float = 3.0
"""
    cfg_path = tmp_path / "config.py"
    cfg_path.write_text(config_text, encoding="utf-8")

    monkeypatch.setattr(app, "CONFIG_PATH", cfg_path)

    app._save_strategy_defaults_to_config(
        risk_per_trade_pct=0.02,
        reward_risk_ratio=3.5,
        k_sigma_err=2.5,
        k_atr_min_tp=4.0,
    )

    updated = cfg_path.read_text(encoding="utf-8")

    assert "risk_per_trade_pct: float = 0.02" in updated
    assert "reward_risk_ratio: float = 3.5" in updated
    assert "k_sigma_err: float = 2.5" in updated
    assert "k_atr_min_tp: float = 4.0" in updated
