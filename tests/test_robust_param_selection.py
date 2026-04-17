"""Unit tests for scripts/robust_param_selection.py.

All tests are in-memory / fixture-based. We do NOT run real backtests, touch
real CSVs, real models, or the registry.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module loader — scripts/ is not a package, so import by path.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SCRIPT_PATH = _HERE.parent / "scripts" / "robust_param_selection.py"


@pytest.fixture(scope="module")
def rps():
    spec = importlib.util.spec_from_file_location(
        "robust_param_selection", str(_SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1) LHS sampling: shape and per-dim range bounds
# ---------------------------------------------------------------------------
def test_lhs_sample_shape_and_ranges(rps):
    n = 2000
    df = rps.lhs_sample(n_samples=n, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n
    # All 6 param columns present
    for name in rps.PARAM_NAMES:
        assert name in df.columns, f"missing param column {name}"
        lo, hi = rps.PARAM_RANGES[name]
        col = df[name].astype(float)
        # With rounding the extremes can fall fractionally outside the declared
        # bounds; allow a small epsilon equal to the rounding step.
        # risk_per_trade_pct rounds to 4 dp (5e-5), others to 2-3 dp (5e-3).
        eps = 5e-3 if name != "risk_per_trade_pct" else 5e-5
        assert col.min() >= lo - eps, f"{name} min {col.min()} < {lo}"
        assert col.max() <= hi + eps, f"{name} max {col.max()} > {hi}"
    # Reproducibility: same seed → identical frame
    df2 = rps.lhs_sample(n_samples=n, seed=42)
    pd.testing.assert_frame_equal(df, df2)


# ---------------------------------------------------------------------------
# 2) maximin_select: pick_order present in BOTH branches
# ---------------------------------------------------------------------------
def _make_viable_df(n: int, rps_mod, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic viable DF covering the 6 params + an aggregate score."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {}
        for name in rps_mod.PARAM_NAMES:
            lo, hi = rps_mod.PARAM_RANGES[name]
            row[name] = float(rng.uniform(lo, hi))
        row["mean_sharpe"] = float(rng.uniform(-0.5, 1.5))
        row["oos_score"] = float(rng.uniform(-0.5, 2.0))
        rows.append(row)
    return pd.DataFrame(rows)


def test_maximin_select_pick_order_early_return_branch(rps):
    # Branch (a): len(viable) <= n → early return path
    viable = _make_viable_df(5, rps, seed=1)
    out = rps.maximin_select(viable, n=10)
    assert "pick_order" in out.columns, "early-return branch must add pick_order"
    assert list(out["pick_order"]) == list(range(1, len(out) + 1))
    assert len(out) == 5


def test_maximin_select_pick_order_full_loop_branch(rps):
    # Branch (b): len(viable) > n → maximin loop path
    viable = _make_viable_df(50, rps, seed=2)
    out = rps.maximin_select(viable, n=10)
    assert "pick_order" in out.columns, "maximin-loop branch must add pick_order"
    assert list(out["pick_order"]) == list(range(1, 11))
    assert len(out) == 10


# ---------------------------------------------------------------------------
# 3) --long-only flag propagates allow_shorts=False into _one_backtest
# ---------------------------------------------------------------------------
def test_long_only_flag_disables_shorts(rps, monkeypatch):
    """Invoke _one_backtest with long_only=True and confirm run_backtest_for_ui
    is called with allow_shorts=False, enable_longs=True."""
    captured_kwargs: dict = {}

    # The import inside _one_backtest is `from src.backtest import run_backtest_for_ui`.
    # We patch that binding on the src.backtest module so the import returns our fake.
    import src.backtest as sb

    def fake_run_backtest_for_ui(**kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        eq = pd.DataFrame({"equity": [1.0]})
        tr = pd.DataFrame()
        m = {"sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
             "total_return": 0.0, "cagr": 0.0, "win_rate": 0.0, "n_trades": 0}
        return eq, tr, m

    monkeypatch.setattr(sb, "run_backtest_for_ui", fake_run_backtest_for_ui)

    params = {
        "k_sigma_long": 0.3, "k_sigma_short": 0.3,
        "k_atr_long": 0.4, "k_atr_short": 0.4,
        "reward_risk_ratio": 2.0, "risk_per_trade_pct": 0.01,
    }

    # --- long_only=True ---
    rps._one_backtest("NVDA", "/dev/null/fake.csv", params,
                      start="2024-01-01", end="2024-04-01", long_only=True)
    assert captured_kwargs.get("enable_longs") is True, "enable_longs must stay True"
    assert captured_kwargs.get("allow_shorts") is False, \
        "long_only=True must set allow_shorts=False"

    # --- long_only=False (default) ---
    rps._one_backtest("NVDA", "/dev/null/fake.csv", params,
                      start="2024-01-01", end="2024-04-01", long_only=False)
    assert captured_kwargs.get("enable_longs") is True
    assert captured_kwargs.get("allow_shorts") is True, \
        "long_only=False must allow shorts"


def test_long_only_flag_propagates_through_stage_a(rps, monkeypatch):
    """Integration check: stage_a_prefilter must pass long_only through to
    _one_backtest (guards against someone adding a bypass path)."""
    captured_calls: list[dict] = []

    def fake_one_backtest(symbol, predictions_csv, params, start, end,
                          long_only=False):
        captured_calls.append({"long_only": long_only, "params": dict(params)})
        return {"sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "cagr": 0.0, "win_rate": 0.0, "n_trades": 0}

    monkeypatch.setattr(rps, "_one_backtest", fake_one_backtest)

    samples = rps.lhs_sample(n_samples=3, seed=0)
    rep_fold = {
        "fold_idx": 5, "preds_csv": "fake.csv",
        "test_start": "2024-01-01", "test_end": "2024-04-01",
    }
    rps.stage_a_prefilter("NVDA", rep_fold, samples, long_only=True)
    assert len(captured_calls) == 3
    assert all(c["long_only"] is True for c in captured_calls), \
        "stage_a_prefilter must forward long_only=True to _one_backtest"


# ---------------------------------------------------------------------------
# 4) rank_candidates aggregates per candidate_id
# ---------------------------------------------------------------------------
def test_rank_candidates_aggregates_per_candidate_id(rps):
    """Build a synthetic oos_long DF with 2 candidates × 3 folds each and
    verify the aggregation math matches expectations."""
    # Candidate 1: sharpe = [1.0, 0.5, -0.2] → mean=0.4333, min=-0.2, pct_pos=2/3
    # Candidate 2: sharpe = [0.1, 0.1, 0.1] → mean=0.1, min=0.1, pct_pos=3/3
    params_1 = {"k_sigma_long": 0.3, "k_sigma_short": 0.3,
                "k_atr_long": 0.4, "k_atr_short": 0.4,
                "reward_risk_ratio": 2.0, "risk_per_trade_pct": 0.01}
    params_2 = {"k_sigma_long": 0.5, "k_sigma_short": 0.2,
                "k_atr_long": 0.5, "k_atr_short": 0.3,
                "reward_risk_ratio": 2.5, "risk_per_trade_pct": 0.015}

    rows = []
    for fold_idx, sh in enumerate([1.0, 0.5, -0.2], start=1):
        rows.append({
            "fold_idx": fold_idx, "candidate_id": 1,
            **params_1,
            "oos_sharpe_ratio": sh,
            "oos_profit_factor": 1.2 + 0.1 * fold_idx,
            "oos_max_drawdown": -0.1,
            "oos_n_trades": 30,
        })
    for fold_idx, sh in enumerate([0.1, 0.1, 0.1], start=1):
        rows.append({
            "fold_idx": fold_idx, "candidate_id": 2,
            **params_2,
            "oos_sharpe_ratio": sh,
            "oos_profit_factor": 1.05,
            "oos_max_drawdown": -0.05,
            "oos_n_trades": 25,
        })
    oos_long = pd.DataFrame(rows)

    ranked = rps.rank_candidates(oos_long)

    # Shape: one row per candidate_id
    assert len(ranked) == 2
    assert set(ranked["candidate_id"]) == {1, 2}

    # Find each
    c1 = ranked[ranked["candidate_id"] == 1].iloc[0]
    c2 = ranked[ranked["candidate_id"] == 2].iloc[0]

    # Aggregates for C1
    assert c1["mean_sharpe"] == pytest.approx((1.0 + 0.5 - 0.2) / 3.0, rel=1e-6)
    assert c1["min_sharpe"]  == pytest.approx(-0.2, rel=1e-6)
    assert c1["median_sharpe"] == pytest.approx(0.5, rel=1e-6)
    assert c1["pct_pos_folds"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert c1["n_folds"] == 3
    # std_sharpe — pandas default ddof=1
    expected_std = pd.Series([1.0, 0.5, -0.2]).std()
    assert c1["std_sharpe"] == pytest.approx(expected_std, rel=1e-6)

    # Aggregates for C2
    assert c2["mean_sharpe"] == pytest.approx(0.1, rel=1e-6)
    assert c2["min_sharpe"]  == pytest.approx(0.1, rel=1e-6)
    assert c2["pct_pos_folds"] == pytest.approx(1.0, rel=1e-6)

    # Params preserved per candidate (attached from first row per group)
    for name, val in params_1.items():
        assert c1[name] == pytest.approx(val, rel=1e-6)
    for name, val in params_2.items():
        assert c2[name] == pytest.approx(val, rel=1e-6)

    # Ranking: oos_score should place C1 above C2 here
    # (C1 mean 0.433, C2 mean 0.1; both fully or mostly positive).
    # Whichever wins, the df is sorted desc by oos_score — check that ordering is monotonic.
    assert ranked["oos_score"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Extra: subset_oos_long_to_diverse is the new Phase 5 bridge — sanity check.
# ---------------------------------------------------------------------------
def test_subset_oos_long_to_diverse(rps):
    # Three candidates scored on 2 folds each; diverse set picks candidates 1 & 3.
    rows = []
    params = {k: 0.3 for k in rps.PARAM_NAMES}
    for cid in (1, 2, 3):
        for fold_idx in (1, 2):
            rows.append({
                "fold_idx": fold_idx, "candidate_id": cid,
                **params,
                "oos_sharpe_ratio": 0.1 * cid,
                "oos_profit_factor": 1.1, "oos_max_drawdown": -0.1,
                "oos_n_trades": 10,
            })
    oos_long_b = pd.DataFrame(rows)

    diverse = pd.DataFrame([
        {"pick_order": 1, "candidate_id": 1, **params},
        {"pick_order": 2, "candidate_id": 3, **params},
    ])

    subset = rps.subset_oos_long_to_diverse(oos_long_b, diverse)
    assert set(subset["candidate_id"]) == {1, 3}
    assert len(subset) == 4  # 2 candidates × 2 folds
