"""Robust strategy-parameter selection + honest model-OOS walk-forward.

Pipeline per symbol:
  Phase 0 (new): Generate 9 walk-forward folds of test-window predictions ONCE.
                 Each fold retrains a fresh model on its train slice and predicts
                 its genuinely-OOS test slice. Cached to fold01_predictions.csv..
                 fold09_predictions.csv.
  Phase 1: Latin Hypercube Sampling over a 6D parameter space.
  Phase 2 (rewritten, two-stage, honest OOS):
      Stage A: score each of N LHS candidates on a SINGLE representative fold
               (the middle fold) — cheap pre-filter against genuinely-OOS
               predictions.
      Stage B: take top 10% by Stage-A Sharpe, run a full 9-fold backtest on
               each, aggregate with the same robustness-adjusted score used in
               Phase 6.
  Phase 3: Filter viable candidates from Stage-B aggregated metrics.
  Phase 4: Select N diverse candidates via maximin farthest-point sampling.
  Phase 5 (slimmed): Subset Stage-B results to the N diverse winners — no
                     retraining needed since Stage B already scored each
                     candidate on all 9 cached folds.
  Phase 6: Rank by robustness-adjusted OOS score.
  Phase 7: (optional) Auto-promote top-1 to configs/symbols/{SYM}/active.json.

Usage (from repo root):
    PYTHONPATH=. python scripts/robust_param_selection.py --symbol MSFT --auto-promote
    PYTHONPATH=. python scripts/robust_param_selection.py --symbol NVDA --long-only --auto-promote
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


FREQUENCY = "60min"
TSTEPS = 5

PARAM_RANGES = {
    "k_sigma_long":       (0.10, 0.80),
    "k_sigma_short":      (0.10, 0.80),
    "k_atr_long":         (0.15, 0.85),
    "k_atr_short":        (0.15, 0.85),
    "reward_risk_ratio":  (1.5, 3.5),
    "risk_per_trade_pct": (0.005, 0.020),
}
PARAM_NAMES = list(PARAM_RANGES.keys())


# ---------------------------------------------------------------------------
# Phase 1: LHS sampling
# ---------------------------------------------------------------------------
def lhs_sample(n_samples: int, seed: int = 42) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n_samples)
    lows  = np.array([PARAM_RANGES[k][0] for k in PARAM_NAMES])
    highs = np.array([PARAM_RANGES[k][1] for k in PARAM_NAMES])
    scaled = qmc.scale(unit, lows, highs)
    df = pd.DataFrame(scaled, columns=PARAM_NAMES)
    # Round to sensible precision so results are reproducible and human-readable.
    df["k_sigma_long"]       = df["k_sigma_long"].round(3)
    df["k_sigma_short"]      = df["k_sigma_short"].round(3)
    df["k_atr_long"]         = df["k_atr_long"].round(3)
    df["k_atr_short"]        = df["k_atr_short"].round(3)
    df["reward_risk_ratio"]  = df["reward_risk_ratio"].round(2)
    df["risk_per_trade_pct"] = df["risk_per_trade_pct"].round(4)
    return df


# ---------------------------------------------------------------------------
# Single backtest primitive (used everywhere)
# ---------------------------------------------------------------------------
def _one_backtest(symbol: str, predictions_csv: str, params: dict,
                  start: str | None, end: str | None,
                  long_only: bool = False) -> dict:
    from src.backtest import run_backtest_for_ui
    if long_only:
        enable_longs, allow_shorts = True, False
    else:
        enable_longs, allow_shorts = True, True
    try:
        _eq, _tr, m = run_backtest_for_ui(
            symbol=symbol,
            frequency=FREQUENCY,
            prediction_mode="csv",
            predictions_csv=predictions_csv,
            start_date=start,
            end_date=end,
            k_sigma_long=float(params["k_sigma_long"]),
            k_sigma_short=float(params["k_sigma_short"]),
            k_atr_long=float(params["k_atr_long"]),
            k_atr_short=float(params["k_atr_short"]),
            risk_per_trade_pct=float(params["risk_per_trade_pct"]),
            reward_risk_ratio=float(params["reward_risk_ratio"]),
            enable_longs=enable_longs,
            allow_shorts=allow_shorts,
        )
        return {
            "sharpe_ratio":  float(m.get("sharpe_ratio", 0.0) or 0.0),
            "profit_factor": float(m.get("profit_factor", 0.0) or 0.0),
            "max_drawdown":  float(m.get("max_drawdown", 0.0) or 0.0),
            "total_return":  float(m.get("total_return", 0.0) or 0.0),
            "cagr":          float(m.get("cagr", 0.0) or 0.0),
            "win_rate":      float(m.get("win_rate", 0.0) or 0.0),
            "n_trades":      int(m.get("n_trades", 0) or 0),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "cagr": 0.0, "win_rate": 0.0, "n_trades": 0,
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Phase 0: Generate walk-forward fold predictions (ONCE, up-front)
# ---------------------------------------------------------------------------
def _get_registry_hparams(symbol: str) -> dict:
    from src.model_registry import get_best_model_entry
    entry = get_best_model_entry(symbol, FREQUENCY, TSTEPS)
    if not entry:
        raise RuntimeError(f"No registry entry for {symbol}")
    return dict(entry.get("hparams") or {})


def _train_one_fold(symbol: str, train_start: str, train_end: str,
                    hp: dict) -> tuple[float, str, str]:
    import src.train as _train
    # Patch optimizer/loss to match symbol registry (train_model reads module globals).
    _train.OPTIMIZER_NAME = hp.get("optimizer_name") or _train.OPTIMIZER_NAME
    _train.LOSS_FUNCTION  = hp.get("loss_function")  or _train.LOSS_FUNCTION
    result = _train.train_model(
        frequency=FREQUENCY,
        tsteps=TSTEPS,
        lstm_units=hp.get("lstm_units"),
        learning_rate=hp.get("learning_rate"),
        epochs=hp.get("epochs"),
        current_batch_size=hp.get("batch_size"),
        n_lstm_layers=hp.get("n_lstm_layers"),
        stateful=hp.get("stateful", True),
        features_to_use=hp.get("features_to_use"),
        train_start_date=train_start,
        train_end_date=train_end,
        symbol=symbol,
    )
    if result is None:
        raise RuntimeError(f"Training failed for {symbol} fold {train_start}..{train_end}")
    return result  # (val_loss, model_path, bias_path)


def _generate_test_predictions_csv(symbol: str, test_start: str, test_end: str,
                                   fold_predictions_csv: str,
                                   long_only: bool = False) -> None:
    """Run model-mode backtest over the test window; it writes the checkpoint CSV
    as a side effect. Copy that file to `fold_predictions_csv`."""
    from src.backtest import run_backtest_for_ui
    from src.config import get_predictions_csv_path
    if long_only:
        enable_longs, allow_shorts = True, False
    else:
        enable_longs, allow_shorts = True, True
    # Model-mode backtest over test window → side-effect writes checkpoint CSV.
    run_backtest_for_ui(
        symbol=symbol, frequency=FREQUENCY,
        prediction_mode="model",
        start_date=test_start, end_date=test_end,
        # Minimal strategy args — values irrelevant (only the predictions file matters).
        k_sigma_long=0.3, k_sigma_short=0.3, k_atr_long=0.4, k_atr_short=0.4,
        risk_per_trade_pct=0.01, reward_risk_ratio=2.5,
        enable_longs=enable_longs, allow_shorts=allow_shorts,
    )
    default_checkpoint = get_predictions_csv_path(symbol.lower(), FREQUENCY)
    Path(fold_predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(default_checkpoint, fold_predictions_csv)


def _get_walkforward_windows(symbol: str, first_test_start: str,
                             test_span_months: int, train_lookback_months: int,
                             overall_end: str):
    from src.walkforward import generate_walkforward_windows, infer_data_horizon
    import src.config as cfg
    hourly_csv = cfg.get_hourly_data_csv_path(FREQUENCY, symbol=symbol)
    df_full = pd.read_csv(hourly_csv)
    data_t_start, data_t_end = infer_data_horizon(df_full)
    eff_t_end = min(data_t_end, overall_end)
    windows = generate_walkforward_windows(
        data_t_start, eff_t_end,
        test_span_months=test_span_months,
        train_lookback_months=train_lookback_months,
        min_lookback_months=max(6, train_lookback_months - 6),
        first_test_start=first_test_start,
    )
    if not windows:
        raise RuntimeError(f"No walk-forward windows for {symbol}")
    return windows


def generate_fold_predictions(
    symbol: str,
    output_dir: Path,
    windows,
    long_only: bool = False,
) -> list[dict]:
    """Phase 0: retrain a model per fold, emit its test-window predictions to
    output_dir/fold{NN}_predictions.csv. Restores the production registry entry
    and symbol-scaler on exit.

    Returns a list of fold metadata dicts:
        {"fold_idx", "train_start", "train_end", "test_start", "test_end",
         "preds_csv", "model_path", "bias_path", "val_loss"}
    """
    from src.model_registry import get_best_model_entry, update_best_model
    import src.config as cfg

    saved_entry = get_best_model_entry(symbol, FREQUENCY, TSTEPS)
    if not saved_entry:
        raise RuntimeError(f"No registry entry for {symbol}")
    saved_model_path = saved_entry["model_path"]
    saved_bias_path  = saved_entry.get("bias_path")
    saved_val_loss   = saved_entry["validation_loss"]
    saved_hp         = dict(saved_entry.get("hparams") or {})

    scaler_path = cfg.get_scaler_params_json_path(FREQUENCY, symbol=symbol)
    scaler_backup = scaler_path + ".before_oos_walkforward.bak"
    if os.path.exists(scaler_path):
        shutil.copy2(scaler_path, scaler_backup)

    fold_meta: list[dict] = []
    try:
        print(f"[{symbol}] Phase 0: generating {len(windows)} fold predictions "
              f"(long_only={long_only})")
        for fold_idx, (train_w, test_w) in enumerate(windows, start=1):
            train_start = train_w.start
            train_end   = train_w.end
            test_start  = test_w.start
            test_end    = test_w.end
            fold_preds_csv = str(output_dir / f"fold{fold_idx:02d}_predictions.csv")
            print(f"[{symbol}] fold {fold_idx}/{len(windows)}: "
                  f"train [{train_start}..{train_end}) "
                  f"test [{test_start}..{test_end})", flush=True)

            try:
                val_loss, model_path, bias_path = _train_one_fold(
                    symbol, train_start, train_end, saved_hp)
            except Exception as exc:
                print(f"[{symbol}] fold {fold_idx} train failed: {exc}")
                continue

            # Promote this fold's model in the registry so prediction context picks it up.
            update_best_model(
                symbol=symbol, frequency=FREQUENCY, tsteps=TSTEPS,
                val_loss=float(val_loss),
                model_path=str(model_path),
                bias_path=str(bias_path) if bias_path else None,
                hparams=saved_hp, force=True,
            )

            try:
                _generate_test_predictions_csv(
                    symbol, test_start, test_end, fold_preds_csv,
                    long_only=long_only)
            except Exception as exc:
                print(f"[{symbol}] fold {fold_idx} prediction generation failed: {exc}")
                continue

            fold_meta.append({
                "fold_idx":     fold_idx,
                "train_start":  train_start,
                "train_end":    train_end,
                "test_start":   test_start,
                "test_end":     test_end,
                "preds_csv":    fold_preds_csv,
                "model_path":   str(model_path),
                "bias_path":    str(bias_path) if bias_path else None,
                "val_loss":     float(val_loss),
            })
    finally:
        # Restore the saved model + scaler so the symbol remains ready for production use.
        update_best_model(
            symbol=symbol, frequency=FREQUENCY, tsteps=TSTEPS,
            val_loss=float(saved_val_loss),
            model_path=str(saved_model_path),
            bias_path=str(saved_bias_path) if saved_bias_path else None,
            hparams=saved_hp, force=True,
        )
        if os.path.exists(scaler_backup):
            shutil.copy2(scaler_backup, scaler_path)
            try:
                os.remove(scaler_backup)
            except OSError:
                pass

    # Persist fold metadata for --phase5-only / reproducibility.
    meta_path = output_dir / "fold_meta.json"
    meta_path.write_text(json.dumps(fold_meta, indent=2), encoding="utf-8")
    return fold_meta


def load_fold_predictions(output_dir: Path) -> list[dict]:
    """Load fold metadata produced by generate_fold_predictions. Used by
    --phase5-only resume path."""
    meta_path = output_dir / "fold_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"fold_meta.json missing in {output_dir} — cannot resume. "
            f"Run Phase 0 first (drop --phase5-only).")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase 2 Stage A: cheap single-fold pre-filter
# ---------------------------------------------------------------------------
def stage_a_prefilter(symbol: str, rep_fold: dict, samples: pd.DataFrame,
                      long_only: bool = False) -> pd.DataFrame:
    """Score each LHS candidate on ONE representative fold's test window.
    Returns `samples` with a `stage_a_*` metrics columns appended."""
    preds_csv = rep_fold["preds_csv"]
    test_start = rep_fold["test_start"]
    test_end = rep_fold["test_end"]

    print(f"  Stage A: single-fold prefilter on fold {rep_fold['fold_idx']} "
          f"[{test_start}..{test_end}), {len(samples)} candidates")
    rows = []
    n = len(samples)
    t0 = time.time()
    for idx, row in samples.reset_index(drop=True).iterrows():
        params = row[PARAM_NAMES].to_dict()
        m = _one_backtest(symbol, preds_csv, params, test_start, test_end,
                          long_only=long_only)
        row_out = {**params,
                   **{f"stage_a_{k}": v for k, v in m.items() if k != "error"}}
        rows.append(row_out)
        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n - (idx + 1)) / rate if rate > 0 else 0.0
            print(f"  [stage_a] scored {idx+1}/{n}  rate={rate:.1f}/s  "
                  f"eta={eta/60:.1f} min", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2 Stage B: full-WF scoring on top candidates
# ---------------------------------------------------------------------------
def stage_b_full_wf(symbol: str, fold_meta: list[dict],
                    top_candidates: pd.DataFrame,
                    long_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each of the top_candidates, score on every fold in fold_meta.
    Returns (oos_long_df, oos_agg_df):
        oos_long_df: one row per (candidate, fold) — same shape as legacy
                     05_oos_long.csv.
        oos_agg_df : one row per candidate, aggregated with the same formula as
                     rank_candidates (Phase 6).
    """
    n_cand = len(top_candidates)
    n_folds = len(fold_meta)
    print(f"  Stage B: full walk-forward, {n_cand} candidates × {n_folds} folds "
          f"= {n_cand * n_folds} backtests")

    # Re-number candidate_id to align with Stage-B subset identity (stable for Phase 5 subset).
    top = top_candidates.reset_index(drop=True).copy()
    if "candidate_id" not in top.columns:
        top.insert(0, "candidate_id", range(1, len(top) + 1))

    rows: list[dict] = []
    t0 = time.time()
    total = n_cand * n_folds
    done = 0
    for fold in fold_meta:
        preds_csv = fold["preds_csv"]
        test_start = fold["test_start"]
        test_end = fold["test_end"]
        for _, cand in top.iterrows():
            params = {k: cand[k] for k in PARAM_NAMES}
            m = _one_backtest(symbol, preds_csv, params, test_start, test_end,
                              long_only=long_only)
            rows.append({
                "fold_idx":     fold["fold_idx"],
                "train_start":  fold["train_start"],
                "train_end":    fold["train_end"],
                "test_start":   test_start,
                "test_end":     test_end,
                "candidate_id": int(cand["candidate_id"]),
                **params,
                **{f"oos_{k}": v for k, v in m.items() if k != "error"},
                "fold_model":   os.path.basename(fold.get("model_path") or ""),
                "fold_val_loss": float(fold.get("val_loss") or 0.0),
            })
            done += 1
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(f"  [stage_b] {done}/{total}  rate={rate:.1f}/s  "
                      f"eta={eta/60:.1f} min", flush=True)

    oos_long = pd.DataFrame(rows)
    oos_agg = rank_candidates(oos_long) if not oos_long.empty else pd.DataFrame()
    return oos_long, oos_agg


# ---------------------------------------------------------------------------
# Phase 3: Filter viable candidates (operates on Stage-B OOS aggregate)
# ---------------------------------------------------------------------------
def filter_viable_oos(ranked: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    """Adaptive tier filter applied to per-candidate aggregated OOS metrics
    (output of rank_candidates / Stage B). Columns expected:
    `mean_sharpe`, `min_sharpe`, `pct_pos_folds`, `mean_pf`, `mean_dd`,
    `mean_trades`, `std_sharpe`."""
    if ranked.empty:
        return ranked.copy()

    # Tier list: (mean_sharpe_min, mean_pf_min, mean_dd_min, mean_trades_min,
    #             min_sharpe_min, pct_pos_min, min_dd_min)
    # `min_dd_min` is the worst-fold drawdown tolerance (e.g. -0.40 means no
    # single fold may have drawn down more than 40%). Critical for rejecting
    # two-sided candidates with catastrophic short-side tail losses — without
    # this gate, mean_dd can look fine while one quarter wipes out 90%.
    # Strict tiers gate on clean-Sharpe shape; permissive tiers relax min_sharpe
    # specifically to accommodate high-volatility symbols like NVDA where a
    # single quarter with Sharpe -3 is normal even for a profitable strategy.
    tiers = [
        (0.50, 1.20, -0.30, 40,  0.00, 0.60, -0.25),  # strict
        (0.40, 1.15, -0.32, 35, -0.10, 0.55, -0.30),
        (0.30, 1.10, -0.35, 30, -0.30, 0.50, -0.35),
        (0.20, 1.05, -0.40, 25, -0.50, 0.45, -0.40),
        (0.00, 1.01, -0.50, 20, -1.00, 0.40, -0.45),
        (0.00, 1.01, -0.55, 20, -2.00, 0.33, -0.50),  # high-vol tolerance
        (0.00, 1.00, -0.60, 20, -4.00, 0.25, -0.50),  # ultra-permissive (still min_dd-gated)
    ]
    for tier_idx, (ms, pf, dd, tr, mns, pct, mndd) in enumerate(tiers):
        mask = (
            (ranked["mean_sharpe"]   >  ms) &
            (ranked["mean_pf"]       >  pf) &
            (ranked["mean_dd"]       >  dd) &
            (ranked["mean_trades"]   >= tr) &
            (ranked["min_sharpe"]    >  mns) &
            (ranked["pct_pos_folds"] >= pct) &
            (ranked["min_dd"]        >  mndd)
        )
        out = ranked[mask].copy().reset_index(drop=True)
        if len(out) >= min_count or tier_idx == len(tiers) - 1:
            print(f"  filter tier {tier_idx+1}/{len(tiers)}: "
                  f"mean_sh>{ms} mean_pf>{pf} mean_dd>{dd} mean_trades>={tr} "
                  f"min_sh>{mns} pct_pos>={pct} min_dd>{mndd} → {len(out)} viable",
                  flush=True)
            if len(out) > 0:
                return out
            # Else fall through to final oos_score-only fallback below.
            break

    # No tier yielded any candidates. Surface top-5 by oos_score for human
    # inspection in the log, then return empty so Phase 3 errors out cleanly
    # rather than silently auto-promoting a structurally-unsafe candidate.
    # (Previously the fallback returned top-N and downstream would auto-promote
    # the "best of bad" — leading to the 2026-04-17 NVDA two-sided incident
    # where every candidate had ≥49% worst-fold DD but one got promoted anyway.)
    top = ranked.sort_values("oos_score", ascending=False).head(5)
    print("  no tier yielded viable candidates. Top 5 by oos_score (for inspection only,",
          flush=True)
    print("  not promoted — fix filters or investigate pipeline before rerunning):",
          flush=True)
    for _, r in top.iterrows():
        print(f"    cand#{int(r.candidate_id):3d} mean_sh={r.mean_sharpe:+.2f} "
              f"pct_pos={r.pct_pos_folds:.2f} min_dd={r.min_dd:+.2%} "
              f"mean_pf={r.mean_pf:.2f} oos_score={r.oos_score:+.2f}",
              flush=True)
    return ranked.iloc[0:0].copy()


# Legacy alias retained for backward-compat if any caller imports this name.
def filter_viable(scored: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    """Back-compat shim. Dispatches to filter_viable_oos when OOS columns are
    present, else falls back to the legacy full-window filter."""
    if "mean_sharpe" in scored.columns:
        return filter_viable_oos(scored, min_count=min_count)
    # Legacy full-window filter (pre-refactor). Retained so older analysis
    # notebooks don't break. Not used by the main pipeline anymore.
    tiers = [
        (0.5, 1.2, -0.30, 40,  0.0),
        (0.4, 1.15, -0.32, 35, -0.1),
        (0.3, 1.10, -0.35, 30, -0.3),
        (0.2, 1.05, -0.40, 25, -0.5),
        (0.0, 1.01, -0.50, 20, -1.0),
    ]
    for tier_idx, (sh, pf, dd, tr, hm) in enumerate(tiers):
        mask = (
            (scored["full_sharpe_ratio"] >  sh) &
            (scored["full_profit_factor"] > pf) &
            (scored["full_max_drawdown"] >  dd) &
            (scored["full_n_trades"]    >= tr)
        )
        if "half_min_sharpe" in scored.columns:
            mask &= scored["half_min_sharpe"].fillna(-9e9) > hm
        out = scored[mask].copy().reset_index(drop=True)
        if len(out) >= min_count or tier_idx == len(tiers) - 1:
            return out
    return pd.DataFrame(columns=scored.columns)


# ---------------------------------------------------------------------------
# Phase 4: Maximin diversification
# ---------------------------------------------------------------------------
def maximin_select(viable: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select `n` diverse points in PARAM_NAMES-space via farthest-point sampling.
    Anchored at the highest-score viable row.

    IMPORTANT: both branches (<=n early return and full maximin loop) MUST return
    a DataFrame with a `pick_order` column 1..N — downstream rank_candidates
    uses it as candidate_id."""
    # Pick the anchor score column — prefer OOS if present, else full-window legacy.
    if "oos_score" in viable.columns:
        score_col = "oos_score"
    elif "mean_sharpe" in viable.columns:
        score_col = "mean_sharpe"
    elif "full_sharpe_ratio" in viable.columns:
        # Legacy: stability-adjusted if half_min_sharpe present.
        if "half_min_sharpe" in viable.columns:
            viable = viable.copy()
            viable["_anchor_score"] = (
                viable["full_sharpe_ratio"]
                * np.maximum(viable["half_min_sharpe"], 0.0)
            )
            score_col = "_anchor_score"
        else:
            score_col = "full_sharpe_ratio"
    else:
        raise ValueError("maximin_select: no recognized score column on viable df")

    if len(viable) <= n:
        out = viable.copy().sort_values(score_col, ascending=False).reset_index(drop=True)
        # Must add pick_order here too — Phase 5 uses it as candidate_id.
        # Without it every row collapses to candidate_id=-1 in rank_candidates.
        out.insert(0, "pick_order", range(1, len(out) + 1))
        return out

    # Normalize params to [0,1] using global ranges for distance.
    lows = np.array([PARAM_RANGES[k][0] for k in PARAM_NAMES])
    highs = np.array([PARAM_RANGES[k][1] for k in PARAM_NAMES])
    X = (viable[PARAM_NAMES].to_numpy() - lows) / (highs - lows)

    anchor_idx = int(viable[score_col].values.argmax())

    chosen = [anchor_idx]
    # Track each candidate's distance to the nearest chosen point.
    min_dist = np.linalg.norm(X - X[anchor_idx], axis=1)
    while len(chosen) < n:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break
        chosen.append(next_idx)
        new_dists = np.linalg.norm(X - X[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
        min_dist[next_idx] = 0  # exclude already-picked

    selected = viable.iloc[chosen].reset_index(drop=True)
    selected.insert(0, "pick_order", range(1, len(selected) + 1))
    return selected


# ---------------------------------------------------------------------------
# Phase 5 (slimmed): subset Stage-B results to the diverse winners
# ---------------------------------------------------------------------------
def subset_oos_long_to_diverse(oos_long_b: pd.DataFrame,
                               diverse: pd.DataFrame) -> pd.DataFrame:
    """Stage B already scored every top-200 candidate on all 9 folds. Phase 5's
    old semantic (retrain per fold, score N diverse candidates) is now just a
    row-subset of Stage-B's oos_long by candidate_id.

    `diverse` carries `candidate_id` from Stage-B scoring (we preserved it
    through maximin_select's pick_order column). We translate pick_order →
    original Stage-B candidate_id via the `candidate_id` column on diverse.
    """
    if oos_long_b.empty or diverse.empty:
        return pd.DataFrame()
    # If the diverse df carried candidate_id through maximin (preferred), use it.
    if "candidate_id" in diverse.columns:
        ids = diverse["candidate_id"].astype(int).tolist()
        return oos_long_b[oos_long_b["candidate_id"].isin(ids)].copy().reset_index(drop=True)
    # Fallback: match by the 6 param columns.
    key = PARAM_NAMES
    merged = oos_long_b.merge(
        diverse[key].drop_duplicates(),
        on=key, how="inner")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 6: Rank candidates by robustness-adjusted OOS score
# ---------------------------------------------------------------------------
def rank_candidates(oos_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-(candidate, fold) oos_long DataFrame into one row per
    candidate with mean/std/min sharpe, pct_pos_folds, etc. Scored by
    `oos_score` (higher = better)."""
    if oos_long.empty:
        return pd.DataFrame()

    # Profit-factor of exactly 0.0 is a sentinel for "undefined" (single-trade
    # or all-winning folds where gross losses = 0 and the backtester stored 0
    # instead of inf/NaN). Replace with NaN so groupby("mean") skips them —
    # otherwise a profitable strategy with a few sparse-trade folds gets its
    # mean_pf dragged to ~0.9 and fails the filter despite compound profitability.
    oos_long = oos_long.copy()
    oos_long["oos_profit_factor_clean"] = oos_long["oos_profit_factor"].replace(0.0, float("nan"))

    g = oos_long.groupby("candidate_id", as_index=False)
    agg = g.agg(
        mean_sharpe      = ("oos_sharpe_ratio", "mean"),
        std_sharpe       = ("oos_sharpe_ratio", "std"),
        min_sharpe       = ("oos_sharpe_ratio", "min"),
        median_sharpe    = ("oos_sharpe_ratio", "median"),
        mean_pf          = ("oos_profit_factor_clean", "mean"),
        mean_dd          = ("oos_max_drawdown", "mean"),
        min_dd           = ("oos_max_drawdown", "min"),
        mean_trades      = ("oos_n_trades", "mean"),
        n_folds          = ("oos_sharpe_ratio", "size"),
    )
    # Ensure mean_pf has a finite fallback when ALL folds were sentinel-zero
    # (rare; means no fold had both wins and losses). Treat as neutral 1.0.
    agg["mean_pf"] = agg["mean_pf"].fillna(1.0)
    pos_folds = (
        oos_long.assign(_pos=(oos_long["oos_sharpe_ratio"] > 0).astype(float))
                .groupby("candidate_id", as_index=False)["_pos"].mean()
                .rename(columns={"_pos": "pct_pos_folds"})
    )
    agg = agg.merge(pos_folds, on="candidate_id", how="left")

    # Attach params from first row per candidate
    first_rows = oos_long.groupby("candidate_id", as_index=False).first()[
        ["candidate_id"] + PARAM_NAMES]
    agg = agg.merge(first_rows, on="candidate_id", how="left")

    agg["std_sharpe"] = agg["std_sharpe"].fillna(0.0)

    agg["oos_score"] = (
          agg["mean_sharpe"]
        - 0.5 * agg["std_sharpe"]
        + 0.3 * agg["pct_pos_folds"]
        - 1.0 * np.maximum(0.0, -agg["min_sharpe"])
        - 0.5 * np.maximum(0.0, -0.25 - agg["min_dd"])
    )
    return agg.sort_values("oos_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 7: Auto-promote winner
# ---------------------------------------------------------------------------
def promote_winner(symbol: str, winner: pd.Series, oos_stats: pd.Series,
                   long_only: bool = False) -> None:
    from src.core.config_resolver import save_strategy_defaults
    if long_only:
        enable_longs, allow_shorts = True, False
    else:
        enable_longs, allow_shorts = True, True
    save_strategy_defaults(
        risk_per_trade_pct=float(winner["risk_per_trade_pct"]),
        reward_risk_ratio=float(winner["reward_risk_ratio"]),
        k_sigma_long=float(winner["k_sigma_long"]),
        k_sigma_short=float(winner["k_sigma_short"]),
        k_atr_long=float(winner["k_atr_long"]),
        k_atr_short=float(winner["k_atr_short"]),
        enable_longs=enable_longs,
        allow_shorts=allow_shorts,
        symbol=symbol,
        frequency=FREQUENCY,
        source=f"robust_param_selection@{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    )
    # Also append a candidate record into configs/library/{SYM}/{FREQ}/ for traceability.
    from src.core.config_library import _get_repo_root  # type: ignore
    lib_dir = _get_repo_root() / "configs" / "library" / symbol.upper() / FREQUENCY
    lib_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lib_path = lib_dir / f"cfg_{ts}_robust_oos.json"
    lib_path.write_text(json.dumps({
        "meta": {
            "symbol": symbol, "frequency": FREQUENCY,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "scripts/robust_param_selection.py",
            "long_only": bool(long_only),
        },
        "strategy": {
            "risk_per_trade_pct": float(winner["risk_per_trade_pct"]),
            "reward_risk_ratio":  float(winner["reward_risk_ratio"]),
            "k_sigma_long":       float(winner["k_sigma_long"]),
            "k_sigma_short":      float(winner["k_sigma_short"]),
            "k_atr_long":         float(winner["k_atr_long"]),
            "k_atr_short":        float(winner["k_atr_short"]),
            "enable_longs":  enable_longs,
            "allow_shorts":  allow_shorts,
        },
        "oos_metrics": {
            "mean_sharpe":   float(oos_stats["mean_sharpe"]),
            "std_sharpe":    float(oos_stats["std_sharpe"]),
            "min_sharpe":    float(oos_stats["min_sharpe"]),
            "median_sharpe": float(oos_stats["median_sharpe"]),
            "pct_pos_folds": float(oos_stats["pct_pos_folds"]),
            "mean_pf":       float(oos_stats["mean_pf"]),
            "mean_dd":       float(oos_stats["mean_dd"]),
            "n_folds":       int(oos_stats["n_folds"]),
            "oos_score":     float(oos_stats["oos_score"]),
        },
    }, indent=2), encoding="utf-8")
    print(f"[{symbol}] promoted to configs/symbols/{symbol.upper()}/active.json")
    print(f"[{symbol}] candidate logged to {lib_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=2000,
                   help="LHS sample count for phase 1")
    p.add_argument("--n-diverse", type=int, default=30,
                   help="Number of diverse candidates to select from viable set")
    p.add_argument("--stage-b-top", type=int, default=200,
                   help="Top N Stage-A candidates forwarded to full-WF Stage B")
    p.add_argument("--window-start", type=str, default="2023-01-03")
    p.add_argument("--window-end",   type=str, default="2026-04-02")
    p.add_argument("--first-test-start", type=str, default="2024-01-01")
    p.add_argument("--test-span-months", type=int, default=3)
    p.add_argument("--train-lookback-months", type=int, default=12)
    p.add_argument("--skip-phase5", action="store_true",
                   help="Skip Phase 5 ranking + auto-promote (stop after Stage B/Phase 4)")
    p.add_argument("--phase5-only", action="store_true",
                   help="Resume at Phase 5: load cached fold predictions + "
                        "04_diverse.csv from output dir")
    p.add_argument("--long-only", action="store_true", default=False,
                   help="Disable short signals in every backtest (allow_shorts=False). "
                        "Keeps the 6D LHS intact — short params are ignored by the strategy.")
    p.add_argument("--auto-promote", action="store_true",
                   help="Write top-1 to configs/symbols/{SYM}/active.json")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    symbol = args.symbol.upper()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "backtests" / f"robust_selection_{symbol.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"===== {symbol} : robust param selection =====")
    print(f"Output directory: {out_dir}")
    print(f"Long-only: {args.long_only}")

    # Write run metadata up-front so downstream analysis knows the mode.
    run_meta = {
        "symbol": symbol,
        "frequency": FREQUENCY,
        "long_only": bool(args.long_only),
        "n_samples": int(args.n_samples),
        "n_diverse": int(args.n_diverse),
        "stage_b_top": int(args.stage_b_top),
        "window_start": args.window_start,
        "window_end": args.window_end,
        "first_test_start": args.first_test_start,
        "test_span_months": int(args.test_span_months),
        "train_lookback_months": int(args.train_lookback_months),
        "seed": int(args.seed),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2),
                                           encoding="utf-8")

    # --------- Phase 0: walk-forward windows + fold predictions ---------
    windows = _get_walkforward_windows(
        symbol=symbol,
        first_test_start=args.first_test_start,
        test_span_months=args.test_span_months,
        train_lookback_months=args.train_lookback_months,
        overall_end=args.window_end,
    )

    if args.phase5_only:
        # Resume path: load cached fold predictions and the diverse set.
        fold_meta = load_fold_predictions(out_dir)
        diverse_csv = out_dir / "04_diverse.csv"
        if not diverse_csv.exists():
            print(f"ERROR: --phase5-only requires {diverse_csv} to exist")
            sys.exit(1)
        diverse = pd.read_csv(diverse_csv)
        oos_long_b_csv = out_dir / "05_oos_long.csv"
        if not oos_long_b_csv.exists():
            print(f"ERROR: --phase5-only requires {oos_long_b_csv} to exist")
            sys.exit(1)
        oos_long_b = pd.read_csv(oos_long_b_csv)
        print(f"[Resume] Loaded {len(fold_meta)} folds, {len(diverse)} diverse, "
              f"{len(oos_long_b)} Stage-B rows")
        return _run_phase5_onwards(args, symbol, out_dir, diverse, oos_long_b)

    # Generate fold predictions (once).
    t0 = time.time()
    fold_meta = generate_fold_predictions(
        symbol=symbol, output_dir=out_dir, windows=windows,
        long_only=args.long_only,
    )
    print(f"  Phase 0 done in {(time.time()-t0)/60:.1f} min "
          f"({len(fold_meta)} folds cached)")
    if not fold_meta:
        print(f"ERROR: no fold predictions generated for {symbol}")
        sys.exit(2)

    # --------- Phase 1: LHS ---------
    print(f"\n[Phase 1] LHS sampling: {args.n_samples} points, seed={args.seed}")
    samples = lhs_sample(args.n_samples, seed=args.seed)
    samples.to_csv(out_dir / "01_samples.csv", index=False)

    # --------- Phase 2: two-stage honest-OOS scoring ---------
    # Representative fold = middle fold for prefilter.
    rep_fold = fold_meta[len(fold_meta) // 2]
    print(f"\n[Phase 2] Two-stage OOS scoring")
    t0 = time.time()
    stage_a = stage_a_prefilter(symbol, rep_fold, samples, long_only=args.long_only)
    stage_a.to_csv(out_dir / "02a_stage_a.csv", index=False)

    # Take top-N by Stage-A Sharpe (defaults to 200 / 10% of 2000).
    k = min(int(args.stage_b_top), len(stage_a))
    top = stage_a.nlargest(k, "stage_a_sharpe_ratio").copy().reset_index(drop=True)
    top.insert(0, "candidate_id", range(1, len(top) + 1))
    top.to_csv(out_dir / "02b_stage_b_input.csv", index=False)
    print(f"  Stage A done. Taking top {k} for Stage B.")

    oos_long_b, scored = stage_b_full_wf(
        symbol=symbol, fold_meta=fold_meta, top_candidates=top,
        long_only=args.long_only,
    )
    # Write as 02_scored.csv (Phase-2 final output) AND 05_oos_long.csv
    # (Stage-B long form == what Phase 5 used to produce).
    scored.to_csv(out_dir / "02_scored.csv", index=False)
    oos_long_b.to_csv(out_dir / "05_oos_long.csv", index=False)
    print(f"  Phase 2 done in {(time.time()-t0)/60:.1f} min")

    # --------- Phase 3: filter viable ---------
    viable = filter_viable_oos(scored)
    viable.to_csv(out_dir / "03_viable.csv", index=False)
    print(f"\n[Phase 3] Viable candidates: {len(viable)}/{len(scored)}")
    if viable.empty:
        print(f"ERROR: no viable candidates for {symbol}. Try loosening filters.")
        sys.exit(2)

    # --------- Phase 4: maximin diversification ---------
    diverse = maximin_select(viable, n=args.n_diverse)
    diverse.to_csv(out_dir / "04_diverse.csv", index=False)
    print(f"\n[Phase 4] Maximin-selected: {len(diverse)} diverse candidates")

    if args.skip_phase5:
        print("\nSkipping Phase 5+ (--skip-phase5).")
        return

    _run_phase5_onwards(args, symbol, out_dir, diverse, oos_long_b)


def _run_phase5_onwards(args, symbol: str, out_dir: Path,
                        diverse: pd.DataFrame,
                        oos_long_b: pd.DataFrame) -> None:
    """Phases 5-7. Stage B has already produced oos_long_b (one row per
    (candidate, fold)); Phase 5 is now just a subset + rerank for the diverse
    winners. No retraining happens here."""
    # --------- Phase 5 (slimmed): subset Stage-B oos_long to diverse winners ---------
    print(f"\n[Phase 5] Subsetting Stage-B OOS results to {len(diverse)} diverse winners")
    oos_long_subset = subset_oos_long_to_diverse(oos_long_b, diverse)
    if oos_long_subset.empty:
        print("ERROR: no Stage-B rows match the diverse set — candidate_id mismatch?")
        sys.exit(3)
    oos_long_subset.to_csv(out_dir / "05_oos_long_diverse.csv", index=False)

    # --------- Phase 6 ---------
    ranked = rank_candidates(oos_long_subset)
    ranked.to_csv(out_dir / "06_ranked.csv", index=False)
    print(f"\n[Phase 6] Ranked {len(ranked)} candidates; top 5:")
    with pd.option_context("display.max_columns", None, "display.width", 180,
                           "display.float_format", "{:.4f}".format):
        print(ranked.head(5).to_string(index=False))

    if ranked.empty:
        print("ERROR: no OOS results to rank.")
        sys.exit(3)

    winner_row = ranked.iloc[0]
    winner_params = winner_row[PARAM_NAMES]
    winner_stats  = winner_row[["mean_sharpe","std_sharpe","min_sharpe","median_sharpe",
                                "pct_pos_folds","mean_pf","mean_dd","n_folds","oos_score"]]
    (out_dir / "winner.json").write_text(json.dumps({
        "symbol": symbol,
        "long_only": bool(args.long_only),
        "params": winner_params.to_dict(),
        "oos_stats": winner_stats.to_dict(),
    }, indent=2))
    print(f"\n[Winner for {symbol}]")
    print(json.dumps({
        "long_only": bool(args.long_only),
        "params":    winner_params.to_dict(),
        "oos_stats": winner_stats.to_dict(),
    }, indent=2))

    if args.auto_promote:
        print(f"\n[Phase 7] Auto-promoting winner to configs/symbols/{symbol}/active.json")
        promote_winner(symbol, winner_params, winner_stats, long_only=args.long_only)


if __name__ == "__main__":
    main()
