"""Stage 2 of portfolio-Sharpe parameter optimisation.

Consumes the Stage 1 output of `scripts/robust_param_selection.py` (one
output directory per symbol, each containing ``fold_meta.json`` and
``04_diverse.csv``) and searches over cross-symbol parameter combinations,
selecting the one that maximises the portfolio Sharpe ratio across the
9 walk-forward folds.

Inputs (--run-root):
    <run_root>/<SYM>/fold_meta.json
    <run_root>/<SYM>/04_diverse.csv
    <run_root>/<SYM>/foldNN_predictions.csv

Outputs:
    <run_root>/summary.json                 — winner params + portfolio metrics
    <run_root>/combinations_scored.csv      — all M sampled combos (ranked)
    configs/symbols/<SYM>/active.json       — promoted (unless --no-promote)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.portfolio.capital_allocator import AllocationConfig
from src.portfolio.portfolio_backtest import (
    PortfolioBacktestConfig,
    SymbolBarData,
    run_portfolio_backtest,
)
from src.strategy import StrategyConfig

PARAM_NAMES = [
    "k_sigma_long", "k_sigma_short", "k_atr_long", "k_atr_short",
    "reward_risk_ratio", "risk_per_trade_pct",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_portfolio_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ohlc(symbol: str, frequency: str = "60min") -> pd.DataFrame:
    csv_path = REPO_ROOT / "data" / "processed" / f"{symbol.lower()}_{frequency}.csv"
    df = pd.read_csv(csv_path)
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def load_symbol_stage1(sym_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    """Return (fold_meta, top_k_params_df) for a symbol's Stage-1 output.

    Prefers ``04_diverse.csv`` (the Phase-3+4 filtered+diversified winners).
    Falls back to the top of ``02_scored.csv`` by ``oos_score`` when the
    symbol's Phase-3 filter yielded no viable candidates — we still want
    the symbol to participate in the portfolio-level search; the portfolio
    optimiser will decide whether its least-bad candidates are worth
    including based on overall portfolio Sharpe.
    """
    fm_path = sym_dir / "fold_meta.json"
    if not fm_path.exists():
        raise FileNotFoundError(f"Missing {fm_path}")

    fold_meta = json.loads(fm_path.read_text(encoding="utf-8"))
    # Re-resolve preds_csv relative to sym_dir so we do not depend on absolute
    # paths that may have been recorded for a different output_dir.
    for fm in fold_meta:
        local_csv = sym_dir / f"fold{fm['fold_idx']:02d}_predictions.csv"
        if local_csv.exists():
            fm["preds_csv"] = str(local_csv)

    div_path = sym_dir / "04_diverse.csv"
    if div_path.exists():
        diverse = pd.read_csv(div_path)
    else:
        scored_path = sym_dir / "02_scored.csv"
        if not scored_path.exists():
            raise FileNotFoundError(
                f"Neither {div_path} nor {scored_path} exist. Stage 1 did not "
                f"produce usable candidates for {sym_dir.name}.")
        print(f"  [{sym_dir.name}] NOTE: 04_diverse.csv missing; falling back "
              f"to top-K by oos_score from 02_scored.csv "
              f"(Phase-3 filter rejected all candidates).", flush=True)
        scored = pd.read_csv(scored_path)
        diverse = scored.sort_values("oos_score", ascending=False).reset_index(drop=True)

    missing = [c for c in PARAM_NAMES if c not in diverse.columns]
    if missing:
        raise ValueError(f"{div_path} missing params: {missing}")
    return fold_meta, diverse.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-combination scoring
# ---------------------------------------------------------------------------
def _slice_ohlc(ohlc_full: pd.DataFrame, test_start: str, test_end: str) -> pd.DataFrame:
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    mask = (ohlc_full["Time"] >= start) & (ohlc_full["Time"] < end)
    return ohlc_full.loc[mask].reset_index(drop=True)


_PREDS_CACHE: dict[str, pd.DataFrame] = {}


def _load_preds_cached(path: str) -> pd.DataFrame:
    cached = _PREDS_CACHE.get(path)
    if cached is not None:
        return cached
    from src.backtest import _load_predictions_csv
    df = _load_predictions_csv(path)
    _PREDS_CACHE[path] = df
    return df


def build_aligned_symbol_data(
    symbols: list[str],
    ohlc_by_sym: dict[str, pd.DataFrame],
    preds_csv_by_sym: dict[str, str],
    test_start: str,
    test_end: str,
) -> dict[str, SymbolBarData]:
    """For a single fold: slice each symbol's OHLC to the test window, inner-join
    on Time, then build one SymbolBarData per symbol with a predictor aligned
    to the common index."""
    from src.backtest import _make_csv_prediction_provider

    sliced: dict[str, pd.DataFrame] = {}
    common: pd.Index | None = None
    for sym in symbols:
        df = _slice_ohlc(ohlc_by_sym[sym], test_start, test_end)
        if df.empty:
            raise ValueError(f"{sym}: empty OHLC for [{test_start}..{test_end})")
        df["Time"] = pd.to_datetime(df["Time"])
        sliced[sym] = df
        idx = pd.Index(df["Time"])
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError(f"No common timestamps across {symbols} in [{test_start}..{test_end})")
    common = common.sort_values()

    result: dict[str, SymbolBarData] = {}
    for sym in symbols:
        df = sliced[sym]
        aligned = df[df["Time"].isin(common)].sort_values("Time").reset_index(drop=True)
        preds_df = _load_preds_cached(preds_csv_by_sym[sym])
        provider = _make_csv_prediction_provider(preds_df, aligned)
        ohlc_indexed = aligned.set_index("Time")[["Open", "High", "Low", "Close"]]
        result[sym] = SymbolBarData(ohlc=ohlc_indexed, predictions=provider)
    return result


def score_combination(
    combo_idx: tuple[int, ...],
    symbols: list[str],
    params_by_sym: dict[str, list[dict]],
    fold_meta_by_sym: dict[str, list[dict]],
    ohlc_by_sym: dict[str, pd.DataFrame],
    allocation_cfg: AllocationConfig,
    initial_equity: float = 100_000.0,
    long_only: bool = True,
) -> dict:
    """Score one combination (one candidate index per symbol) across all folds.

    Returns a dict with combo_idx, per-fold portfolio Sharpe/DD, aggregate
    portfolio_score, and per-symbol breakdowns from the last fold (for sanity).
    """
    # Build StrategyConfig per symbol once (same across folds).
    strat_by_sym: dict[str, StrategyConfig] = {}
    for sym, idx in zip(symbols, combo_idx):
        p = params_by_sym[sym][idx]
        strat_by_sym[sym] = StrategyConfig(
            risk_per_trade_pct=float(p["risk_per_trade_pct"]),
            reward_risk_ratio=float(p["reward_risk_ratio"]),
            k_sigma_long=float(p["k_sigma_long"]),
            k_sigma_short=float(p["k_sigma_short"]),
            k_atr_long=float(p["k_atr_long"]),
            k_atr_short=float(p["k_atr_short"]),
            enable_longs=True,
            allow_shorts=False if long_only else True,
        )

    # Folds are assumed aligned (same test_start/test_end across symbols).
    n_folds = len(fold_meta_by_sym[symbols[0]])
    fold_sharpes: list[float] = []
    fold_dds: list[float] = []
    fold_n_trades: list[int] = []

    for f in range(n_folds):
        ref = fold_meta_by_sym[symbols[0]][f]
        test_start, test_end = ref["test_start"], ref["test_end"]
        preds_csv_by_sym = {sym: fold_meta_by_sym[sym][f]["preds_csv"]
                            for sym in symbols}
        aligned = build_aligned_symbol_data(
            symbols=symbols, ohlc_by_sym=ohlc_by_sym,
            preds_csv_by_sym=preds_csv_by_sym,
            test_start=test_start, test_end=test_end,
        )

        pbt_cfg = PortfolioBacktestConfig(
            symbols=list(symbols),
            initial_equity=initial_equity,
            allocation_config=allocation_cfg,
            per_symbol_strategy=strat_by_sym,
            commission_per_unit_per_leg=0.005,
            min_commission_per_order=1.0,
        )
        try:
            result = run_portfolio_backtest(aligned, pbt_cfg)
        except Exception as exc:  # noqa: BLE001
            fold_sharpes.append(0.0)
            fold_dds.append(0.0)
            fold_n_trades.append(0)
            continue

        fold_sharpes.append(result.portfolio_sharpe())
        fold_dds.append(result.max_drawdown())
        fold_n_trades.append(sum(len(t) for t in result.per_symbol_trades.values()))

    sharpes = np.array(fold_sharpes, dtype=float)
    dds = np.array(fold_dds, dtype=float)
    mean_sh = float(np.mean(sharpes))
    std_sh = float(np.std(sharpes, ddof=0))
    min_sh = float(np.min(sharpes))
    median_sh = float(np.median(sharpes))
    pct_pos = float(np.mean(sharpes > 0))
    mean_dd = float(np.mean(dds))
    min_dd = float(np.min(dds))

    # Same robustness formula as Phase 6 in robust_param_selection.
    score = (
        mean_sh
        - 0.5 * std_sh
        + 0.3 * pct_pos
        - 1.0 * max(0.0, -min_sh)
        - 0.5 * max(0.0, -0.25 - min_dd)
    )

    return {
        "combo_idx": list(combo_idx),
        "mean_portfolio_sharpe": mean_sh,
        "std_portfolio_sharpe": std_sh,
        "min_portfolio_sharpe": min_sh,
        "median_portfolio_sharpe": median_sh,
        "pct_pos_folds": pct_pos,
        "mean_portfolio_dd": mean_dd,
        "min_portfolio_dd": min_dd,
        "portfolio_score": float(score),
        "fold_sharpes": fold_sharpes,
        "fold_dds": fold_dds,
        "fold_n_trades": fold_n_trades,
    }


# ---------------------------------------------------------------------------
# Worker entrypoint for ProcessPoolExecutor (must be top-level, picklable)
# ---------------------------------------------------------------------------
# ``_WORKER_CTX`` is populated once per worker via ``_worker_init`` so we don't
# pay the cost of pickling the full OHLC tables across thousands of tasks.
_WORKER_CTX: dict | None = None


def _worker_init(ctx):
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _worker_score(combo_idx):
    ctx = _WORKER_CTX
    if ctx is None:
        raise RuntimeError("Worker not initialised")
    return score_combination(
        combo_idx=tuple(combo_idx),
        symbols=ctx["symbols"],
        params_by_sym=ctx["params_by_sym"],
        fold_meta_by_sym=ctx["fold_meta_by_sym"],
        ohlc_by_sym=ctx["ohlc_by_sym"],
        allocation_cfg=ctx["allocation_cfg"],
        initial_equity=ctx["initial_equity"],
        long_only=ctx["long_only"],
    )


# ---------------------------------------------------------------------------
# Combinations sampler
# ---------------------------------------------------------------------------
def sample_combinations(k_per_sym: dict[str, int], m: int, seed: int = 42) -> np.ndarray:
    """Sample M random (n_symbols,) tuples where each column i is in [0, k_i).

    Always includes the (0, 0, ..., 0) all-top-1 baseline as the first row.
    """
    rng = np.random.default_rng(seed)
    syms = list(k_per_sym.keys())
    ks = np.array([k_per_sym[s] for s in syms], dtype=int)
    # All-top-1 baseline first.
    combos = [np.zeros(len(syms), dtype=int)]
    seen = {tuple(combos[0])}
    attempts = 0
    while len(combos) < m and attempts < m * 5:
        c = tuple(int(rng.integers(0, k)) for k in ks)
        attempts += 1
        if c in seen:
            continue
        seen.add(c)
        combos.append(np.array(c, dtype=int))
    return np.array(combos, dtype=int)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
def promote_winner(
    winner_params_by_sym: dict[str, dict],
    frequency: str,
    long_only: bool,
    source_tag: str,
) -> None:
    from datetime import datetime, timezone
    from src.core.config_resolver import save_strategy_defaults
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for sym, p in winner_params_by_sym.items():
        save_strategy_defaults(
            risk_per_trade_pct=float(p["risk_per_trade_pct"]),
            reward_risk_ratio=float(p["reward_risk_ratio"]),
            k_sigma_long=float(p["k_sigma_long"]),
            k_sigma_short=float(p["k_sigma_short"]),
            k_atr_long=float(p["k_atr_long"]),
            k_atr_short=float(p["k_atr_short"]),
            enable_longs=True,
            allow_shorts=(not long_only),
            symbol=sym,
            frequency=frequency,
            source=f"{source_tag}@{ts}",
        )
        print(f"  promoted {sym} -> configs/symbols/{sym}/active.json", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=str,
                    help="Stage 1 output root. Must contain <SYM>/ subdirs with "
                         "fold_meta.json and 04_diverse.csv")
    ap.add_argument("--portfolio-config", type=str,
                    default=str(REPO_ROOT / "configs" / "portfolio.json"))
    ap.add_argument("--n-combinations", type=int, default=2000)
    ap.add_argument("--k-per-symbol", type=int, default=25,
                    help="Take top-K rows (by file order — 04_diverse.csv is "
                         "already pick-ordered) per symbol into the search")
    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--long-only", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=max(1, (sys.platform != "win32") * 6 or 4))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-promote", action="store_true",
                    help="Skip overwriting configs/symbols/<SYM>/active.json")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        print(f"ERROR: run-root does not exist: {run_root}")
        sys.exit(2)

    pconf = load_portfolio_config(Path(args.portfolio_config))
    symbols: list[str] = list(pconf["symbols"])
    frequency: str = pconf.get("frequency", "60min")

    print(f"===== Portfolio param search =====")
    print(f"run-root:    {run_root}")
    print(f"symbols:     {symbols}")
    print(f"k/symbol:    {args.k_per_symbol}")
    print(f"n combos:    {args.n_combinations}")
    print(f"workers:     {args.workers}")
    print(f"long-only:   {args.long_only}")

    # ── Load Stage-1 artifacts per symbol ──────────────────────────────
    fold_meta_by_sym: dict[str, list[dict]] = {}
    params_by_sym: dict[str, list[dict]] = {}
    k_per_sym: dict[str, int] = {}
    for sym in symbols:
        sym_dir = run_root / sym
        fold_meta, diverse = load_symbol_stage1(sym_dir)
        fold_meta_by_sym[sym] = fold_meta
        top_k = diverse.head(args.k_per_symbol).reset_index(drop=True)
        params_by_sym[sym] = top_k[PARAM_NAMES].to_dict(orient="records")
        k_per_sym[sym] = len(params_by_sym[sym])
        print(f"  [{sym}] folds={len(fold_meta)}  top-k={k_per_sym[sym]}")

    # Intersect fold windows across all symbols so every scored fold is
    # covered by every symbol's fresh predictions. Any symbol missing a
    # specific fold (e.g. data-span mismatch causing Phase 0 to skip it)
    # would otherwise cause an IndexError or a silent skew.
    fold_window_sets = [
        {(fm["test_start"], fm["test_end"]) for fm in fold_meta_by_sym[sym]}
        for sym in symbols
    ]
    common_windows = sorted(set.intersection(*fold_window_sets))
    for sym in symbols:
        before = len(fold_meta_by_sym[sym])
        fold_meta_by_sym[sym] = [
            fm for fm in fold_meta_by_sym[sym]
            if (fm["test_start"], fm["test_end"]) in set(common_windows)
        ]
        # Reorder so the by-position indexing in score_combination lines up.
        fold_meta_by_sym[sym] = sorted(
            fold_meta_by_sym[sym], key=lambda fm: fm["test_start"])
        dropped = before - len(fold_meta_by_sym[sym])
        if dropped:
            print(f"  [{sym}] dropped {dropped} fold(s) not shared by all symbols",
                  flush=True)
    print(f"  common folds to score: {len(common_windows)}", flush=True)

    # ── OHLC load (once) ──────────────────────────────────────────────
    ohlc_by_sym: dict[str, pd.DataFrame] = {sym: load_ohlc(sym, frequency)
                                            for sym in symbols}

    # ── Allocation config ─────────────────────────────────────────────
    alloc_cfg = AllocationConfig(
        symbols=list(symbols),
        max_gross_exposure_pct=float(pconf["allocation"]["max_gross_exposure_pct"]),
        max_per_symbol_pct=float(pconf["allocation"]["max_per_symbol_pct"]),
        sizing_mode="equal",
    )

    # ── Sample combinations ───────────────────────────────────────────
    combos = sample_combinations(k_per_sym, m=args.n_combinations, seed=args.seed)
    print(f"\nScoring {len(combos)} combinations "
          f"(baseline all-top-1 is row 0)...\n", flush=True)

    ctx = {
        "symbols": symbols,
        "params_by_sym": params_by_sym,
        "fold_meta_by_sym": fold_meta_by_sym,
        "ohlc_by_sym": ohlc_by_sym,
        "allocation_cfg": alloc_cfg,
        "initial_equity": float(args.initial_equity),
        "long_only": bool(args.long_only),
    }

    t0 = time.time()
    results: list[dict] = []

    if args.workers <= 1:
        for i, combo in enumerate(combos):
            r = score_combination(
                combo_idx=tuple(int(x) for x in combo),
                symbols=symbols,
                params_by_sym=params_by_sym,
                fold_meta_by_sym=fold_meta_by_sym,
                ohlc_by_sym=ohlc_by_sym,
                allocation_cfg=alloc_cfg,
                initial_equity=float(args.initial_equity),
                long_only=bool(args.long_only),
            )
            results.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(combos):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(combos) - (i + 1)) / rate if rate > 0 else 0.0
                print(f"  [{i+1}/{len(combos)}] rate={rate:.1f}/s  eta={eta/60:.1f} min",
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init, initargs=(ctx,)) as ex:
            tasks = [[int(x) for x in c] for c in combos]
            futures = {ex.submit(_worker_score, t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(combos):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(combos) - done) / rate if rate > 0 else 0.0
                    print(f"  [{done}/{len(combos)}] rate={rate:.1f}/s  "
                          f"eta={eta/60:.1f} min", flush=True)

    dur_min = (time.time() - t0) / 60
    print(f"\nAll combinations scored in {dur_min:.1f} min", flush=True)

    # ── Rank ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values("portfolio_score", ascending=False).reset_index(drop=True)
    df.to_csv(run_root / "combinations_scored.csv", index=False)

    winner = df.iloc[0]
    baseline_row = next((r for r in results if r["combo_idx"] == [0] * len(symbols)),
                        None)

    print("\n===== Top 5 combinations =====")
    cols = ["combo_idx", "portfolio_score", "mean_portfolio_sharpe",
            "std_portfolio_sharpe", "min_portfolio_sharpe",
            "pct_pos_folds", "min_portfolio_dd"]
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(df[cols].head(5).to_string(index=False))

    if baseline_row is not None:
        print(f"\nBaseline all-top-1: score={baseline_row['portfolio_score']:.4f}  "
              f"mean_sharpe={baseline_row['mean_portfolio_sharpe']:.4f}")

    winner_params_by_sym: dict[str, dict] = {}
    for sym, idx in zip(symbols, winner["combo_idx"]):
        winner_params_by_sym[sym] = params_by_sym[sym][int(idx)]

    summary = {
        "run_root": str(run_root),
        "symbols": symbols,
        "frequency": frequency,
        "long_only": bool(args.long_only),
        "n_combinations_scored": int(len(results)),
        "winner_combo_idx": [int(x) for x in winner["combo_idx"]],
        "winner_portfolio_metrics": {
            "portfolio_score": float(winner["portfolio_score"]),
            "mean_portfolio_sharpe": float(winner["mean_portfolio_sharpe"]),
            "std_portfolio_sharpe": float(winner["std_portfolio_sharpe"]),
            "min_portfolio_sharpe": float(winner["min_portfolio_sharpe"]),
            "median_portfolio_sharpe": float(winner["median_portfolio_sharpe"]),
            "pct_pos_folds": float(winner["pct_pos_folds"]),
            "mean_portfolio_dd": float(winner["mean_portfolio_dd"]),
            "min_portfolio_dd": float(winner["min_portfolio_dd"]),
            "fold_sharpes": list(map(float, winner["fold_sharpes"])),
            "fold_dds": list(map(float, winner["fold_dds"])),
            "fold_n_trades": list(map(int, winner["fold_n_trades"])),
        },
        "baseline_all_top1": None if baseline_row is None else {
            "portfolio_score": float(baseline_row["portfolio_score"]),
            "mean_portfolio_sharpe": float(baseline_row["mean_portfolio_sharpe"]),
        },
        "winner_params_by_symbol": winner_params_by_sym,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {run_root / 'summary.json'}", flush=True)

    # ── Promote ──────────────────────────────────────────────────────
    # Safety guards: do not overwrite production active.json with parameters
    # that fail any of these gates (they'd be strictly worse than not trading).
    winner_sh = float(winner["mean_portfolio_sharpe"])
    winner_min_dd = float(winner["min_portfolio_dd"])
    beats_baseline = (baseline_row is None
                      or winner["portfolio_score"] >= baseline_row["portfolio_score"])
    positive_expectation = winner_sh > 0.0
    survivable_dd = winner_min_dd > -0.50

    gates = {
        "winner_beats_baseline": bool(beats_baseline),
        "winner_mean_sharpe_positive": bool(positive_expectation),
        "worst_fold_dd_above_-50%": bool(survivable_dd),
    }
    promote_ok = all(gates.values())
    if args.no_promote:
        print("\n--no-promote: skipping active.json updates.", flush=True)
    elif not promote_ok:
        print("\nNOT PROMOTING. Safety gates:", flush=True)
        for g, ok in gates.items():
            print(f"  {g}: {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"  (winner mean Sharpe={winner_sh:.3f}, min fold DD={winner_min_dd:.2%})",
              flush=True)
    else:
        print("\nPromoting winner to configs/symbols/<SYM>/active.json ...",
              flush=True)
        promote_winner(winner_params_by_sym, frequency,
                       long_only=bool(args.long_only),
                       source_tag="portfolio_param_search")


if __name__ == "__main__":
    main()
