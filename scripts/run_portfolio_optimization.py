"""Top-level orchestrator: fan out Stage 1 per-symbol optimisation, then run
the Stage 2 portfolio-Sharpe combination search.

Stage 1: launches N (one per symbol) `scripts/robust_param_selection.py`
         subprocesses in parallel, each writing to
         ``<run-root>/<SYM>/``. Each worker runs Phase 0 (fresh walk-forward
         fold predictions) through Phase 4 (diverse-winner selection). We
         pass ``--skip-phase5`` because Stage 2 is the sole promoter.

Stage 2: invokes `scripts/portfolio_param_search.py --run-root <run-root>`
         which selects the cross-symbol combination maximising portfolio
         Sharpe and auto-promotes to configs/symbols/<SYM>/active.json.

Usage::

    python scripts/run_portfolio_optimization.py --long-only
    python scripts/run_portfolio_optimization.py --long-only \\
        --stage1-workers 3 --n-combinations 1500
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_portfolio_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_active_configs(symbols: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        src = REPO_ROOT / "configs" / "symbols" / sym / "active.json"
        if src.exists():
            shutil.copy2(src, dest / f"{sym}_active.json")


def launch_stage1_worker(
    sym: str, sym_dir: Path, *, long_only: bool,
    n_samples: int, stage_b_top: int, n_diverse: int,
    first_test_start: str, window_end: str,
    test_span_months: int, train_lookback_months: int,
    log_path: Path,
) -> subprocess.Popen:
    sym_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",  # unbuffered for live log tailing
        str(REPO_ROOT / "scripts" / "robust_param_selection.py"),
        "--symbol", sym,
        "--output-dir", str(sym_dir),
        "--n-samples", str(n_samples),
        "--stage-b-top", str(stage_b_top),
        "--n-diverse", str(n_diverse),
        "--first-test-start", first_test_start,
        "--window-end", window_end,
        "--test-span-months", str(test_span_months),
        "--train-lookback-months", str(train_lookback_months),
        "--skip-phase5",  # Stage 2 handles promotion
    ]
    if long_only:
        cmd.append("--long-only")
    env = None  # inherit
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)
    log_f.write(f"# CMD: {' '.join(cmd)}\n")
    log_f.flush()
    popen = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    popen._log_fh = log_f  # keep reference alive
    return popen


def wait_stage1(procs: dict[str, subprocess.Popen]) -> dict[str, int]:
    """Wait for all Stage 1 workers; return exit codes."""
    codes: dict[str, int] = {}
    pending = dict(procs)
    while pending:
        for sym in list(pending):
            rc = pending[sym].poll()
            if rc is not None:
                codes[sym] = rc
                try:
                    pending[sym]._log_fh.close()
                except Exception:
                    pass
                del pending[sym]
                status = "OK" if rc == 0 else f"FAILED ({rc})"
                print(f"[{time.strftime('%H:%M:%S')}] Stage 1 {sym}: {status}",
                      flush=True)
        if pending:
            time.sleep(15)
    return codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio-config", type=str,
                    default=str(REPO_ROOT / "configs" / "portfolio.json"))
    ap.add_argument("--run-root", type=str, default=None,
                    help="Override run root directory. Default: runs/portfolio_opt_<ts>/")
    ap.add_argument("--long-only", action="store_true", default=True)
    ap.add_argument("--stage1-workers", type=int, default=6,
                    help="Max concurrent Stage 1 subprocesses")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="Stage 1 LHS sample count")
    ap.add_argument("--stage-b-top", type=int, default=200)
    ap.add_argument("--n-diverse", type=int, default=25)
    ap.add_argument("--first-test-start", type=str, default="2024-01-01")
    ap.add_argument("--window-end", type=str, default="2026-04-02")
    ap.add_argument("--test-span-months", type=int, default=3)
    ap.add_argument("--train-lookback-months", type=int, default=12)
    ap.add_argument("--n-combinations", type=int, default=2000)
    ap.add_argument("--k-per-symbol", type=int, default=25)
    ap.add_argument("--stage2-workers", type=int, default=4)
    ap.add_argument("--skip-stage1", action="store_true",
                    help="Skip Stage 1 (reuse existing <run-root>/<SYM>/ dirs)")
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    pconf = load_portfolio_config(Path(args.portfolio_config))
    symbols: list[str] = list(pconf["symbols"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) if args.run_root else (
        REPO_ROOT / "runs" / f"portfolio_opt_{ts}")
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"===== Portfolio optimisation orchestrator =====")
    print(f"run-root: {run_root}")
    print(f"symbols:  {symbols}")
    print(f"stage1 workers: {args.stage1_workers}")
    print(f"long-only: {args.long_only}")

    # 1) snapshot current active.json files for reversibility
    snapshot_active_configs(symbols, run_root / "backup")
    print(f"Snapshot -> {run_root / 'backup'}")

    # 2) write run manifest
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbols": symbols,
        "long_only": bool(args.long_only),
        "stage1_workers": int(args.stage1_workers),
        "stage1_args": {
            "n_samples": int(args.n_samples),
            "stage_b_top": int(args.stage_b_top),
            "n_diverse": int(args.n_diverse),
            "first_test_start": args.first_test_start,
            "window_end": args.window_end,
            "test_span_months": int(args.test_span_months),
            "train_lookback_months": int(args.train_lookback_months),
        },
        "stage2_args": {
            "n_combinations": int(args.n_combinations),
            "k_per_symbol": int(args.k_per_symbol),
            "stage2_workers": int(args.stage2_workers),
            "no_promote": bool(args.no_promote),
        },
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")

    # 3) Stage 1 fan-out
    if not args.skip_stage1:
        print(f"\n=== Stage 1 — per-symbol candidate mining ===")
        # Simple wave-scheduler: start up to `stage1_workers` at once.
        pending_syms = list(symbols)
        running: dict[str, subprocess.Popen] = {}
        exit_codes: dict[str, int] = {}
        try:
            while pending_syms or running:
                while pending_syms and len(running) < args.stage1_workers:
                    sym = pending_syms.pop(0)
                    sym_dir = run_root / sym
                    log_path = run_root / f"{sym}_stage1.log"
                    proc = launch_stage1_worker(
                        sym, sym_dir,
                        long_only=args.long_only,
                        n_samples=args.n_samples,
                        stage_b_top=args.stage_b_top,
                        n_diverse=args.n_diverse,
                        first_test_start=args.first_test_start,
                        window_end=args.window_end,
                        test_span_months=args.test_span_months,
                        train_lookback_months=args.train_lookback_months,
                        log_path=log_path,
                    )
                    running[sym] = proc
                    print(f"[{time.strftime('%H:%M:%S')}] launched {sym}  "
                          f"(pid={proc.pid}, log={log_path.name})", flush=True)

                # Poll
                for sym in list(running):
                    rc = running[sym].poll()
                    if rc is not None:
                        exit_codes[sym] = rc
                        try:
                            running[sym]._log_fh.close()
                        except Exception:
                            pass
                        del running[sym]
                        status = "OK" if rc == 0 else f"FAILED ({rc})"
                        print(f"[{time.strftime('%H:%M:%S')}] Stage 1 {sym}: {status}",
                              flush=True)
                if running:
                    time.sleep(20)
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating running workers...", flush=True)
            for proc in running.values():
                try:
                    proc.terminate()
                except Exception:
                    pass
            raise

        failed = [s for s, rc in exit_codes.items() if rc != 0]
        if failed:
            print(f"\nERROR: Stage 1 failed for: {failed}", flush=True)
            sys.exit(3)
        print("\nStage 1 complete for all symbols.", flush=True)

    # 4) Stage 2
    print(f"\n=== Stage 2 — portfolio combination search ===")
    stage2_cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "portfolio_param_search.py"),
        "--run-root", str(run_root),
        "--portfolio-config", str(args.portfolio_config),
        "--n-combinations", str(args.n_combinations),
        "--k-per-symbol", str(args.k_per_symbol),
        "--workers", str(args.stage2_workers),
    ]
    if args.long_only:
        stage2_cmd.append("--long-only")
    if args.no_promote:
        stage2_cmd.append("--no-promote")

    stage2_log = run_root / "stage2.log"
    with open(stage2_log, "w", encoding="utf-8", buffering=1) as lf:
        lf.write(f"# CMD: {' '.join(stage2_cmd)}\n")
        rc = subprocess.run(stage2_cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(REPO_ROOT)).returncode
    if rc != 0:
        print(f"ERROR: Stage 2 failed (rc={rc}). See {stage2_log}", flush=True)
        sys.exit(4)
    print(f"Stage 2 complete. summary -> {run_root / 'summary.json'}", flush=True)

    # 5) echo summary
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        m = s["winner_portfolio_metrics"]
        b = s.get("baseline_all_top1") or {}
        print("\n===== Winner =====")
        print(f"  portfolio_score   : {m['portfolio_score']:.4f}  "
              f"(baseline all-top-1 = {b.get('portfolio_score', float('nan')):.4f})")
        print(f"  mean Sharpe       : {m['mean_portfolio_sharpe']:.4f}")
        print(f"  std Sharpe        : {m['std_portfolio_sharpe']:.4f}")
        print(f"  min Sharpe (fold) : {m['min_portfolio_sharpe']:.4f}")
        print(f"  pct pos folds     : {m['pct_pos_folds']:.2%}")
        print(f"  min fold DD       : {m['min_portfolio_dd']:.2%}")


if __name__ == "__main__":
    main()
