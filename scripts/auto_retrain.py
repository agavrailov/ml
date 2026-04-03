"""Autonomous model retraining and auto-promotion pipeline.

Intended to run every Saturday at 2 AM ET, before the IB Gateway weekend
restart (~11 PM Sat ET).

Pipeline
--------
  1. Refresh processed data  (no live IBKR connection needed)
  2. Train a new model with the current best hyperparameters
  3. Gate 1 — RMSE   : new val_loss ≤ old_val_loss × RMSE_TOLERANCE
  4. Gate 2 — Signal : Pearson correlation ≥ MIN_CORRELATION
  5. Provisional promotion → update best_hyperparameters.json
  6. Generate per-bar predictions CSV for the new model
  7. Gate 3 — Walk-forward: mean fold Sharpe ≥ MIN_WALKFORWARD_SHARPE
  8. Confirmed: keep promotion.  Failed: roll back best_hyperparameters.json.
  9. Write JSON report to  runs/auto_retrain/<run_id>/report.json

Exit codes
----------
  0 — pipeline completed normally (promotion may or may not have happened)
  1 — unexpected crash
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Suppress matplotlib GUI before any downstream import touches it
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Gate thresholds  (the only place to tune them)
# ---------------------------------------------------------------------------

# New model RMSE (sqrt of val_loss) must be ≤ old × this factor.
# 1.05 = allow up to 5 % regression in RMSE; keeps models close in quality.
RMSE_TOLERANCE: float = 1.05

# Minimum Pearson correlation between predicted and actual log-returns.
# Near-zero means the model has no directional signal whatsoever.
MIN_CORRELATION: float = 0.03

# Mean Sharpe ratio across all walk-forward folds.
MIN_WALKFORWARD_SHARPE: float = 1.0

# Fraction of walk-forward folds that must be individually profitable (return > 0).
MIN_PROFITABLE_FOLD_FRACTION: float = 0.50

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FREQUENCY: str = "60min"
DEFAULT_TSTEPS: int = 5
DEFAULT_SYMBOL: str = "nvda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_s() -> float:
    return time.monotonic()


def _load_best_hps(base_dir: str) -> dict:
    path = os.path.join(base_dir, "best_hyperparameters.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else {}
    except Exception:
        return {}


def _save_best_hps(base_dir: str, hps: dict) -> None:
    path = os.path.join(base_dir, "best_hyperparameters.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hps, f, indent=4)
    os.replace(tmp, path)


def _old_val_loss(hps: dict, frequency: str, tsteps: int) -> float | None:
    try:
        return float(hps[frequency][str(tsteps)]["validation_loss"])
    except (KeyError, TypeError, ValueError):
        return None


def _write_report(report_dir: Path, report: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[auto_retrain] Report written to {path}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_data_pipeline(*, log: list[str]) -> dict[str, Any]:
    """Run the daily data pipeline (no ingestion — offline mode)."""
    t0 = _now_s()
    log.append("Stage 1: data pipeline (skip_ingestion=True)")
    try:
        from src.daily_data_agent import run_daily_pipeline
        run_daily_pipeline(skip_ingestion=True)
        dur = _now_s() - t0
        log.append(f"  Data pipeline finished in {dur:.1f}s")
        return {"status": "ok", "duration_s": round(dur, 1)}
    except Exception as exc:
        dur = _now_s() - t0
        log.append(f"  Data pipeline FAILED: {exc}")
        return {"status": "error", "error": repr(exc), "duration_s": round(dur, 1)}


def _stage_train(
    *,
    frequency: str,
    tsteps: int,
    log: list[str],
) -> dict[str, Any]:
    """Train a new model. Returns stage dict + (val_loss, model_path, bias_path)."""
    t0 = _now_s()
    log.append(f"Stage 2: training  frequency={frequency}  tsteps={tsteps}")
    try:
        from src.train import train_model
        val_loss, model_path, bias_path = train_model(
            frequency=frequency,
            tsteps=tsteps,
        )
        dur = _now_s() - t0
        log.append(f"  Training finished in {dur:.1f}s  val_loss={val_loss:.6f}")
        log.append(f"  New model: {model_path}")
        return {
            "status": "ok",
            "new_val_loss": val_loss,
            "new_model_path": model_path,
            "bias_correction_path": bias_path,
            "duration_s": round(dur, 1),
        }
    except Exception as exc:
        dur = _now_s() - t0
        log.append(f"  Training FAILED: {exc}\n{traceback.format_exc()}")
        return {"status": "error", "error": repr(exc), "duration_s": round(dur, 1)}


def _stage_evaluate(
    *,
    model_path: str,
    bias_path: str | None,
    frequency: str,
    tsteps: int,
    log: list[str],
) -> dict[str, Any]:
    """Evaluate new model: returns (mae, correlation) and stage dict."""
    t0 = _now_s()
    log.append("Stage 3: model evaluation (MAE + correlation)")
    try:
        from src.config import get_run_hyperparameters
        from src.evaluate_model import evaluate_model_performance

        hps = get_run_hyperparameters(frequency, tsteps)
        lstm_units = hps.get("lstm_units")
        n_lstm_layers = hps.get("n_lstm_layers")
        features = hps.get("features_to_use")

        result = evaluate_model_performance(
            model_path=model_path,
            frequency=frequency,
            tsteps=tsteps,
            lstm_units=lstm_units,
            n_lstm_layers=n_lstm_layers,
            features_to_use=features,
            bias_correction_path=bias_path,
        )
        # evaluate_model_performance can return None on data errors (code has
        # bare return statements without values); treat that as failure.
        if result is None:
            dur = _now_s() - t0
            log.append("  Evaluation returned None (data error)")
            return {"status": "error", "error": "evaluate_model_performance returned None",
                    "duration_s": round(dur, 1)}

        mae, correlation = result
        dur = _now_s() - t0
        log.append(f"  Evaluation finished in {dur:.1f}s  mae={mae:.6f}  correlation={correlation:.4f}")
        return {
            "status": "ok",
            "mae": float(mae),
            "correlation": float(correlation),
            "duration_s": round(dur, 1),
        }
    except Exception as exc:
        dur = _now_s() - t0
        log.append(f"  Evaluation FAILED: {exc}\n{traceback.format_exc()}")
        return {"status": "error", "error": repr(exc), "duration_s": round(dur, 1)}


def _stage_generate_predictions(
    *,
    frequency: str,
    symbol: str,
    predictions_path: str,
    log: list[str],
) -> dict[str, Any]:
    """Generate per-bar prediction CSV for the (now-provisional) active model."""
    t0 = _now_s()
    log.append(f"Stage 5: generating predictions CSV -> {predictions_path}")
    try:
        # Delete stale checkpoint so predictions are regenerated from the new model.
        checkpoint = os.path.join(
            "backtests", f"{symbol.lower()}_{frequency}_model_predictions_checkpoint.csv"
        )
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
            log.append(f"  Deleted stale checkpoint: {checkpoint}")

        from scripts.generate_predictions_csv import generate_predictions_for_csv
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        generate_predictions_for_csv(frequency=frequency, output_path=predictions_path)

        dur = _now_s() - t0
        log.append(f"  Predictions generated in {dur:.1f}s")
        return {"status": "ok", "predictions_path": predictions_path, "duration_s": round(dur, 1)}
    except Exception as exc:
        dur = _now_s() - t0
        log.append(f"  Prediction generation FAILED: {exc}\n{traceback.format_exc()}")
        return {"status": "error", "error": repr(exc), "duration_s": round(dur, 1)}


def _stage_walkforward(
    *,
    frequency: str,
    tsteps: int,
    symbol: str,
    predictions_path: str,
    log: list[str],
) -> dict[str, Any]:
    """Run walk-forward backtests on the new predictions."""
    t0 = _now_s()
    log.append("Stage 6: walk-forward backtest")
    try:
        from scripts.run_walkforward_backtest import run_walkforward
        df = run_walkforward(
            frequency=frequency,
            tsteps=tsteps,
            symbol=symbol,
            test_span_months=3,
            train_lookback_months=24,
            min_lookback_months=18,
            t_start=None,
            t_end=None,
            predictions_csv=predictions_path,
            first_test_start="2023-07-01",
        )
        dur = _now_s() - t0

        if df.empty:
            log.append("  Walk-forward returned empty DataFrame")
            return {"status": "error", "error": "empty results", "duration_s": round(dur, 1)}

        folds = df.to_dict(orient="records")
        sharpe_values = [r["sharpe_ratio"] for r in folds if r.get("sharpe_ratio") is not None]
        mean_sharpe = float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0
        profitable = sum(1 for r in folds if r.get("total_return", 0.0) > 0)
        profitable_fraction = profitable / len(folds) if folds else 0.0

        log.append(
            f"  Walk-forward finished in {dur:.1f}s  "
            f"n_folds={len(folds)}  mean_sharpe={mean_sharpe:.2f}  "
            f"profitable_folds={profitable}/{len(folds)}"
        )
        return {
            "status": "ok",
            "mean_sharpe": mean_sharpe,
            "n_folds": len(folds),
            "profitable_fraction": profitable_fraction,
            "folds": folds,
            "duration_s": round(dur, 1),
        }
    except Exception as exc:
        dur = _now_s() - t0
        log.append(f"  Walk-forward FAILED: {exc}\n{traceback.format_exc()}")
        return {"status": "error", "error": repr(exc), "duration_s": round(dur, 1)}


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def _check_rmse_gate(
    new_val_loss: float,
    old_val_loss: float | None,
    log: list[str],
) -> tuple[bool, dict]:
    new_rmse = math.sqrt(max(new_val_loss, 1e-12))
    if old_val_loss is None:
        log.append(f"  RMSE gate: PASS (no baseline — first model)  new_rmse={new_rmse:.6f}")
        return True, {"status": "passed", "reason": "no_baseline", "new_rmse": new_rmse,
                       "old_rmse": None, "tolerance": RMSE_TOLERANCE}

    old_rmse = math.sqrt(max(old_val_loss, 1e-12))
    threshold = old_rmse * RMSE_TOLERANCE
    passed = new_rmse <= threshold
    status = "passed" if passed else "failed"
    log.append(
        f"  RMSE gate: {status.upper()}  new={new_rmse:.6f}  old={old_rmse:.6f}  "
        f"threshold={threshold:.6f}  tolerance={RMSE_TOLERANCE}"
    )
    return passed, {
        "status": status,
        "new_rmse": new_rmse,
        "old_rmse": old_rmse,
        "threshold": threshold,
        "tolerance": RMSE_TOLERANCE,
    }


def _check_correlation_gate(correlation: float, log: list[str]) -> tuple[bool, dict]:
    passed = correlation >= MIN_CORRELATION
    status = "passed" if passed else "failed"
    log.append(
        f"  Correlation gate: {status.upper()}  correlation={correlation:.4f}  "
        f"threshold={MIN_CORRELATION}"
    )
    return passed, {
        "status": status,
        "correlation": correlation,
        "threshold": MIN_CORRELATION,
    }


def _check_walkforward_gate(wf_stage: dict, log: list[str]) -> tuple[bool, dict]:
    mean_sharpe = wf_stage.get("mean_sharpe", 0.0)
    profitable_fraction = wf_stage.get("profitable_fraction", 0.0)

    sharpe_ok = mean_sharpe >= MIN_WALKFORWARD_SHARPE
    profitable_ok = profitable_fraction >= MIN_PROFITABLE_FOLD_FRACTION
    passed = sharpe_ok and profitable_ok
    status = "passed" if passed else "failed"

    reasons = []
    if not sharpe_ok:
        reasons.append(f"mean_sharpe={mean_sharpe:.2f} < {MIN_WALKFORWARD_SHARPE}")
    if not profitable_ok:
        reasons.append(f"profitable_fraction={profitable_fraction:.2f} < {MIN_PROFITABLE_FOLD_FRACTION}")

    log.append(
        f"  Walk-forward gate: {status.upper()}  mean_sharpe={mean_sharpe:.2f}  "
        f"profitable_fraction={profitable_fraction:.2f}"
        + (f"  reasons: {'; '.join(reasons)}" if reasons else "")
    )
    return passed, {
        "status": status,
        "mean_sharpe": mean_sharpe,
        "profitable_fraction": profitable_fraction,
        "sharpe_threshold": MIN_WALKFORWARD_SHARPE,
        "profitable_threshold": MIN_PROFITABLE_FOLD_FRACTION,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# best_hyperparameters.json promotion / rollback
# ---------------------------------------------------------------------------

def _promote_new_model(
    *,
    base_dir: str,
    frequency: str,
    tsteps: int,
    new_val_loss: float,
    model_path: str,
    bias_path: str | None,
    old_hps: dict,
    log: list[str],
) -> dict:
    """Write new model into best_hyperparameters.json (provisional or final)."""
    from src.config import get_run_hyperparameters
    hps = get_run_hyperparameters(frequency, tsteps)

    new_hps = dict(old_hps)
    if frequency not in new_hps:
        new_hps[frequency] = {}
    new_hps[frequency][str(tsteps)] = {
        "validation_loss": new_val_loss,
        "model_filename": os.path.basename(model_path),
        "lstm_units": hps.get("lstm_units"),
        "learning_rate": hps.get("learning_rate"),
        "epochs": hps.get("epochs"),
        "batch_size": hps.get("batch_size"),
        "n_lstm_layers": hps.get("n_lstm_layers"),
        "stateful": hps.get("stateful", True),
        "optimizer_name": hps.get("optimizer_name", "rmsprop"),
        "loss_function": hps.get("loss_function", "mse"),
        "bias_correction_filename": os.path.basename(bias_path) if bias_path else None,
        "promoted_at_utc": _ts(),
    }
    _save_best_hps(base_dir, new_hps)
    log.append(f"  Promoted {os.path.basename(model_path)} -> best_hyperparameters.json")
    return new_hps


def _rollback_hps(base_dir: str, old_hps: dict, log: list[str]) -> None:
    """Restore best_hyperparameters.json to the pre-run state."""
    _save_best_hps(base_dir, old_hps)
    log.append("  Rolled back best_hyperparameters.json to previous state")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_auto_retrain(
    *,
    frequency: str = DEFAULT_FREQUENCY,
    tsteps: int = DEFAULT_TSTEPS,
    symbol: str = DEFAULT_SYMBOL,
    dry_run: bool = False,
) -> dict:
    """Run the full retraining pipeline. Returns the report dict."""
    from src.config import BASE_DIR

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = Path(BASE_DIR) / "runs" / "auto_retrain" / run_id
    started_utc = _ts()
    log: list[str] = []
    stages: dict[str, Any] = {}
    promoted = False
    error_msg: str | None = None

    print(f"\n{'='*60}")
    print(f"[auto_retrain] Starting run {run_id}")
    print(f"[auto_retrain] frequency={frequency}  tsteps={tsteps}  dry_run={dry_run}")
    print(f"{'='*60}\n")

    try:
        # Snapshot current state so we can roll back on failure
        old_hps = _load_best_hps(BASE_DIR)
        old_val_loss = _old_val_loss(old_hps, frequency, tsteps)
        log.append(f"Baseline val_loss={old_val_loss}")

        # -------------------------------------------------------------------
        # Stage 1: Data pipeline
        # -------------------------------------------------------------------
        s1 = _stage_data_pipeline(log=log)
        stages["data_pipeline"] = s1
        if s1["status"] != "ok":
            log.append("Aborting: data pipeline failed.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=s1["error"])

        if dry_run:
            log.append("dry_run=True: stopping after data pipeline.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=None)

        # -------------------------------------------------------------------
        # Stage 2: Train
        # -------------------------------------------------------------------
        s2 = _stage_train(frequency=frequency, tsteps=tsteps, log=log)
        stages["training"] = s2
        if s2["status"] != "ok":
            log.append("Aborting: training failed.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=s2["error"])

        new_val_loss: float = s2["new_val_loss"]
        new_model_path: str = s2["new_model_path"]
        bias_path: str | None = s2.get("bias_correction_path")
        s2["old_val_loss"] = old_val_loss

        # -------------------------------------------------------------------
        # Gate 1: RMSE
        # -------------------------------------------------------------------
        log.append("Gate 1: RMSE check")
        rmse_passed, rmse_result = _check_rmse_gate(new_val_loss, old_val_loss, log)
        stages["rmse_gate"] = rmse_result
        if not rmse_passed:
            log.append("RMSE gate FAILED — keeping existing model.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=None)

        # -------------------------------------------------------------------
        # Stage 3: Evaluate (MAE + correlation)
        # -------------------------------------------------------------------
        s3 = _stage_evaluate(
            model_path=new_model_path,
            bias_path=bias_path,
            frequency=frequency,
            tsteps=tsteps,
            log=log,
        )
        stages["evaluation"] = s3
        if s3["status"] != "ok":
            log.append("Evaluation failed — keeping existing model.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=s3["error"])

        # -------------------------------------------------------------------
        # Gate 2: Directional accuracy (correlation)
        # -------------------------------------------------------------------
        log.append("Gate 2: correlation check")
        corr_passed, corr_result = _check_correlation_gate(s3["correlation"], log)
        stages["correlation_gate"] = corr_result
        if not corr_passed:
            log.append("Correlation gate FAILED — keeping existing model.")
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=None)

        # -------------------------------------------------------------------
        # Stage 4: Provisional promotion (needed to generate predictions)
        # -------------------------------------------------------------------
        log.append("Stage 4: provisional promotion (gates 1+2 passed)")
        _promote_new_model(
            base_dir=BASE_DIR,
            frequency=frequency,
            tsteps=tsteps,
            new_val_loss=new_val_loss,
            model_path=new_model_path,
            bias_path=bias_path,
            old_hps=old_hps,
            log=log,
        )

        # -------------------------------------------------------------------
        # Stage 5: Generate predictions for the new model
        # -------------------------------------------------------------------
        predictions_path = os.path.join(
            BASE_DIR, "backtests", f"auto_retrain_{symbol.lower()}_{frequency}_predictions.csv"
        )
        s5 = _stage_generate_predictions(
            frequency=frequency,
            symbol=symbol,
            predictions_path=predictions_path,
            log=log,
        )
        stages["predictions"] = s5
        if s5["status"] != "ok":
            log.append("Prediction generation failed — rolling back.")
            _rollback_hps(BASE_DIR, old_hps, log)
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=s5["error"])

        # -------------------------------------------------------------------
        # Stage 6: Walk-forward backtest
        # -------------------------------------------------------------------
        s6 = _stage_walkforward(
            frequency=frequency,
            tsteps=tsteps,
            symbol=symbol,
            predictions_path=predictions_path,
            log=log,
        )
        stages["walkforward"] = s6
        if s6["status"] != "ok":
            log.append("Walk-forward failed — rolling back.")
            _rollback_hps(BASE_DIR, old_hps, log)
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=s6["error"])

        # -------------------------------------------------------------------
        # Gate 3: Walk-forward Sharpe
        # -------------------------------------------------------------------
        log.append("Gate 3: walk-forward Sharpe check")
        wf_passed, wf_result = _check_walkforward_gate(s6, log)
        stages["walkforward_gate"] = wf_result
        if not wf_passed:
            log.append("Walk-forward gate FAILED — rolling back to previous model.")
            _rollback_hps(BASE_DIR, old_hps, log)
            return _finalise(report_dir, run_id, started_utc, frequency, tsteps,
                             stages, promoted=False, log=log, error=None)

        # -------------------------------------------------------------------
        # All gates passed — promotion is confirmed
        # -------------------------------------------------------------------
        promoted = True
        log.append(
            f"ALL GATES PASSED — new model is now active: {os.path.basename(new_model_path)}"
        )
        print(f"\n[auto_retrain] ✓ NEW MODEL PROMOTED: {os.path.basename(new_model_path)}\n")

    except Exception as exc:
        error_msg = repr(exc)
        log.append(f"UNEXPECTED ERROR: {exc}\n{traceback.format_exc()}")
        print(f"[auto_retrain] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()

    return _finalise(
        report_dir, run_id, started_utc, frequency, tsteps,
        stages, promoted=promoted, log=log, error=error_msg,
    )


def _finalise(
    report_dir: Path,
    run_id: str,
    started_utc: str,
    frequency: str,
    tsteps: int,
    stages: dict,
    *,
    promoted: bool,
    log: list[str],
    error: str | None,
) -> dict:
    report = {
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": _ts(),
        "frequency": frequency,
        "tsteps": tsteps,
        "promoted": promoted,
        "stages": stages,
        "log": log,
        "error": error,
    }
    _write_report(report_dir, report)

    # Print concise summary to stdout so it lands in service logs
    status = "PROMOTED" if promoted else ("ERROR" if error else "NO_CHANGE")
    print(f"\n[auto_retrain] Run {run_id} finished — status={status}")
    for line in log:
        print(f"[auto_retrain]   {line}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous LSTM model retraining + auto-promotion pipeline."
    )
    parser.add_argument(
        "--frequency", default=DEFAULT_FREQUENCY,
        help=f"Bar frequency (default: {DEFAULT_FREQUENCY})",
    )
    parser.add_argument(
        "--tsteps", type=int, default=DEFAULT_TSTEPS,
        help=f"Model timesteps (default: {DEFAULT_TSTEPS})",
    )
    parser.add_argument(
        "--symbol", default=DEFAULT_SYMBOL,
        help=f"Symbol label for predictions/walkforward (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run data pipeline only; skip training and promotion",
    )
    args = parser.parse_args()

    report = run_auto_retrain(
        frequency=args.frequency,
        tsteps=args.tsteps,
        symbol=args.symbol,
        dry_run=args.dry_run,
    )
    sys.exit(0 if report.get("error") is None else 1)


if __name__ == "__main__":
    main()
