"""Per-symbol hyperparameter search for the LSTM model.

Fixes frequency=60min, tsteps=5, stateful=True, optimizer=rmsprop and
sweeps the architecture + loss function parameters that most affect model
quality.  After finding the best model it automatically regenerates the
prediction checkpoint and re-runs the strategy grid search.

Grid (32 combos):
    lstm_units:    [32, 64, 128, 256]
    n_lstm_layers: [1, 2]
    batch_size:    [64, 128]
    loss_function: ["mae", "mse"]

Usage (from repo root):
    python -m scripts.experiment_runner --symbol JPM
    python -m scripts.experiment_runner --symbol XOM --no-grid-search
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

FREQUENCY      = "60min"
TSTEPS         = 5
STATEFUL       = True
OPTIMIZER      = "rmsprop"
EPOCHS         = 20

LSTM_UNITS_OPTIONS    = [32, 64, 128, 256]
N_LSTM_LAYERS_OPTIONS = [1, 2]
BATCH_SIZE_OPTIONS    = [64, 128]
LOSS_OPTIONS          = ["mae", "mse"]


def _paths(symbol: str) -> dict[str, str]:
    s = symbol.lower()
    return {
        "results":    f"experiment_results_{s}.json",
        "checkpoint": os.path.join("backtests", f"{s}_{FREQUENCY}_model_predictions_checkpoint.csv"),
        "grid_csv":   os.path.join("backtests", f"{s}_grid_search_top20.csv"),
        "wf_csv":     os.path.join("backtests", f"{s}_walkforward_sweep_results.csv"),
    }


def run_search(symbol: str) -> list[dict]:
    from src.train import train_model

    combos = list(itertools.product(
        LSTM_UNITS_OPTIONS,
        N_LSTM_LAYERS_OPTIONS,
        BATCH_SIZE_OPTIONS,
        LOSS_OPTIONS,
    ))
    total = len(combos)
    print(f"{symbol} model search: {total} combos | freq={FREQUENCY} tsteps={TSTEPS} "
          f"stateful={STATEFUL} optimizer={OPTIMIZER}")
    print()

    results: list[dict] = []
    best_val_loss = float("inf")
    best_combo: dict = {}

    for idx, (lstm_units, n_layers, batch_size, loss_fn) in enumerate(combos, start=1):
        print(
            f"[{idx:2d}/{total}] lstm_units={lstm_units:3d} n_layers={n_layers} "
            f"batch={batch_size} loss={loss_fn}",
            flush=True,
        )

        try:
            result = train_model(
                symbol=symbol,
                frequency=FREQUENCY,
                tsteps=TSTEPS,
                lstm_units=lstm_units,
                n_lstm_layers=n_layers,
                current_batch_size=batch_size,
                stateful=STATEFUL,
                epochs=EPOCHS,
                learning_rate=0.01,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            continue

        if not result:
            continue

        val_loss, model_path, bias_path = result
        row = {
            "timestamp":      datetime.now().isoformat(),
            "symbol":         symbol,
            "frequency":      FREQUENCY,
            "tsteps":         TSTEPS,
            "lstm_units":     lstm_units,
            "n_lstm_layers":  n_layers,
            "batch_size":     batch_size,
            "loss_function":  loss_fn,
            "optimizer":      OPTIMIZER,
            "stateful":       STATEFUL,
            "val_loss":       val_loss,
            "model_path":     model_path,
            "bias_path":      bias_path,
        }
        results.append(row)

        marker = " <-- best" if val_loss < best_val_loss else ""
        print(f"    val_loss={val_loss:.6f}{marker}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_combo = row

    paths = _paths(symbol)
    with open(paths["results"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {paths['results']}")

    if best_combo:
        print(f"\n=== BEST MODEL ===")
        print(f"  lstm_units   = {best_combo['lstm_units']}")
        print(f"  n_layers     = {best_combo['n_lstm_layers']}")
        print(f"  batch_size   = {best_combo['batch_size']}")
        print(f"  loss         = {best_combo['loss_function']}")
        print(f"  val_loss     = {best_combo['val_loss']:.6f}")
        print(f"  model_path   = {best_combo['model_path']}")

    return results


def regenerate_checkpoint(symbol: str) -> None:
    """Re-run backtest in model mode to refresh the predictions checkpoint."""
    print(f"\n--- Regenerating {symbol} predictions checkpoint ---")
    cmd = [
        sys.executable, "-m", "src.backtest",
        "--symbol", symbol,
        "--frequency", FREQUENCY,
        "--prediction-mode", "model",
        "--report",
    ]
    subprocess.run(cmd, check=True)


def run_grid_search(symbol: str) -> None:
    """Re-run strategy grid search and save top-20 to CSV."""
    paths = _paths(symbol)
    print(f"\n--- Running strategy grid search for {symbol} ---")
    cmd = [
        sys.executable, "-m", "scripts.grid_search_strategy",
        "--symbol", symbol,
        "--no-save",
        "--save-results", paths["grid_csv"],
    ]
    subprocess.run(cmd, check=True)


def run_walkforward_sweep(symbol: str) -> None:
    """Rank top-20 param sets by OOS walk-forward Sharpe."""
    paths = _paths(symbol)
    print(f"\n--- Running walk-forward param sweep for {symbol} ---")
    cmd = [
        sys.executable, "-m", "scripts.walkforward_param_sweep",
        "--symbol", symbol,
        "--top", "20",
        "--test-span-months", "3",
        "--first-test-start", "2024-06-01",
        "--output", paths["wf_csv"],
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-symbol LSTM hyperparameter search + strategy optimisation.",
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to search (e.g. JPM, XOM)")
    parser.add_argument(
        "--no-grid-search", action="store_true",
        help="Skip strategy grid search and walk-forward after model search.",
    )
    parser.add_argument(
        "--grid-search-only", action="store_true",
        help="Skip model search; only regenerate checkpoint and re-run grid/walkforward.",
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()

    if not args.grid_search_only:
        run_search(symbol)

    if not args.no_grid_search:
        regenerate_checkpoint(symbol)
        run_grid_search(symbol)
        run_walkforward_sweep(symbol)
        paths = _paths(symbol)
        print(f"\nDone. Check {paths['wf_csv']} for OOS ranking.")
    else:
        print("\nModel search complete. Run with --grid-search-only to proceed to strategy search.")


if __name__ == "__main__":
    main()
