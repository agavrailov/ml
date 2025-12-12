import os
import json
import sys
from datetime import datetime # Moved import to top

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_latest_model_path, MODEL_REGISTRY_DIR, BASE_DIR


def promote_model(*, frequency: str | None = None, tsteps: int | None = None) -> None:
    """Promote the latest model in the registry as the active/best model.
    This updates ``best_hyperparameters.json`` so that helpers like
    :func:`get_latest_best_model_path` and :func:`get_active_model_path` resolve
    this model as the globally best one.
    """
    if frequency is None and tsteps is None:
        print("Attempting to promote the latest model...")
    else:
        print(f"Attempting to promote latest model for frequency={frequency}, tsteps={tsteps}...")

    # 1. Find the latest model (optionally filtered)
    latest_model_path = get_latest_model_path(frequency=frequency, tsteps=tsteps)

    if latest_model_path is None:
        print(f"Error: No models found in the registry at '{MODEL_REGISTRY_DIR}'.")
        print("Please train a model first using 'python src/train.py'.")
        return

    print(f"Found latest model: {latest_model_path}")

    # 2. Update best_hyperparameters.json so the platform resolves this model
    # as the globally best one. We infer (frequency, tsteps) from the filename
    # following the convention::
    #
    #     my_lstm_model_{frequency}_tsteps{tsteps}_{timestamp}.keras
    #
    best_hps_path = os.path.join(BASE_DIR, "best_hyperparameters.json")
    fname = os.path.basename(latest_model_path)

    freq_key = None
    tsteps_key = None
    if fname.startswith("my_lstm_model_") and "_tsteps" in fname:
        try:
            # Strip prefix and split at "_tsteps" to get frequency and remainder.
            rest = fname[len("my_lstm_model_"):]
            freq_part, after_freq = rest.split("_tsteps", 1)
            freq_key = freq_part
            # after_freq starts with the integer tsteps, then an underscore.
            tsteps_str = after_freq.split("_", 1)[0]
            if tsteps_str.isdigit():
                tsteps_key = tsteps_str
        except ValueError:
            freq_key = None
            tsteps_key = None

    if freq_key is None or tsteps_key is None:
        print(
            "Warning: Could not infer (frequency, tsteps) from model filename; "
            "not updating best_hyperparameters.json."
        )
        return

    # Load existing best_hyperparameters if present.
    if os.path.exists(best_hps_path):
        try:
            with open(best_hps_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            best_hps_overall = json.loads(content) if content else {}
        except json.JSONDecodeError:
            best_hps_overall = {}
    else:
        best_hps_overall = {}

    if freq_key not in best_hps_overall:
        best_hps_overall[freq_key] = {}

    # Force this model to be the best for its (frequency, tsteps) pair by
    # assigning a very low validation_loss.
    best_hps_overall[freq_key][tsteps_key] = {
        "validation_loss": 0.0,
        "model_filename": fname,
    }

    try:
        with open(best_hps_path, "w", encoding="utf-8") as f:
            json.dump(best_hps_overall, f, indent=4)
        print(f"Successfully promoted model. Updated best_hyperparameters.json at: {best_hps_path}")
    except Exception as e:
        print(f"Error: Failed to write best_hyperparameters.json at '{best_hps_path}'.")
        print(f"Details: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Promote a model from the registry.")
    parser.add_argument(
        "--frequency",
        type=str,
        default=None,
        help="Optional resample frequency filter (e.g. 1min, 15min, 60min).",
    )
    parser.add_argument(
        "--tsteps",
        type=int,
        default=None,
        help="Optional TSTEPS filter.",
    )

    args = parser.parse_args()
    promote_model(frequency=args.frequency, tsteps=args.tsteps)
