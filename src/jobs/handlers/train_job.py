from __future__ import annotations

from pathlib import Path

from src.core.contracts import TrainRequest, TrainResult
from src.jobs import store
from src.train import train_model


def _maybe_copy(job_id: str, path: str | None) -> tuple[str | None, str | None]:
    """Copy a file into artifacts/.

    Returns (filename, dest_path_str) or (None, None) when input is missing.
    """

    if not path:
        return None, None

    p = Path(path)
    if not p.exists():
        return None, None

    dest_path = store.copy_file_artifact(job_id, p)
    return p.name, dest_path


def run(job_id: str, request: TrainRequest) -> TrainResult:
    """Execute a training job and persist the produced model artifacts.

    Artifacts (best-effort):
    - artifacts/<model_filename>
    - artifacts/<model_filename_without_ext>.metrics.json
    - artifacts/<bias_correction_filename>

    Result:
    - result.json (small, stable fields)
    """

    result = train_model(
        frequency=request.frequency,
        tsteps=int(request.tsteps),
        lstm_units=request.lstm_units,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        current_batch_size=request.batch_size,
        n_lstm_layers=request.n_lstm_layers,
        stateful=request.stateful,
        features_to_use=request.features_to_use,
        train_start_date=request.train_start_date,
        train_end_date=request.train_end_date,
    )

    if result is None:
        raise RuntimeError("Training failed or not enough data.")

    validation_loss, model_path, bias_path = result

    model_filename, _ = _maybe_copy(job_id, str(model_path))
    if not model_filename:
        # The model is expected to exist because train_model just saved it.
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Copy optional sidecars next to the model (created by train_and_save_model).
    metrics_filename = None
    metrics_path = Path(str(model_path))
    if metrics_path.suffix == ".keras":
        candidate = metrics_path.with_suffix("").with_suffix(".metrics.json")
        metrics_filename, _ = _maybe_copy(job_id, str(candidate))

    bias_filename, _ = _maybe_copy(job_id, str(bias_path) if bias_path else None)

    out = TrainResult(
        validation_loss=float(validation_loss),
        model_filename=str(model_filename),
        bias_correction_filename=str(bias_filename) if bias_filename else None,
        metrics_filename=str(metrics_filename) if metrics_filename else None,
    )

    store.write_result(job_id, out.to_dict())
    return out
