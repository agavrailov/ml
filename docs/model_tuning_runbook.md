# Model Tuning & Promotion Runbook

This document summarizes how to run hyperparameter experiments for the LSTM model, inspect the results, and how the "best" model is selected and promoted for use by the rest of the platform.

---

## 1. Running experiments

Use the `src.experiments` module to run a small grid over learning rate, LSTM units, and batch size. For example:

```bash
python -m src.experiments \
  --learning-rates 0.0001 0.0003 0.001 \
  --lstm-units 32 64 \
  --batch-sizes 32 64
```

Notes:

- All runs use the current defaults from `TrainingConfig` in `src/config.py` for parameters that are *not* part of the grid.
- Each configuration trains an LSTM model, evaluates it on a validation split, and saves the model file into `MODEL_REGISTRY_DIR` (e.g. `models/registry/`).

---

## 2. Where experiment results are stored

Every experiment appends JSON records to an append-only log:

- `experiments/experiments_log.jsonl` — one JSON object per line.

Each training run written by the grid has at least:

- `phase = "grid_search_initial"`
- `config` — the hyperparameters used
- `final_val_loss` — best validation loss for that run
- `model_path` — filesystem path to the saved `.keras` model

Additionally, at the end of each grid invocation a summary record is logged with:

- `phase = "grid_search_initial_summary"`
- `num_runs`, `best_final_val_loss`, `best_config`, `best_model_path`

You can inspect the log in a notebook, for example:

```python
import json
import pandas as pd
from pathlib import Path

project_root = Path().resolve().parent
log_path = project_root / "experiments" / "experiments_log.jsonl"

records = []
with log_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)
df_grid = df[df["phase"] == "grid_search_initial"]

# Show best few configs by validation loss
cols = ["final_val_loss", "timestamp", "model_path", "config"]
print(df_grid[cols].sort_values("final_val_loss").head(10))
```

---

## 3. How the best model is chosen

The experiments module maintains a single *global* notion of the best model seen so far across **all** logged grid runs:

1. Each grid invocation logs its own runs and summary to `experiments_log.jsonl`.
2. At the end of `run_lr_units_batch_grid`, `src.experiments` calls an internal helper that:
   - Loads all records from `experiments/experiments_log.jsonl`.
   - Filters to `phase == "grid_search_initial"` entries that contain `final_val_loss`, `config`, and `model_path`.
   - Selects the configuration with the lowest `final_val_loss`.
3. That global best is then persisted to:
   - `experiments/best_config.json`
   - `models/active_model.txt` (via `ACTIVE_MODEL_PATH_FILE`).

The `experiments/best_config.json` file has the form:

```json
{
  "final_val_loss": <float>,
  "config": { ... hyperparameters ... },
  "model_path": ".../models/registry/my_lstm_model_... .keras"
}
```

The `models/active_model.txt` file contains a single line: the filesystem path to the currently promoted model.

---

## 4. How the platform uses the active model

Runtime code that needs a model for predictions or backtests should resolve the active model path via `src.config` and load it using `src.model`:

```python
from src.config import get_active_model_path
from src.model import load_model

model_path = get_active_model_path()
model = load_model(model_path)
```

This ensures that:

- Whatever model was last selected as globally best by the experiments module is used consistently.
- You do **not** need to modify `config.py` or any source files when promoting a new model; rerunning experiments and letting them update `best_config.json` and `active_model.txt` is sufficient.

---

## 5. Typical workflow

1. Run one or more grids with `python -m src.experiments ...`.
2. After they finish, inspect `experiments/best_config.json` or the log in a notebook if you want more detail.
3. The active model for all consumers is automatically updated via `models/active_model.txt`.
4. Use the standard prediction / backtest CLIs, which will transparently load the active model via `get_active_model_path()`.
