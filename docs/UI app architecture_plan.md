# Problem statement
`src/app.py` is a monolithic Streamlit script that mixes UI rendering, long-running computation (train/backtest/optimize), file/state persistence, and operational live-trading controls. This creates high coupling, frequent regressions, slow/expensive refactors, and instability (Streamlit reruns interrupt work, hidden side effects).
# Current state (observed in repo)
* Streamlit UI (`src/app.py`) owns end-to-end workflows: Live ops dashboard, data ingestion/quality, experiments, training/promote, backtest/optimize, walk-forward.
* Live trading already runs out-of-process via `python -m src.ibkr_live_session`, writing JSONL logs under `ui_state/live/` and using a file-based kill switch (`ui_state/live/KILL_SWITCH`). The UI reads these logs.
* Training/backtest logic exists in reusable modules (e.g. `src/train.py`, `src/backtest.py` / `src/backtest_engine.py`), but the UI still orchestrates and persists history/config itself.
* Config is partially structured in `src/config.py`, but the UI also edits `config.py` on disk via regex substitution, which is a major source of hidden side effects and regressions.
# Target architecture (lightweight, single-user optimized)
The goal is to keep development fast (Streamlit stays), but enforce strong boundaries so refactors are cheap and regressions are contained.
## 1) Core (pure library)
A set of modules that are:
* importable from CLI, jobs, tests, and UI
* deterministic given explicit inputs
* free of Streamlit/session_state
* free of “write config.py” mutations
Implementation strategy (minimal-diff):
* Do not immediately move all code.
* Instead, define a “core contract layer” (dataclasses + functions) and gradually route UI/jobs through it.
Core boundaries to formalize:
* data: load/resample/quality/features
* modeling: train/evaluate/predict + model registry operations
* strategy: parameter schema + decision logic
* backtest: run + metrics + artifact creation
* walkforward: windowing + robustness evaluation
## 2) Jobs (filesystem-backed job runner)
All long-running work runs outside Streamlit.
Design:
* Each job gets a unique `job_id` and a run directory: `runs/<job_id>/`
* Files in the run dir become the source of truth:
    * `request.json`: full input payload
    * `status.json`: {state: QUEUED|RUNNING|SUCCEEDED|FAILED, timestamps, error}
    * `result.json`: small summary (metrics, key artifact paths)
    * artifacts: CSVs/PNGs/models/metrics, etc.
Execution model (simple first):
* Streamlit starts jobs via `subprocess` (`python -m src.jobs.run --job-id ... --type ... --request ...`).
* The job process writes status/artifacts; UI polls by reading `status.json`.
* No FastAPI required to get the benefits. (FastAPI can be added later if/when multi-user becomes real.)
## 3) Live daemon (always-on engine)
Live trading is treated as a separate product/process.
Direction:
* Keep (and strengthen) `src.ibkr_live_session` as the daemon entrypoint.
* Ensure the daemon’s inputs/outputs are stable contracts:
    * inputs: config JSON + command line flags
    * outputs: append-only JSONL event log + kill switch file + optional periodic snapshots
* Streamlit becomes a viewer/controller only (no trading logic).
## 4) Thin UI (Streamlit multipage or page modules)
Streamlit becomes a thin client of:
* the job system (train/backtest/optimize/walkforward/data tasks)
* the live log (read-only dashboards + kill switch toggles)
Key rule:
* UI files may render and validate inputs, but must not own domain logic.
# Proposed directory layout
This layout minimizes disruption while creating clear seams:
* `src/ui/`
    * `app.py` (Streamlit entry)
    * `pages/` (one module per workflow)
* `src/core/`
    * `contracts.py` (dataclasses for inputs/outputs)
    * `data.py`, `modeling.py`, `backtesting.py`, `walkforward.py`, etc. (thin wrappers at first)
* `src/jobs/`
    * `types.py` (JobType enums + request/result schemas)
    * `store.py` (create run dir, read/write status/result)
    * `run.py` (CLI entrypoint to execute a job)
    * `handlers/` (train/backtest/optimize/predict_csv/etc.)
* `src/live/`
    * `engine.py` (thin wrapper around current live session runner or a moved implementation)
    * keep `src/ibkr_live_session.py` as a compatibility shim calling `src.live.engine`
* `runs/` (generated)
* `configs/`
    * `active.json` (user-chosen defaults for strategy + run params)
# Critical design decisions (to prevent regressions)
## Stop editing `src/config.py` from the UI
Instead:
* Treat `src/config.py` as code defaults + paths.
* Introduce `configs/active.json` as the mutable “user configuration”.
* Add a small resolver function (in core) that merges:
    * defaults from `src/config.py`
    * overrides from `configs/active.json`
    * per-run overrides from job requests
This change removes a major class of “invisible state changes” and makes runs reproducible.
## Standardize artifacts and schemas
Define explicit schemas for:
* predictions CSV columns
* backtest outputs (equity/trades/metrics)
* live log event types (bar, decision, order_status, fill, error, heartbeat, snapshot)
# Execution plan (phased, keeps app usable)
## Phase 0: Baseline + contracts (stability before refactor)
Deliverables:
* Add/strengthen a small set of regression tests that lock in today’s behavior for:
    * backtest metrics on a small deterministic dataset
    * predictions CSV alignment behavior (Time-join vs positional)
    * config resolution behavior (current defaults)
* Add `src/core/contracts.py` with dataclasses for:
    * `TrainRequest`, `TrainResult`
    * `BacktestRequest`, `BacktestResult`
    * `OptimizeRequest`, `OptimizeResult`
    * `WalkForwardRequest`, `WalkForwardResult`
Acceptance criteria:
* Tests pass and provide a safety net for the upcoming moves.
## Phase 1: Decompose UI without behavior change
Deliverables:
* Split `src/app.py` into page modules (still same UX):
    * `src/ui/pages/live.py`
    * `src/ui/pages/data.py`
    * `src/ui/pages/experiments.py`
    * `src/ui/pages/train.py`
    * `src/ui/pages/backtest.py`
    * `src/ui/pages/walkforward.py`
* Keep `src/app.py` as a compatibility entrypoint that imports and runs `src/ui/app.py`.
Acceptance criteria:
* Streamlit UI behaves the same, but code is navigable and bounded by files.
## Phase 2: Introduce filesystem job runner for long tasks
Deliverables:
* Add `src/jobs/*` and implement first job types:
    * TrainJob (wraps `src.train.train_model`)
    * BacktestJob (wraps `src.backtest.run_backtest_for_ui` or a new core wrapper)
    * OptimizeJob (wraps current grid-search loop)
    * PredictCsvJob (wraps existing predictions generation)
* Implement job status writing and crash-safe error capture.
Acceptance criteria:
* Jobs can be launched from CLI and produce `runs/<job_id>/...` artifacts.
* A job run is reproducible from `request.json`.
## Phase 3: Switch Streamlit to “start job + poll + render artifacts”
Deliverables:
* Update the Train/Backtest/Optimize/WalkForward/Data pages:
    * Start job via subprocess
    * Show live status
    * Render results from `runs/<job_id>/` artifacts
* Remove long-running loops from the Streamlit process.
Acceptance criteria:
* Streamlit reruns do not interrupt training/optimization.
* UI can recover after refresh and still show latest job results.
## Phase 4: Harden the always-on live daemon boundary
Deliverables:
* Treat `src.ibkr_live_session` as the daemon API boundary:
    * stabilize event schema and document it
    * ensure kill switch + logging are robust
* (Optional but high ROI) add a lightweight “supervisor” CLI:
    * start/stop/status helpers (still single-user, local)
Acceptance criteria:
* Live trading runs independently of Streamlit.
* UI is purely observational/control-plane (kill switch, notes), not execution.
## Phase 5: Gradual core cleanup (pay down coupling)
Deliverables:
* Move remaining “logic hiding in UI” into core functions.
* Replace ad-hoc JSON histories with job-run artifacts (history becomes “list runs”).
* Add contract tests per job type:
    * validate required artifact files exist and contain expected columns/fields
Acceptance criteria:
* Adding a feature typically touches one bounded area (core + one UI page) instead of `src/app.py`.
* Regressions are caught by job/contract tests.
# Risks and mitigations
* Risk: large refactor stalls feature development.
  Mitigation: phase-by-phase approach; keep `src/app.py` working via compatibility shims.
* Risk: too much new infrastructure.
  Mitigation: filesystem job runner first (no FastAPI/Celery); keep abstractions minimal.
* Risk: hidden config state continues to leak.
  Mitigation: stop UI edits to `config.py`; switch to `configs/active.json` and explicit merges.
# Follow-up options (only if needed later)
* Add FastAPI endpoints around the filesystem job store (no change to jobs themselves).
* Multi-user/auth/remote execution can be layered on once the single-user architecture is stable.
