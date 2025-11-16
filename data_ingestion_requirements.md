# Data Ingestion & Curated Bars Requirements (Repo-Scoped)

## 1. Purpose & Scope

This document defines **repo-specific** requirements for refactoring data ingestion in this project. The goal is to:

- Cleanly **separate TWS ingestion** from the rest of the ML code.
- Introduce a minimal but well-defined **raw → curated bars pipeline** backed by local files.
- Keep existing **training/serving code and `src/config.py`** working with minimal disruption.

This is a **narrow slice** of the broader platform design (see HLD/LLD). It focuses on historical/daily ingestion and basic curated minute bars for NVDA (and optionally a small set of symbols), not the full feature platform.

### In-Scope (v1 Refactor)

1. **Historical & daily ingestion from TWS into a raw store**
   - Pull OHLC bars from IB TWS/IB Gateway using `ib_insync`.
   - Append them to a **raw local storage layout** (files under `data/`), with simple but explicit metadata.

2. **Raw → curated bars transformation**
   - Clean and normalize raw bars for NVDA (and possibly a few other symbols) into **curated 1-minute bars**.
   - Perform basic **gap handling** and **deduplication** consistent with existing behavior.

3. **CLI-based entrypoints** for ingestion & transformation
   - Commands (Python CLIs) that can be used by local scripts / Task Scheduler / cron.

4. **Stable, minimal programmatic interfaces** used by this repo’s code
   - A small set of functions (Python APIs) that training/processing code can call to read raw/curated data.

5. **Interoperability with existing code**
   - Existing modules (`src/data_ingestion.py`, `src/daily_data_agent.py`, `src/data_processing.py`, `src/data_updater.py`, etc.) should either:
     - Call the new ingestion/curation functions, or
     - Continue working against the refactored implementation without breaking their public behavior.

### Out of Scope (for this refactor)

The following are covered in the HLD/LLD but **not required in v1**:

- Live/streaming ingestion from TWS.
- Generic multi-source connectors beyond the current TWS usage.
- General feature store, `get_features_for_training` / `get_features_for_inference` APIs.
- Labels/feature tables schemas and their storage.
- External orchestration systems (Airflow/Prefect). We use **CLI-only**.
- Switching storage to a DB/warehouse. We stick to **local files** (CSV/Parquet) in `data/`.
- Replacing `src/config.py`. New ingestion code must **reuse** existing config constants.


## 2. Functional Requirements

### 2.1 Raw Historical Ingestion

**FR-1: Historical ingestion function**  
The system must provide a function (Python API) that ingests historical bars from TWS into a raw store:

- Logical signature (Python-level):

  ```text
  trigger_historical_ingestion(
    symbols: list[str],        # e.g., ["NVDA"]
    start: datetime,
    end: datetime,
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
  ) -> None | JobHandle
  ```

- v1 can be implemented as a synchronous wrapper around an async implementation (like `fetch_historical_data`) and **may return `None`** instead of a full job object.

**FR-2: Raw storage layout**  
Historical bars must be persisted locally under `data/` using a consistent naming convention. Minimal requirement:

- For NVDA 1-minute bars, raw file path MUST be:
  - `data/raw/nvda_minute.csv` (existing convention), OR
  - A new path that is clearly documented and used consistently by all consumers.
- The raw file must contain at least:
  - `DateTime` (string, `%Y-%m-%d %H:%M:%S`, UTC or consistent timezone).
  - `Open`, `High`, `Low`, `Close` (floats).
- Existing consumers (e.g., `src/data_processing.py`, `daily_data_agent`, tests) must still be able to find the raw data using `src.config.RAW_DATA_CSV`.

**FR-3: Incremental ingestion**  
The ingestion implementation MUST:

- Detect the last ingested `DateTime` in the raw file and only fetch **new** bars beyond that, unless a `strict_range=True` flag is passed.
- Be safe to re-run for the same `(symbol, start, end)` without producing duplicate rows.

**FR-4: TWS connection & pacing**  
The ingestion code must:

- Connect to TWS using `TWS_HOST`, `TWS_PORT`, and `TWS_CLIENT_ID` from `src.config`.
- Enforce IB pacing constraints by limiting concurrent historical requests:
  - Apply a hard cap of 50 concurrent requests.
  - Use a lower effective concurrency for small bar sizes (≤ 30 seconds).
- Log key events (connect, qualify contract, request ranges, errors).

### 2.2 Raw → Curated Bars Transformation

**FR-5: Curated bars function**  
Provide a function that transforms raw minute bars into curated, cleaned bars aligned to a canonical timeframe:

- Logical signature:

  ```text
  run_transform_minute_bars(
    symbol: str,
    date: date | None = None,
  ) -> None
  ```

- v1 can operate on the entire raw file if per-day partitioning is not yet implemented, but should be designed so per-day processing is easy later.

**FR-6: Curated storage layout**  
Curated bars for NVDA must be written to a stable, documented path, for example:

- `data/curated/nvda_1min_curated.csv` or `data/processed/nvda_1min_curated.csv`.

Curated file must contain at least:

- `Time` or `DateTime` (canonical timestamp string),
- `Open`, `High`, `Low`, `Close`,
- Optional gap/imputation indicators and quality status (see FR-7).

Existing downstream code that currently reads `data/processed/nvda_hourly.csv` may remain unchanged; curated minute bars can be an intermediate used by `data_processing`.

**FR-7: Cleaning & gap handling**  
The transform must:

- Sort by timestamp and drop duplicate rows.
- Support **gap detection** consistent with one of:
  - Current `analyze_gaps.py` + `fill_gaps(...)` behavior, or
  - A documented subset of the gap-engine design in the HLD.
- For v1, it is acceptable to implement **only simple forward-fill gap filling** for gaps below a configurable threshold and leave longer gaps unfilled.

**FR-8: Idempotent transform**  
Re-running the transform for the same symbol/date(s) must either:

- Overwrite the existing curated file(s) for that period, or
- Produce identical output (when raw data unchanged), without duplicating or corrupting rows.

### 2.3 CLI Entry Points

**FR-9: Historical ingestion CLI**  
Provide a CLI entrypoint that can be invoked from the repo root, e.g.:

- `python -m src.data_ingestion --symbol NVDA --start 2024-01-01 --end 2024-02-01`

For v1, this can be a simple `if __name__ == "__main__"` block or a small `click`/`argparse`-based CLI. It should:

- Accept `symbol`, `start`, `end`, and optional `bar_size` parameters.
- Call `trigger_historical_ingestion` (or equivalent) under the hood.

_In this repo, this is implemented by the `__main__` handler in `src/data_ingestion.py`, which wraps `trigger_historical_ingestion(...)` via an `argparse` CLI._

**FR-10: Transform CLI**  
Provide a CLI entrypoint for transforming raw to curated bars, e.g.:

- `python -m src.ingestion.curated_minute --symbol NVDA`

This can be invoked directly or from existing scripts like `daily_data_agent`.

_In this repo, this is implemented by the `__main__` handler in `src/ingestion/curated_minute.py`, which calls `run_transform_minute_bars(...)` via an `argparse` CLI._

### 2.4 Programmatic Interfaces for Internal Use

**FR-11: Minimal access APIs**  
Implement at least the following internal Python APIs, backed by local files:

- `get_raw_bars(symbol: str, start: datetime, end: datetime) -> pandas.DataFrame`
- `get_curated_bars(symbol: str, start: datetime, end: datetime) -> pandas.DataFrame`

These functions should:

- Use the storage layout decided in FR-2 and FR-6.
- Return DataFrames with stable column names.
- Be safe for use by training/preprocessing code (`data_processing`, `daily_data_agent`, etc.).


## 3. Non-Functional Requirements

### 3.1 Performance & Reliability

**NFR-1: Daily ingestion latency**  
The ingestion pipeline should be able to ingest one trading day of 1-minute bars for a single symbol (NVDA) within a few minutes on a typical development machine.

**NFR-2: Retry behavior**  
In case of transient TWS or network errors, the ingestion code should:

- Retry connection and/or requests with a modest backoff.
- Surface clear error messages if the job ultimately fails.

**NFR-3: Logging**  
All CLIs should log to stdout/stderr:

- Start/finish times.
- Requested date ranges and symbols.
- Number of bars fetched/written.
- Any errors and their context.

### 3.2 Interoperability and Backwards Compatibility

**NFR-4: Preserve `src/config.py` as the source of truth**  
New code MUST:

- Use existing config values from `src.config`, especially:
  - `RAW_DATA_CSV`, `PROCESSED_DATA_DIR`, `TWS_HOST`, `TWS_PORT`, `TWS_CLIENT_ID`, `NVDA_CONTRACT_DETAILS`, `TWS_MAX_CONCURRENT_REQUESTS`, `DATA_BATCH_SAVE_SIZE`.
- Avoid introducing a parallel, conflicting config system for ingestion in v1.

**NFR-5: Compatibility with existing scripts**  
Existing scripts should continue to function after the refactor, either by:

- Delegating to new ingestion/transform functions internally, or
- Calling a thin shim layer that preserves their current CLI and behavior.

Priority is:

- `src/data_ingestion.py` (thin wrapper over the refactored ingestion logic and historical ingestion CLI).
- `src/daily_data_agent.py` (orchestrates ingestion, gap handling, curated-minute generation, and resampling/feature engineering).


## 4. Testing Requirements (Repo-Specific)

**TR-1: Unit tests for ingestion helpers**  
Add or adapt unit tests to cover:

- Timestamp parsing and `_get_latest_timestamp_from_csv`-like logic.
- Effective concurrency calculation based on `barSizeSetting` and `TWS_MAX_CONCURRENT_REQUESTS`.
- Date range partitioning into daily chunks.

**TR-2: Integration tests for historical ingestion**  
Extend/adjust `tests/test_data_ingestion.py` to:

- Use mocks for `IB`, `util.df`, and file writes.
- Assert that `trigger_historical_ingestion` (or its core implementation) calls IB with correct parameters and writes correctly formatted raw bars.

**TR-3: Integration tests for raw → curated transform**  
Add tests (new file or extend `test_data_processing.py`) that:

- Build a small synthetic raw CSV with gaps/duplicates.
- Run `run_transform_minute_bars` (or equivalent).
- Validate:
  - Sorted, de-duplicated rows.
  - Gap-filling behavior (for short gaps).
  - Output schema and basic correctness of OHLC aggregation (if any resampling is done).

**TR-4: Backwards-compat tests**  
Ensure existing tests that depend on raw CSV (`test_data_processing`, `test_data_ingestion`) still pass after refactor, or are updated to the new paths/semantics without losing coverage.


## 5. Open Questions (to be resolved during implementation)

1. **Exact curated file path and naming**  
   - Do we keep using `data/processed/nvda_hourly.csv` as the primary downstream input, with curated minute bars as an internal detail, or create a new explicit curated-minute file?

2. **Symbol generalization**  
   - v1 is NVDA-focused; do we want minimal support for additional symbols via parameters (`symbols: list[str]`) even if not immediately used?

3. **Metadata columns**  
   - Which subset of the richer raw/curated schemas (from `data_ingestion_data_model.md`) do we want to implement now (e.g., `batch_id`, `ingestion_ts`, `dq_status`)?

4. **CLI UX**  
   - How “polished” should the new CLIs be (full argument parsing, help, etc.) vs. simple scripts mainly for personal use?

These can be refined as implementation progresses; this document should be updated accordingly when decisions are made.