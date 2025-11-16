# Data Ingestion & Feature Pipeline Test Strategy

## 1. Purpose

This document defines the testing strategy for the refactored data ingestion and feature pipeline, including test types, coverage goals, and example test cases.

---

## 2. Testing Objectives

- Ensure correctness of ingestion, cleaning, gap-filling, alignment, and feature computation.
- Prevent regressions during refactors and extensions.
- Detect data-quality issues as part of CI where feasible.
- Validate performance and scalability for daily runs and backfills.

---

## 3. Test Types

### 3.1 Unit Tests

**Scope:** Individual functions/classes with no external dependencies or with dependencies mocked.

**Targets:**
- TWS client helpers (e.g., bar parsing, rate-limit logic).
- Transformation functions (e.g., type coercion, schema normalization).
- Data quality rule evaluation (range checks, monotonicity, enums).
- Gap-filling algorithms (forward fill, interpolation).
- Feature functions (rolling averages, lags, volatility).

**Examples:**
- `test_forward_fill_missing_values` — verify imputation matches spec for synthetic series.
- `test_rolling_volatility_matches_manual_computation`.
- `test_monotonic_time_rule_detects_disordered_timestamps`.

---

### 3.2 Integration Tests

**Scope:** Multiple components working together with realistic data, typically in-memory or in a test environment.

**Targets:**
- `TwsIngestor` + `RawStore` on a small sample of mocked TWS responses.
- Raw→curated pipeline for a 1-day window with realistic edge cases (gaps, outliers).
- Curated→feature pipeline for a few symbols.

**Examples:**
- `test_raw_to_curated_pipeline_end_to_end` — given synthetic raw bars with known gaps/outliers, verify curated output and DQ metrics.
- `test_feature_pipeline_computes_expected_price_features` — align curated bars and compute features; compare to expected values.

---

### 3.3 Regression / Golden Dataset Tests

**Scope:** Compare outputs of the new pipeline to previously validated outputs on fixed historical windows.

**Approach:**
- Maintain one or more **golden datasets**:
  - Raw inputs for a fixed period.
  - Expected curated outputs.
  - Expected feature outputs.
- Regression tests read golden inputs, run the pipeline, and compare outputs to expected results.

**Examples:**
- `test_regression_tws_jan_2024_week` — run new pipeline on a week of TWS data and diff against stored golden outputs.

Comparison criteria:
- Exact equality for structural fields (keys, DQ flags, gap masks).
- Approximate equality for numeric fields within tolerances (e.g., 1e-8).

---

### 3.4 Contract Tests (API-Level)

**Scope:** Ensure the APIs defined in the API contracts doc behave as specified.

**Targets:**
- `get_curated_bars` returns correct schema and respects frequency.
- `get_features_for_training` returns expected columns and metadata.
- `trigger_backfill` and `get_backfill_job_status` follow state transitions.

**Examples:**
- `test_get_curated_bars_schema_and_keys` — verify columns, dtypes, and primary keys.
- `test_get_features_for_inference_latest_mode` — confirm it returns most recent features up to `ts`.

---

### 3.5 Performance & Load Tests

**Scope:** Validate that pipelines can handle expected daily and backfill volumes within time budgets.

**Approach:**
- Use synthetic or sampled real data to simulate:
  - Daily ingestion for a high number of symbols.
  - Backfill for N days/weeks of data.

**Metrics:**
- Wall-clock time per job.
- Throughput (rows/sec, symbols/sec).
- Resource utilization (CPU, memory, I/O).

**Examples:**
- `load_test_daily_ingestion_100_symbols` — confirm completing within X minutes on baseline hardware.

---

### 3.6 Data Quality Monitoring Tests

**Scope:** Validate that DQ metrics are generated and thresholds enforced.

**Targets:**
- DQ rule engine correctly flags bad data and sets statuses.
- Alerts/triggers for DQ failures (if integrated with monitoring).

**Examples:**
- `test_dq_metrics_written_for_each_pipeline_run` — verify a record is written to `dq_metrics`.
- `test_dataset_health_status_changes_to_red_on_high_gap_rate`.

---

## 4. Test Data Strategy

### 4.1 Synthetic Data

Use synthetic datasets for unit and many integration tests:
- Precisely controlled data for gap-filling, feature function validation, and DQ rule testing.

### 4.2 Sampled Real Data

Use sampled, anonymized slices of real TWS data for:
- Integration and regression tests.
- Performance/load tests.

Ensure sensitive data considerations are respected if any.

---

## 5. Tooling & Execution

- Use an automated test runner (e.g., `pytest`) with a consistent folder structure (e.g., `tests/` mirroring `src/`).
- Use fixtures for:
  - Synthetic time-series.
  - Temporary storage locations.
  - Mock TWS clients.
- Configure CI to run:
  - Unit & integration tests on each commit/PR.
  - Regression tests on a schedule or when pipeline-related code changes.
  - Performance tests periodically or on-demand.

---

## 6. Coverage Goals

- **Unit tests:**
  - >80% function coverage for transformation, DQ, and feature modules.
- **Integration tests:**
  - At least one end-to-end test per major pipeline (ingestion, transform, feature).
- **Regression tests:**
  - At least one golden dataset covering multiple instruments and several days.

These goals can be adjusted as the system matures.

---

## 7. Example Test Matrix

| Component                  | Unit | Integration | Regression | Perf | DQ |
|----------------------------|------|------------|-----------|------|----|
| TWS ingestion              |  X   |     X      |     X     |  X   |    |
| Raw→curated transform      |  X   |     X      |     X     |      | X  |
| Gap engine                 |  X   |     X      |           |      | X  |
| Feature computation        |  X   |     X      |     X     |      |    |
| Feature store API          |  X   |     X      |           |      |    |
| Training dataset builder   |  X   |     X      |     X     |      |    |
| Batch scoring pipeline     |      |     X      |     X     |  X   |    |

This matrix should be periodically reviewed and expanded as new components/features are added.