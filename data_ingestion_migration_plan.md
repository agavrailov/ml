# Data Ingestion Refactor Migration & Cutover Plan

## 1. Purpose

This document describes how to migrate from the current data ingestion pipeline to the refactored architecture defined in the requirements, HLD, and LLD, with minimal disruption and a clear rollback path.

---

## 2. Current vs. Target Overview

### 2.1 Current State (Logical)

- Ingestion logic and data processing are tightly coupled to the model code.
- Data storage may be ad-hoc (e.g., mixed formats, limited versioning, limited DQ/lineage).
- Backfills and manual scripts exist but are not standardized.

### 2.2 Target State

- Clear separation of:
  - Source connectors (TWS and others).
  - Raw storage, curated storage, and feature store.
  - Transformation, alignment, and feature pipelines.
  - Training and scoring consumption layers.
- Explicit schemas, DQ rules, and lineage.
- Standard APIs for training and scoring.

---

## 3. Migration Principles

1. **No data loss**: Raw data must remain available to rebuild curated/feature layers.
2. **Incremental migration**: Introduce new components in phases; avoid a big-bang switch.
3. **Dual running**: Where feasible, run old and new pipelines in parallel on the same data to compare outputs.
4. **Rollback first**: For each phase, define how to revert to the previous state.
5. **Observability**: Instrument the new pipeline early to detect regressions.

---

## 4. Phased Plan

### Phase 0 – Preparation

**Goals:**
- Freeze a snapshot of the current system state.
- Finalize design documents (requirements, HLD, LLD, schemas, APIs).

**Tasks:**
- Ensure snapshot branch exists with the pre-refactor state.
- Lock down and review `data_ingestion_requirements.md`, `data_ingestion_hld.md`, LLD, data model, and APIs.
- Identify existing scripts and entrypoints that perform ingestion, cleaning, and feature generation.

**Exit criteria:**
- Design docs approved.
- Snapshot of current code and data structure recorded.

---

### Phase 1 – Introduce New Data Model and Storage Layer

**Goals:**
- Implement raw and curated table structures without changing existing pipelines.

**Tasks:**
- Implement `RawStore`, `CuratedStore`, and `FeatureStore` wrappers, even if initially they map to current storage mechanisms.
- Create physical tables/paths for:
  - `raw_tws_bars`.
  - `curated_tws_1m_bars`.
  - Initial feature tables (e.g., `features_price_1m_v1`).
- Implement minimal migration scripts to map existing data into the new structure (if applicable).

**Exit criteria:**
- New tables/paths exist and are writable.
- No impact on existing pipeline execution.

**Rollback:**
- Drop or ignore new tables/paths.
- Remove or disable unused store wrappers.

---

### Phase 2 – Implement New Ingestion & Transformation Pipelines

**Goals:**
- Develop new TWS ingestion, raw→curated transform, and DQ/gap-filling pipelines.

**Tasks:**
- Implement `TwsClient` and `TwsIngestor`.
- Implement `run_transform_tws_bars` pipeline.
- Implement DQ rules and gap filling for TWS bars.
- Introduce configuration for canonical timeframe, symbols, and calendars.

**Execution:**
- Run new pipelines in **shadow mode** on a restricted date range (e.g., last 5–10 trading days):
  - Do not change downstream training/scoring yet.
  - Compare outputs to the current pipeline (same inputs).

**Exit criteria:**
- New pipelines complete successfully for test ranges.
- Differences vs. old pipeline are understood and acceptable.

**Rollback:**
- Disable new ingestion and transform jobs.
- Retain code but do not schedule or call it.

---

### Phase 3 – Dual-Run on Production Range

**Goals:**
- Run both old and new pipelines for a longer historical window and near real-time.

**Tasks:**
- For a defined period (e.g., 2–4 weeks of data):
  - Run old pipeline as-is.
  - Run new pipeline to produce `curated_tws_1m_bars` and features.
- Build comparison scripts to:
  - Compare core fields (close, volume, basic features) between old and new outputs.
  - Compute error distributions and difference metrics.

**Exit criteria:**
- Data and feature discrepancies are within acceptable tolerances.
- Performance of new pipeline is acceptable for expected loads.

**Rollback:**
- Continue using old pipeline exclusively.
- Keep new pipeline in shadow mode only.

---

### Phase 4 – Switch Training to New Pipeline

**Goals:**
- Use new pipeline outputs for model training, while keeping old pipeline available as fallback.

**Tasks:**
- Update training scripts to call `get_features_for_training` from the new feature store.
- Train models using features and labels from the new pipeline.
- Compare model performance (offline) vs. models trained with old pipeline.

**Exit criteria:**
- At least one full training run completed with new pipeline.
- Model performance is equal or better, or differences are explainable and acceptable.

**Rollback:**
- Revert training scripts to use old data sources.
- Keep new pipeline running only for evaluation.

---

### Phase 5 – Switch Batch Scoring to New Pipeline

**Goals:**
- Use new pipeline data and features for batch predictions.

**Tasks:**
- Update batch scoring job to fetch features using `get_features_for_inference`.
- For a trial period, run **dual scoring**:
  - Old pipeline + old features → predictions A.
  - New pipeline + new features → predictions B.
- Compare prediction distributions and downstream metrics (if applicable).

**Exit criteria:**
- New predictions behave as expected.
- Operational stability of new pipeline is confirmed.

**Rollback:**
- Switch scoring back to old pipeline outputs.

---

### Phase 6 – Decommission Old Pipeline

**Goals:**
- Remove old ingestion and processing code once confidence in new system is high.

**Tasks:**
- Gradually remove references to old pipeline in code, configs, and schedules.
- Archive scripts and configs of the old pipeline in version control.
- Clean up obsolete tables/paths (only after a safe retention period).

**Exit criteria:**
- All scheduled jobs and consumers use the new pipeline exclusively.
- Old codepaths are removed or clearly marked as deprecated.

**Rollback:**
- After decommission, rollback is limited; keep historical artifacts and the ability to reconstruct old behavior from Git and data snapshots.

---

## 5. Validation & Acceptance Criteria

For each phase, define concrete checks, e.g.:

- **Data-level checks:**
  - Coverage: no unexpected gaps compared to the old pipeline.
  - Value distributions: means/vols within tolerance.
  - DQ metrics: null rates, outlier counts in expected ranges.

- **Performance checks:**
  - Daily job completes within X minutes.
  - Backfill jobs complete within acceptable time for N days of data.

- **Operational checks:**
  - Alerts triggered on expected error conditions.
  - Runbook procedures validated (e.g., backfill, handling TWS outages).

---

## 6. Communication & Coordination

- Document rollout timeline and phases.
- Communicate changes to any stakeholders using the data (even if that is only you right now).
- Keep a simple changelog for each phase (what changed, when, why, issues observed).

---

## 7. Risks & Mitigations

**Risk:** TWS-specific quirks (e.g., rate limits, session changes) cause new pipeline instability.
- **Mitigation:** Emphasize robust TWS client implementation, shadow runs, and conservative backoff.

**Risk:** Hidden assumptions in the old pipeline (e.g., ad-hoc cleaning) are not captured in new design.
- **Mitigation:** Use comparison scripts and deep diffs to uncover discrepancies; adjust new rules or document behavioral changes.

**Risk:** Backfills on large history may be slow.
- **Mitigation:** Parallelize backfill jobs, use efficient storage formats, and plan backfills during low-usage windows.

This plan should be updated as you learn more from early phases and real runs of the new pipeline.