# Data Ingestion Data Model & Schema Specification

## 1. Scope

This document defines the logical data model and schemas for:
- Raw ingestion tables.
- Curated/cleaned time-series tables.
- Feature tables.
- Supporting reference and label tables.

It is designed to support TWS market data and other time-series sources.

---

## 2. Conventions

- **Timestamps**: stored in UTC (`TIMESTAMP`), with timezone conversions handled at the edge.
- **Symbols/Entities**: represented by a canonical `symbol` string and/or `contract_id` from TWS.
- **Partitioning**: tables are partitioned by `date` (derived from timestamp) and optionally by `symbol`.
- **Primary Keys**: specified as logical keys; physical enforcement depends on storage engine.

---

## 3. Raw Data Schemas

### 3.1 `raw_tws_bars`

**Purpose:** Store raw OHLCV bars from TWS/IB Gateway (historical and live).

**Columns:**
- `source` (STRING, NOT NULL) — e.g., `"TWS"`.
- `symbol` (STRING, NOT NULL) — canonical symbol (e.g., `AAPL`).
- `contract_id` (STRING, NOT NULL) — IB contract identifier.
- `ts` (TIMESTAMP, NOT NULL) — bar end timestamp in UTC.
- `open` (DOUBLE, NULLABLE)
- `high` (DOUBLE, NULLABLE)
- `low` (DOUBLE, NULLABLE)
- `close` (DOUBLE, NULLABLE)
- `volume` (DOUBLE, NULLABLE)
- `bar_size` (STRING, NOT NULL) — e.g., `"1 min"`.
- `what_to_show` (STRING, NOT NULL) — e.g., `"TRADES"`.
- `use_rth` (BOOLEAN, NOT NULL) — regular trading hours flag.
- `raw_metadata` (STRING/JSON, NULLABLE) — optional JSON blob with IB-specific fields.

**Ingestion metadata:**
- `batch_id` (STRING, NOT NULL)
- `ingestion_ts` (TIMESTAMP, NOT NULL)
- `pipeline_version` (STRING, NOT NULL)

**Logical primary key:**
- `(source, contract_id, ts, batch_id)`

**Partitioning:**
- `date` = `DATE(ts)`

---

### 3.2 `raw_other_source_*`

Pattern for additional raw sources (files, DB, etc.). Each will have:
- Source-specific fields.
- `source`, `batch_id`, `ingestion_ts`, `pipeline_version` metadata.

Schemas should follow the same conventions as `raw_tws_bars` (timestamp in UTC, partition by date).

---

## 4. Curated Data Schemas

### 4.1 `curated_tws_1m_bars`

**Purpose:** Cleaned, gap-aware 1-minute bars derived from `raw_tws_bars`.

**Columns:**
- `symbol` (STRING, NOT NULL)
- `ts` (TIMESTAMP, NOT NULL) — canonical bar timestamp (UTC, aligned to frequency).
- `open` (DOUBLE, NULLABLE)
- `high` (DOUBLE, NULLABLE)
- `low` (DOUBLE, NULLABLE)
- `close` (DOUBLE, NULLABLE)
- `volume` (DOUBLE, NULLABLE)

**Gap and quality indicators:**
- `is_imputed_open` (BOOLEAN, NOT NULL, default FALSE)
- `is_imputed_high` (BOOLEAN, NOT NULL, default FALSE)
- `is_imputed_low` (BOOLEAN, NOT NULL, default FALSE)
- `is_imputed_close` (BOOLEAN, NOT NULL, default FALSE)
- `is_imputed_volume` (BOOLEAN, NOT NULL, default FALSE)
- `dq_status` (STRING, NOT NULL) — e.g., `"ok"`, `"warn"`, `"error"`.

**Metadata:**
- `processing_ts` (TIMESTAMP, NOT NULL)
- `transform_version` (STRING, NOT NULL)

**Logical primary key:**
- `(symbol, ts)`

**Partitioning:**
- `date` = `DATE(ts)`

---

### 4.2 `curated_reference_symbols`

**Purpose:** Reference table for symbols/contracts.

**Columns:**
- `symbol` (STRING, NOT NULL)
- `contract_id` (STRING, NOT NULL)
- `exchange` (STRING, NULLABLE)
- `currency` (STRING, NULLABLE)
- `asset_class` (STRING, NULLABLE) — e.g., `STK`, `FUT`, `FX`.
- `primary` (BOOLEAN, NOT NULL, default TRUE) — marks primary mapping.

**Metadata:**
- `effective_from` (DATE, NOT NULL)
- `effective_to` (DATE, NULLABLE)
- `source` (STRING, NOT NULL)

**Logical primary key:**
- `(symbol, contract_id, effective_from)`

---

### 4.3 Labels Table (example: `labels_tws_returns`)

**Purpose:** Store target variables such as future returns.

**Columns:**
- `symbol` (STRING, NOT NULL)
- `ts` (TIMESTAMP, NOT NULL) — reference time.
- `horizon` (STRING, NOT NULL) — e.g., `"1h"`, `"1d"`.
- `return` (DOUBLE, NULLABLE)

**Metadata:**
- `label_def_version` (STRING, NOT NULL)
- `generation_ts` (TIMESTAMP, NOT NULL)

**Logical primary key:**
- `(symbol, ts, horizon, label_def_version)`

---

## 5. Feature Schemas

Feature tables are organized by **feature group** and **version**.

### 5.1 `features_price_1m_v1`

**Purpose:** Price-based technical features at 1-minute frequency.

**Columns:**
- `symbol` (STRING, NOT NULL)
- `ts` (TIMESTAMP, NOT NULL)

**Feature columns (examples):**
- `ret_1` (DOUBLE) — 1-bar return.
- `ret_5` (DOUBLE) — 5-bar return.
- `ret_20` (DOUBLE) — 20-bar return.
- `ma_5` (DOUBLE) — 5-bar moving average of close.
- `ma_20` (DOUBLE) — 20-bar moving average.
- `vol_20` (DOUBLE) — 20-bar rolling volatility.

**Quality indicators:**
- `num_missing_in_window_5` (INT)
- `num_missing_in_window_20` (INT)

**Metadata:**
- `feature_group` (STRING, NOT NULL) — `"price_1m"`.
- `feature_version` (STRING, NOT NULL) — `"v1"`.
- `generation_ts` (TIMESTAMP, NOT NULL)

**Logical primary key:**
- `(symbol, ts, feature_group, feature_version)`

**Partitioning:**
- `date` = `DATE(ts)`

---

### 5.2 `features_volume_1m_v1`

**Purpose:** Volume/participation features.

**Columns:**
- `symbol` (STRING, NOT NULL)
- `ts` (TIMESTAMP, NOT NULL)
- `vol_ma_20` (DOUBLE)
- `vol_zscore_20` (DOUBLE)
- `vol_percentile_20` (DOUBLE)

**Metadata/keys:**
- Same pattern as `features_price_1m_v1`.

---

### 5.3 General Feature Table Pattern

For each feature group/version, follow:

- Table name: `features_<group>_<freq>_<version>`.
- Columns:
  - `symbol` (STRING, NOT NULL)
  - `ts` (TIMESTAMP, NOT NULL)
  - Feature columns (group-specific)
  - Quality/coverage indicators
  - `feature_group`, `feature_version`, `generation_ts`
- Logical key: `(symbol, ts, feature_group, feature_version)`

---

## 6. Data Quality Metrics Storage (Optional)

### 6.1 `dq_metrics`

**Purpose:** Store per-batch data quality metrics.

**Columns:**
- `dataset_name` (STRING, NOT NULL) — e.g., `"raw_tws_bars"`, `"curated_tws_1m_bars"`.
- `date` (DATE, NOT NULL)
- `symbol` (STRING, NULLABLE) — optional; null for global metrics.
- `metric_name` (STRING, NOT NULL)
- `metric_value` (DOUBLE, NOT NULL)
- `status` (STRING, NOT NULL) — `"ok"`, `"warn"`, `"error"`.
- `pipeline_version` (STRING, NOT NULL)
- `recorded_ts` (TIMESTAMP, NOT NULL)

**Logical key:**
- `(dataset_name, date, symbol, metric_name, pipeline_version)`

---

## 7. Lineage & Version Metadata (Logical)

In addition to the main tables, the system should maintain lineage information (can be stored in a separate metadata store or tables), such as:

- `model_training_runs`:
  - `run_id`, `model_id`, `training_dataset_path`, `feature_versions`, `label_def_version`, `run_ts`.

- `pipeline_versions`:
  - `pipeline_name`, `version`, `code_hash`, `config_hash`, `effective_from`, `effective_to`.

These are not strictly required for the core data pipeline but strongly recommended for reproducibility and governance.