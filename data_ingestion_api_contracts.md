# Data Ingestion & Feature Platform API / Interface Contracts

## 1. Scope

This document defines the key internal APIs and interfaces for:
- Controlling ingestion and backfills.
- Retrieving curated data and features for training and inference.
- Querying data quality status.

It is implementation-agnostic (can be Python functions, services, or CLI wrappers) but should be treated as the contract between the pipeline and its consumers.

---

## 2. Ingestion Control APIs

### 2.1 Trigger Historical Ingestion

**Name:** `trigger_historical_ingestion`

**Purpose:** Ingest historical data from TWS (or other sources) for a given time range and instrument set.

**Signature (logical):**
```text
trigger_historical_ingestion(
  source: str,
  symbols: list[str],
  start: datetime,
  end: datetime,
  options: HistoricalIngestionOptions
) -> IngestionJobHandle
```

**Options:**
- `bar_size: str` — e.g., `"1 min"`.
- `what_to_show: str` — e.g., `"TRADES"`.
- `use_rth: bool`.
- `max_parallel_requests: int`.

**Behavior:**
- Schedules or immediately starts an ingestion job.
- Returns a handle that can be used to query job status.

### 2.2 Trigger Daily/Periodic Ingestion

**Name:** `trigger_daily_ingestion`

**Purpose:** Run standard daily ingestion for configured sources/instruments.

**Signature:**
```text
trigger_daily_ingestion(
  as_of_date: date
) -> IngestionJobHandle
```

**Behavior:**
- Uses configuration to determine which sources, symbols, and time ranges to ingest.

### 2.3 Job Status Query

**Name:** `get_ingestion_job_status`

**Signature:**
```text
get_ingestion_job_status(job_id: str) -> IngestionJobStatus
```

**`IngestionJobStatus` fields:**
- `job_id: str`
- `state: str` — `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`.
- `start_ts: datetime`
- `end_ts: datetime | None`
- `source: str`
- `params: dict` — input parameter echo.
- `summary: dict` — counts, error summaries.

---

## 3. Curated Data Access APIs

### 3.1 Get Curated Bars

**Name:** `get_curated_bars`

**Purpose:** Retrieve cleaned, gap-aware bars for analysis or as input to custom features.

**Signature:**
```text
get_curated_bars(
  symbols: list[str],
  start: datetime,
  end: datetime,
  frequency: str = "1min"
) -> DataFrame
```

**Guarantees:**
- Returns data aligned to `frequency`.
- Includes gap indicators and `dq_status` fields.

---

## 4. Feature Access APIs

### 4.1 Get Features for Training

**Name:** `get_features_for_training`

**Purpose:** Provide a feature matrix for model training over a historical range.

**Signature:**
```text
get_features_for_training(
  feature_groups: list[str],
  feature_versions: dict[str, str],
  symbols: list[str],
  start: datetime,
  end: datetime,
  include_labels: bool = False,
  label_source: str | None = None,
  label_def_version: str | None = None
) -> TrainingDataset
```

**`TrainingDataset` fields (logical):**
- `features: DataFrame`
- `labels: DataFrame | None`
- `metadata: dict` — feature versions, label def version, extraction timestamp.

### 4.2 Get Features for Inference

**Name:** `get_features_for_inference`

**Purpose:** Provide features for a given timestamp (or latest available) and set of entities.

**Signature:**
```text
get_features_for_inference(
  feature_groups: list[str],
  feature_versions: dict[str, str],
  symbols: list[str],
  ts: datetime,
  mode: str = "exact"  # or "latest"
) -> InferenceFeatures
```

**`InferenceFeatures` fields:**
- `features: DataFrame` — rows keyed by `(symbol, ts)`.
- `metadata: dict` — feature versions, data freshness info.

---

## 5. Backfill & Reprocessing APIs

### 5.1 Trigger Backfill

**Name:** `trigger_backfill`

**Purpose:** Reprocess raw data for a specified period and dataset (raw→curated→features).

**Signature:**
```text
trigger_backfill(
  dataset_name: str,
  symbols: list[str],
  start: datetime,
  end: datetime,
  options: BackfillOptions
) -> BackfillJobHandle
```

**`BackfillOptions`:**
- `overwrite_curated: bool` — whether to overwrite existing curated partitions.
- `overwrite_features: bool` — whether to overwrite feature partitions.
- `max_parallel_jobs: int`.

**Behavior:**
- Reads raw data for the period.
- Reruns transformation, alignment, and feature pipelines.
- Writes results using overwrite semantics consistent with idempotent pipelines.

### 5.2 Backfill Job Status

**Name:** `get_backfill_job_status`

**Signature:**
```text
get_backfill_job_status(job_id: str) -> BackfillJobStatus
```

Fields are analogous to `IngestionJobStatus`.

---

## 6. Data Quality & Health APIs

### 6.1 Get Data Quality Summary

**Name:** `get_data_quality_summary`

**Purpose:** Provide a summarized view of data quality metrics for a dataset/time range.

**Signature:**
```text
get_data_quality_summary(
  dataset_name: str,
  start_date: date,
  end_date: date,
  symbols: list[str] | None = None
) -> DataQualitySummary
```

**`DataQualitySummary` fields:**
- `dataset_name: str`
- `date_range: (date, date)`
- `metrics: list[DataQualityMetric]`

**`DataQualityMetric` fields:**
- `metric_name: str` — e.g., `"null_rate_close"`, `"gap_count"`.
- `value: float`
- `status: str` — `"ok"`, `"warn"`, `"error"`.
- `thresholds: dict` — configured thresholds.

### 6.2 Get Dataset Health Status

**Name:** `get_dataset_health`

**Purpose:** Answer: “Is this dataset good enough to use right now?”

**Signature:**
```text
get_dataset_health(
  dataset_name: str,
  as_of_date: date
) -> DatasetHealthStatus
```

**`DatasetHealthStatus` fields:**
- `dataset_name: str`
- `as_of_date: date`
- `status: str` — `"GREEN"`, `"YELLOW"`, `"RED"`.
- `reasons: list[str]` — human-readable description.

---

## 7. Operational APIs

### 7.1 List Available Datasets & Versions

**Name:** `list_datasets`

**Signature:**
```text
list_datasets() -> list[DatasetInfo]
```

**`DatasetInfo` fields:**
- `name: str`
- `type: str` — `raw`, `curated`, `features`, `labels`.
- `available_date_range: (date, date)`
- `latest_pipeline_version: str`

### 7.2 List Feature Groups & Versions

**Name:** `list_feature_groups`

**Signature:**
```text
list_feature_groups() -> list[FeatureGroupInfo]
```

**`FeatureGroupInfo` fields:**
- `feature_group: str`
- `versions: list[str]`
- `description: str`

---

## 8. Usage Patterns

### 8.1 Training Workflow

1. Use `list_feature_groups` to discover available features.
2. Call `get_features_for_training` with chosen groups/versions and label configuration.
3. Use returned `TrainingDataset` to feed model training.

### 8.2 Batch Scoring Workflow

1. Ensure ingestion/processing jobs have run (via `get_ingestion_job_status`).
2. Call `get_features_for_inference` for scoring timestamp(s).
3. Pass features to model(s) and store predictions.

### 8.3 Backfill Workflow

1. Call `trigger_backfill` for affected dates/symbols.
2. Monitor with `get_backfill_job_status`.
3. Once complete, re-run training or scoring as needed.

These contracts should be kept stable; when changes are required, introduce new versions (e.g., `get_features_for_training_v2`) to avoid breaking existing consumers.