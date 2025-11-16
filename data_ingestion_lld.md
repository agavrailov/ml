# Data Ingestion & Feature Engineering Low-Level Design (LLD)

## 1. Scope & Goals

This LLD specifies concrete modules, classes, functions, and configurations for the refactored data ingestion and feature engineering pipeline, based on the requirements and HLD. It focuses on:
- Interactive Brokers TWS/IB Gateway ingestion.
- Raw and curated storage interfaces.
- Processing (cleaning, gap-filling, data quality).
- Time alignment and feature computation.
- Training/batch scoring data access.

The LLD is language-agnostic in naming but assumes an implementation in a typical Python-based ML stack.

---

## 2. Module & Package Structure

Proposed high-level package layout under `src/`:

- `src/config/`
  - `__init__.py`
  - `settings.py` — central config loading and validation.

- `src/data_sources/`
  - `__init__.py`
  - `tws_client.py` — TWS API wrapper and connection management.
  - `tws_ingestor.py` — historical and live ingestion logic.
  - `file_ingestor.py` — CSV/Parquet file ingestion.
  - `db_ingestor.py` — DB-based ingestion (optional/extendable).

- `src/storage/`
  - `__init__.py`
  - `raw_store.py` — append-only raw storage API.
  - `curated_store.py` — curated tables API.
  - `feature_store.py` — feature storage and lookup.

- `src/pipelines/`
  - `__init__.py`
  - `ingestion_jobs.py` — orchestration entrypoints (CLI/cron).
  - `transform_pipeline.py` — raw → curated transforms.
  - `alignment_pipeline.py` — time alignment and resampling.
  - `feature_pipeline.py` — feature computation flows.

- `src/data_quality/`
  - `__init__.py`
  - `rules.py` — rule definitions and configuration.
  - `validator.py` — execution engine for rules.
  - `metrics.py` — computation and export of DQ metrics.

- `src/training/`
  - `__init__.py`
  - `dataset_builder.py` — training dataset generation.

- `src/scoring/`
  - `__init__.py`
  - `batch_scoring.py` — periodic batch scoring job.

- `src/ops/`
  - `__init__.py`
  - `orchestrator_adapter.py` — integration with scheduler/orchestrator.
  - `logging_setup.py` — logging configuration.

This structure can be adapted to current project layout, but the logical separation should be preserved.

---

## 3. Detailed Component Design

### 3.1 Configuration Layer (`src/config/settings.py`)

**Responsibility:** Centralize configuration for sources, storage, pipelines, and DQ rules.

**Key Concepts:**
- `SourceConfig` (e.g., TWS, files).
- `StorageConfig` (paths, DB URIs).
- `PipelineConfig` (frequencies, entities, windows).

**Example objects:**
- `TwsSourceConfig`:
  - `host: str`
  - `port: int`
  - `client_id: int`
  - `historical_bar_size: str` (e.g., `1 min`)
  - `historical_duration: str` (e.g., `1 D`)
  - `rate_limit_config: RateLimitConfig`

- `TimeSeriesConfig`:
  - `canonical_frequency: str` (e.g., `1min`)
  - `timezone: str`
  - `trading_calendar: str` (identifier for calendar rules)

Configuration is loaded from YAML/JSON/env and validated at startup.

---

### 3.2 TWS Client & Ingestor (`src/data_sources/tws_client.py`, `tws_ingestor.py`)

#### 3.2.1 `TwsClient`

**Responsibility:** Thin wrapper over the IB API implementing:
- Connection lifecycle.
- Request throttling according to IB pacing rules.
- Callbacks normalized into an internal event stream.

**Key methods (pseudo-signatures):**
- `connect() -> None`
- `disconnect() -> None`
- `is_connected() -> bool`
- `request_historical_bars(contract, start, end, bar_size, what_to_show, use_rth) -> Iterable[BarEvent]`
- `subscribe_market_data(contract) -> SubscriptionHandle`
- `unsubscribe_market_data(handle: SubscriptionHandle) -> None`

**Data structures:**
- `BarEvent`:
  - `ts: datetime`
  - `open: float`
  - `high: float`
  - `low: float`
  - `close: float`
  - `volume: float`
  - `contract_id: str`
  - `source: Literal["TWS"]`

#### 3.2.2 `TwsIngestor`

**Responsibility:** Use `TwsClient` + config to ingest historical and live bars and hand them to raw storage.

**Key methods:**
- `ingest_historical_range(instruments: list[Contract], start: datetime, end: datetime) -> None`
  - For each instrument, page through historical bars and send to `RawStore.write_events()`.
- `run_live_stream(instruments: list[Contract]) -> None`
  - Subscribe to live data; in a loop, batch events and send to `RawStore`.

**Error handling:**
- Retry on transient connection errors with exponential backoff.
- Respect rate limits by queueing requests and delaying when necessary.

---

### 3.3 Storage Layer (`src/storage/`)

#### 3.3.1 `RawStore` (`raw_store.py`)

**Responsibility:** Append-only writes of raw events, with partitioning and metadata.

**Core interface:**
- `write_tws_bars(events: Iterable[BarEvent], batch_id: str, ingestion_ts: datetime) -> None`
- `read_tws_bars(symbols: list[str], start: datetime, end: datetime) -> DataFrame`

**Implementation notes:**
- Enforce primary keys `(source, symbol, ts, batch_id)` to avoid duplicates.
- Partition by `date(ts)` and `symbol`.
- Store additional metadata columns from the LLD schema.

#### 3.3.2 `CuratedStore` (`curated_store.py`)

**Responsibility:** Store cleaned, gap-aware, resampled time-series.

**Core interface:**
- `write_curated_bars(df: DataFrame, dataset_name: str, partition_key: PartitionKey) -> None`
- `read_curated_bars(dataset_name: str, symbols: list[str], start: datetime, end: datetime) -> DataFrame`

Curated tables reflect the schema defined in the Data Model doc.

#### 3.3.3 `FeatureStore` (`feature_store.py`)

**Responsibility:** Store and retrieve computed features.

**Core interface:**
- `write_features(df: DataFrame, feature_group: str, version: str, partition_key: PartitionKey) -> None`
- `get_features_for_training(feature_group: str, version: str, start: datetime, end: datetime, symbols: list[str]) -> DataFrame`
- `get_features_for_inference(feature_group: str, version: str, ts: datetime, symbols: list[str]) -> DataFrame`

---

### 3.4 Transformation Pipelines (`src/pipelines/transform_pipeline.py`)

**Responsibility:** Raw → curated transformation including normalization, DQ checks, and gap-filling.

**Entry function (per dataset):**
- `run_transform_tws_bars(date: date, symbols: list[str]) -> None`

**Steps:**
1. Load raw data from `RawStore.read_tws_bars()` for `[date, date+1)`.
2. Apply schema normalization (enforce dtypes, rename columns).
3. Filter invalid records (bad timestamps, invalid prices/volumes).
4. Invoke `DataQualityValidator` with configured rules.
5. Invoke `GapEngine` to:
   - Construct expected timeline at canonical frequency.
   - Detect gaps.
   - Apply imputation strategy.
6. Write curated data to `CuratedStore`.
7. Write DQ metrics to monitoring/metrics sink.

---

### 3.5 Data Quality Engine (`src/data_quality/`)

#### 3.5.1 Rules (`rules.py`)

Define rules as data structures, e.g.:
- `RangeRule(field="close", min=0, max=None, severity="error")`
- `EnumRule(field="bar_type", allowed=["TRADES"], severity="warn")`
- `MonotonicTimeRule(entity_key="symbol", time_field="ts", severity="error")`

Rules are loaded per dataset from a config file.

#### 3.5.2 Validator (`validator.py`)

**Interface:**
- `validate(df: DataFrame, rules: list[Rule]) -> ValidationResult`

**`ValidationResult` includes:**
- `failed_records: DataFrame`
- `metrics: Dict[str, Any]` (null rates, outlier counts, etc.)
- `severity: Literal["ok", "warn", "error"]`

Actions based on severity (configurable):
- `ok` — continue.
- `warn` — log and continue.
- `error` — fail pipeline or quarantine records.

#### 3.5.3 Metrics (`metrics.py`)

- Functions to compute and emit metrics to monitoring system.

---

### 3.6 Gap Detection & Imputation (`GapEngine`)

**Location:** `src/pipelines/transform_pipeline.py` or `src/data_quality/gaps.py`.

**Interface:**
- `fill_gaps(df: DataFrame, cfg: GapConfig) -> Tuple[DataFrame, GapMask]`

**`GapConfig`:**
- `frequency: str` (e.g., `1min`).
- `entity_key: str` (e.g., `symbol`).
- `time_field: str` (e.g., `ts`).
- `imputation_strategies: Dict[str, ImputationStrategy]` per column.

**Output:**
- `df_filled` — time-indexed with all expected timestamps.
- `gap_mask` — boolean columns marking imputed vs. real values.

---

### 3.7 Alignment & Feature Pipelines (`alignment_pipeline.py`, `feature_pipeline.py`)

#### 3.7.1 Time Alignment Pipeline

**Function:**
- `run_alignment(date: date, symbols: list[str]) -> None`

**Steps:**
1. Read curated datasets (e.g., bars, reference data) for the day.
2. Resample if needed to canonical frequency.
3. Join on `(symbol, ts)` using exact or nearest-neighbor rules.
4. Persist aligned view or pass directly to feature computation.

#### 3.7.2 Feature Pipeline

**Function:**
- `run_feature_pipeline(date: date, symbols: list[str], feature_group: str, version: str) -> None`

**Steps:**
1. Read aligned time-series for the date.
2. Apply feature functions (rolling windows, lags, etc.) based on configuration.
3. Attach metadata columns (feature_group, version).
4. Write to `FeatureStore`.

**Feature function examples:**
- `rolling_mean(df, window=20)`
- `rolling_volatility(df, window=20)`
- `lag(df, periods=[1, 5, 10])`

---

### 3.8 Training & Scoring (`training/dataset_builder.py`, `scoring/batch_scoring.py`)

#### 3.8.1 `DatasetBuilder`

**Key method:**
- `build_training_dataset(cfg: TrainingDatasetConfig) -> DatasetHandle`

`TrainingDatasetConfig` includes:
- `feature_groups: list[str]`
- `feature_versions: Dict[str, str]`
- `label_source: str`
- `start: datetime`
- `end: datetime`
- `symbols: list[str]`

Steps:
1. Query `FeatureStore` for requested groups/versions.
2. Join with labels.
3. Apply final filters/transformations.
4. Persist dataset and return a handle/path.

#### 3.8.2 `BatchScoring`

**Key method:**
- `run_batch_scoring(cfg: BatchScoringConfig) -> None`

Steps:
1. Ensure ingestion/transform/feature pipelines have run for scoring window.
2. Fetch features for scoring timestamp(s) from `FeatureStore`.
3. Call external model inference service or local model.
4. Store predictions with metadata.

---

## 4. Error Handling, Logging & Retry

- All pipeline entrypoints log:
  - Job parameters (dates, symbols, feature groups).
  - Source batch IDs and counts.
  - Data quality summary metrics.
- Transient failures (network, temporary storage issues) are retried with exponential backoff.
- Persistent data issues (schema mismatch, severe DQ failures) cause the job to fail fast with clear error messages.

---

## 5. Configuration & Extensibility

- New sources: implement `BaseIngestor` interface and register in config.
- New datasets: define schemas + DQ rules + gap configs, then instantiate standard pipeline templates.
- New features: add functions and update feature configuration; pipeline picks them up based on config.

This LLD should be kept in sync with changes to the codebase and used as the reference when implementing the refactor.