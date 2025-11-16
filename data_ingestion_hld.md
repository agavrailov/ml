# Data Ingestion & Feature Engineering High-Level Design (HLD)

## 1. Overview

This HLD describes the architecture and main components of the data ingestion and feature engineering platform, based on `data_ingestion_requirements.md`. It focuses on:
- Ingesting data from Interactive Brokers TWS/IB Gateway and other sources.
- Storing raw and curated data with full lineage and versioning.
- Cleaning, gap-filling, and aligning time-series data to canonical timeframes.
- Computing and serving features for training (offline) and inference (batch/near-real-time).

The design assumes a time-series centric use case (e.g., market data for LSTM models) but is extensible to other domains.

---

## 2. Logical Architecture

At a high level, the system is structured into layers:

1. **Source Connectors Layer**
   - TWS/IB Gateway connector (historical and live data).
   - File/database/other API connectors.

2. **Ingestion & Raw Data Layer**
   - Orchestrated ingestion jobs (batch & streaming).
   - Raw, append-only storage for ingested data + metadata.

3. **Processing & Curated Data Layer**
   - Schema normalization and validation.
   - Data quality checks, cleaning, and gap-filling.
   - Curated time-series and reference tables.

4. **Time Alignment & Feature Layer**
   - Resampling and multi-source alignment to canonical timeframes.
   - Feature pipelines (rolling windows, lags, etc.).
   - Feature store / feature tables.

5. **Consumption Layer**
   - Training dataset generation.
   - Batch scoring workflows.
   - Ad-hoc analysis and backtesting.

6. **Platform Services** (cross-cutting)
   - Metadata & catalog (schemas, feature definitions, lineage).
   - Orchestration (job scheduling and dependency management).
   - Monitoring & observability.
   - Access control & security.

---

## 3. Component Design

### 3.1 Source Connectors

**Responsibility:** Connect to external data sources, extract data, and emit it in a normalized internal event format.

#### 3.1.1 TWS/IB Gateway Connector

- **Interfaces**
  - Uses IB API to:
    - Request **historical bars** (batch ingestion mode).
    - Subscribe to **live market data** (streaming mode).
- **Core responsibilities**
  - Manage connection lifecycle (connect, heartbeat, reconnect with backoff).
  - Enforce IB pacing/rate limits and handle API error codes.
  - Map IB contracts and fields into internal schema (e.g., symbol, exchange, currency, bar type).
  - Emit normalized bar/tick events:
    - Timestamps in canonical timezone.
    - OHLCV fields and other relevant attributes.
  - Tag each event with metadata:
    - Source (`TWS`), contract ID, request ID/batch ID.
    - Ingestion timestamp.
- **Modes**
  - **Historical mode**: paginates through historical bar requests for given instruments and time ranges, respecting rate limits.
  - **Live mode**: maintains subscriptions for configured instruments; forwards events to ingestion pipeline.

#### 3.1.2 File/DB/Other Connectors

- File connectors (CSV/Parquet from local/remote storage).
- Database connectors (e.g., time-series DB, OLTP snapshot tables).
- Other APIs as needed.

All connectors emit data into a common **ingestion event bus** or directly write into the raw storage layer using a standardized schema.

---

### 3.2 Ingestion & Raw Data Layer

**Responsibility:** Receive data from connectors, validate basic structure, and persist raw records with metadata.

#### 3.2.1 Ingestion Orchestrator

- Schedules and triggers ingestion jobs:
  - Periodic batch jobs (e.g., daily backfill, end-of-day pulls).
  - Event-driven jobs (e.g., when a new file arrives).
  - Manual runs for backfills.
- Tracks job status, parameters (time ranges, instruments, sources), and outcomes.

#### 3.2.2 Raw Storage

- **Data model**
  - Append-only tables/partitions per source (e.g., `raw_tws_bars`, `raw_files`, `raw_labels`).
  - Partitioned by date/time and possibly instrument/entity.
  - Includes ingestion metadata:
    - `source`, `batch_id`, `ingestion_ts`, `pipeline_version`.
- **Behavior**
  - Ingestion is idempotent:
    - Use primary keys like `(source, entity_id, timestamp, batch_id)` to avoid duplicates.
  - Raw data is never mutated in place; corrections arrive as new records with newer metadata.

#### 3.2.3 Schema Registry & Validation

- Maintains schemas per source, with versions.
- On ingestion:
  - Validates incoming structure (fields, types, required columns).
  - Detects schema drift and either fails the batch or routes to quarantine.
- Stores schema definitions in a metadata service/catalog.

---

### 3.3 Processing & Curated Data Layer

**Responsibility:** Transform raw data into cleaned, quality-checked, and gap-aware curated datasets suitable for time alignment and feature computation.

#### 3.3.1 Transformation Pipelines

- Defined as configurable DAGs of steps (per dataset/source):
  1. Read raw partitions for a time range.
  2. Normalize data types and units.
  3. Apply basic business rules (e.g., filter invalid trading sessions, drop non-trading days for TWS).
  4. Run data quality checks (see 3.3.2).
  5. Detect gaps (see 3.3.3).
  6. Apply cleaning and imputation rules.
  7. Write results to curated tables.

- Implemented as reusable pipeline components (e.g., functions/operators) that can be composed and versioned.

#### 3.3.2 Data Quality Service

- Executed as part of transformation pipelines.
- Checks include:
  - Type validation, value ranges, enum checks.
  - Time-series checks: monotonic timestamps, no impossible values.
  - Basic distribution checks for early anomaly detection.
- Outputs:
  - Per-batch quality metrics (null rates, outlier counts, etc.).
  - Flags or quarantines records/batches failing critical checks.
  - Stores quality metrics in a monitoring store for dashboards.

#### 3.3.3 Gap Detection & Imputation Engine

- Given a canonical frequency and an entity/time range:
  - Constructs the expected timeline.
  - Marks missing timestamps as gaps.
- Supports configurable strategies per field/dataset:
  - Forward fill, backward fill, interpolation, constant values, or no-fill.
- Stores both:
  - **Imputed values** in curated tables.
  - **Gap/quality masks** to indicate which values are real vs. imputed.

#### 3.3.4 Curated Storage

- Holds cleaned, normalized time-series and reference tables:
  - E.g., `curated_tws_1m_bars`, `curated_reference_symbols`, `curated_labels`.
- Partitioned similarly to raw data but with a stable schema for downstream consumers.
- Includes metadata columns for data/feature versioning and quality indicators.

---

### 3.4 Time Alignment & Feature Layer

**Responsibility:** Align multiple curated sources onto a canonical timeline per entity and compute features.

#### 3.4.1 Time Alignment Service

- Resamples individual curated datasets to canonical frequencies (e.g., 1-min bars):
  - Upsampling/downsamping rules defined per feature (e.g., last, sum, mean).
- Aligns multiple datasets:
  - On entity (e.g., symbol) and timestamp.
  - Using exact matches or nearest-neighbor join within a tolerance window.
- Handles late-arriving/corrected data:
  - Provides ability to recompute aligned views for a specified time range.

#### 3.4.2 Feature Pipelines

- Configured per **feature group** (e.g., price technicals, volume features, label aggregations).
- Each pipeline defines:
  - Inputs: aligned time-series fields (e.g., `close`, `volume`).
  - Transformations: rolling windows, lags, differences, volatility, category encodings, etc.
  - Output: feature columns, data types, and masks.
- Implemented as versioned code + configuration so that a “feature definition version” can be tied to models.

#### 3.4.3 Feature Store / Feature Tables

- Logical view:
  - Keyed by `(entity_id, timestamp, feature_group, feature_version)`.
  - Contains feature columns, quality indicators, and lineage metadata.
- Supports two main access patterns:
  - **Offline**: bulk export for training over historical ranges.
  - **Online/batch scoring**: lookup of most recent or exact-time features for a given entity.

---

### 3.5 Consumption Layer

**Responsibility:** Provide data products tailored to downstream use cases (training, scoring, analysis).

#### 3.5.1 Training Dataset Generator

- Given a configuration (sources, feature groups, label definitions, time range):
  1. Reads aligned features for entities/timestamps.
  2. Joins labels/targets.
  3. Applies final filters (e.g., drop rows with too many missing values).
  4. Exports datasets in training-ready formats (e.g., Parquet, NumPy, TFRecords).
- Records metadata:
  - Feature group versions.
  - Data time range and extraction timestamp.
  - Label definitions and their versions.

#### 3.5.2 Batch Scoring Jobs

- Workflow:
  1. Ingest latest raw data for the scoring window.
  2. Run transformation, quality, gap-filling, and alignment.
  3. Compute feature groups required by the model.
  4. Retrieve features for the scoring timestamp(s).
  5. Call model inference (external to this HLD) and store predictions.
- Supports re-running scoring for specific time windows when backfills/corrections arrive.

#### 3.5.3 Ad-hoc Analysis & Backtesting Interface

- Provides analysts with access to:
  - Raw data snapshots.
  - Curated, gap-filled time-series.
  - Feature tables for historical periods.
- May be implemented as:
  - Direct query access (SQL/Notebook).
  - A small service/API that materializes datasets into temporary locations.
- Supports “sandbox” execution of alternative cleaning/feature configurations without affecting production tables.

---

### 3.6 Platform Services (Cross-Cutting)

#### 3.6.1 Metadata & Catalog Service

- Stores:
  - Source registrations (TWS, files, DBs, etc.).
  - Schema definitions and versions.
  - Feature definitions and versions.
  - Lineage links between raw → curated → features → training datasets → models.
- Provides APIs/CLI for discovery and governance.

#### 3.6.2 Orchestration Engine

- Manages DAGs for:
  - Ingestion.
  - Transformation/quality/gap-filling.
  - Time alignment and feature computation.
  - Training dataset generation and batch scoring.
- Supports:
  - Scheduling (cron-like).
  - Parameterization (e.g., date ranges, symbols).
  - Retries and failure notifications.

#### 3.6.3 Monitoring & Observability

- Collects metrics:
  - Rows read/written, error counts, processing time per stage.
  - Data quality metrics (null rates, outliers, gap counts).
- Provides dashboards and alarms for:
  - Pipeline health (success/failure, lag).
  - Data anomalies (e.g., sudden drop in volume, missing days).

#### 3.6.4 Security & Access Control

- Enforces role-based access:
  - Restricted write access to ingestion configs, transformations, and feature definitions.
  - Tiered read access for raw vs. curated vs. feature data.
- Integrates with organizational identity and secret management for external connections (e.g., TWS credentials).

---

## 4. Data Flow Diagrams (Logical)

### 4.1 End-to-End Batch Flow (Historical / Daily)

1. **Trigger**: Scheduler starts a daily job for a given instrument universe.
2. **TWS Connector** queries historical bars for `[t_start, t_end]` and emits events.
3. **Ingestion Layer** writes events into `raw_tws_bars` with metadata.
4. **Processing Pipeline** reads `raw_tws_bars` for `[t_start, t_end]`:
   - Normalizes schema.
   - Runs quality checks.
   - Detects gaps and applies imputation.
   - Writes to `curated_tws_1m_bars`.
5. **Alignment Service** resamples/aligns curated data to canonical timeline (if necessary).
6. **Feature Pipelines** compute desired feature groups and write to feature tables.
7. **Training Generator / Batch Scoring** reads feature tables and produces training datasets or predictions.

### 4.2 Live/Streaming Flow (Optional)

1. **TWS Connector** subscribes to live market data for configured instruments.
2. Incoming ticks/bars are normalized and streamed to the ingestion layer.
3. A **micro-batch pipeline** periodically (e.g., every minute) aggregates ticks into bars, runs minimal cleaning, and updates curated/live feature tables.
4. Downstream online inference consumes features from the feature store.

---

## 5. Versioning, Lineage & Reproducibility

- **Schemas**: versioned per source; changes recorded in metadata service.
- **Pipelines**: each transformation and feature pipeline has a code/config version.
- **Data versions**:
  - Raw data: append-only with batch IDs and ingestion timestamps.
  - Curated/feature tables: carry pipeline version and processing timestamp.
- **Lineage**:
  - For a given model, store the IDs of training datasets, feature group versions, and source schema versions used.
  - Enable reconstructing training data at a later date for audit/backtesting.

---

## 6. Non-Functional Considerations

### 6.1 Scalability

- Design pipelines to be horizontally scalable over partitions:
  - Partition by time and entity for both raw and curated tables.
  - Prefer distributed processing frameworks or parallelization where needed.

### 6.2 Reliability & Recovery

- All jobs must be retryable for specific date ranges and sources without manual intervention.
- Raw data storage acts as the single source of truth to rebuild curated and feature layers.
- Use idempotent write patterns (e.g., overwrite partitions by date and source in curated/feature layers).

### 6.3 Security

- Store external API credentials (e.g., TWS) in a secret manager, not in code.
- Audit access to raw data, curated datasets, and feature tables.

---

## 7. Open Design Choices

The following are intentionally left open for technology selection and detailed design:

- Storage technologies for raw/curated data and feature store (e.g., filesystem vs. object store vs. database).
- Orchestration engine (e.g., Airflow-like vs. custom).
- Exact implementation of monitoring/metrics stack.
- Degree of real-time support (batch-only vs. hybrid batch + low-latency features).

These should be decided based on existing infrastructure, team expertise, and performance/latency requirements.