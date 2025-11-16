# Data Ingestion & Feature Pipeline Operational Runbook

## 1. Purpose

This runbook describes how to operate the refactored data ingestion and feature pipeline in day-to-day use, including:
- Routine operations (daily jobs, monitoring).
- Handling common issues (TWS outages, data quality failures).
- Running backfills.

---

## 2. Daily Operations

### 2.1 Scheduled Jobs

Typical daily schedule (example):

1. **TWS Historical/Daily Ingestion**
   - Runs after market close or at a configured time.
   - Populates `raw_tws_bars` for the last trading day.

2. **Transform & Curate**
   - Runs after ingestion completes.
   - Produces `curated_tws_1m_bars` and DQ metrics.

3. **Feature Computation**
   - Runs after curated data is ready.
   - Produces feature tables (e.g., `features_price_1m_v1`).

4. **Batch Scoring (if applicable)**
   - Uses latest features to produce predictions.

### 2.2 Daily Checklist

For each trading day:

1. **Verify job completion**
   - Use orchestration UI/CLI to ensure all scheduled jobs (ingestion, transform, features, scoring) completed successfully.

2. **Check data quality dashboards**
   - Confirm no critical DQ alerts (`RED` status) for yesterday’s data.

3. **Check data coverage**
   - Spot-check that key symbols have data for expected time ranges.

4. **Log anomalies**
   - Record any unusual issues (e.g., missing data in certain symbols, TWS outages).

---

## 3. Monitoring & Alerts

### 3.1 Metrics to Monitor

- **Pipeline metrics:**
  - Job success/failure counts.
  - Job duration and lag vs. schedule.
- **Data metrics:**
  - DQ metrics (null rates, gap counts, outlier counts).
  - Dataset health status (`GREEN/YELLOW/RED`).

### 3.2 Recommended Alerts

- **Ingestion job failures** (TWS or other sources).
- **Transform/feature job failures**.
- **High gap rate** in curated bars above a threshold.
- **High null rate** in key features.

Upon an alert, follow the relevant incident procedure below.

---

## 4. Common Operational Procedures

### 4.1 Handling TWS Connection Failures

**Symptoms:**
- Ingestion job fails with connection errors.
- Logs show repeated timeouts or disconnections from TWS.

**Steps:**
1. Check TWS/IB Gateway is running and logged in.
2. Verify network connectivity from pipeline host to TWS.
3. Review logs from `TwsClient` for specific IB error codes.
4. If the issue appears transient:
   - Re-run the ingestion job for the affected date and symbols.
5. If TWS is down for an extended period:
   - Document the outage times.
   - Plan a backfill once TWS is available again (see Backfills section).

### 4.2 Handling TWS Rate Limit / Pacing Errors

**Symptoms:**
- Logs contain IB pacing violation messages.
- Ingestion slows or fails.

**Steps:**
1. Confirm rate-limit configuration (max parallel requests, pacing windows).
2. Reduce parallelism or increase delays in ingestion config.
3. Re-run the job for affected ranges after adjusting configuration.

### 4.3 Data Quality Failures

**Symptoms:**
- DQ metrics indicate `RED` status for a dataset.
- DQ rules report excessive nulls/outliers/gaps.

**Steps:**
1. Use `get_data_quality_summary` or dashboards to identify the cause (which metric, which symbols/dates).
2. Inspect raw data for the affected symbols/dates.
3. Decide action:
   - If issue is due to upstream data (e.g., exchange outage):
     - Accept and annotate; potentially skip training for that day or adjust models.
   - If issue is due to pipeline bug or misconfig:
     - Fix code/config.
     - Re-run transform and feature pipelines for affected ranges (see Backfills).
4. Document incident and resolution.

### 4.4 Missing or Incomplete Data for a Day

**Symptoms:**
- Curated dataset has large gaps or is missing entirely for a day.

**Steps:**
1. Check ingestion logs for that day.
2. If ingestion failed:
   - Re-run ingestion job for the day.
3. If ingestion succeeded but curated data is missing:
   - Re-run transform and feature pipelines for that day.
4. If upstream data is genuinely missing:
   - Decide whether to impute more aggressively or mark the day as unusable.

---

## 5. Backfills & Reprocessing

### 5.1 When to Backfill

Typical triggers:
- TWS or upstream outages leading to missing data.
- Discovery of a pipeline bug or DQ rule change.
- Change in feature definitions or label logic.

### 5.2 Backfill Procedure

1. **Define scope**
   - Determine the affected symbols and date range.

2. **Plan impact**
   - Estimate duration and resource usage.
   - Plan to run at off-peak times if backfill is large.

3. **Trigger backfill**
   - Use `trigger_backfill(dataset_name, symbols, start, end, options)`.
   - Typically set `overwrite_curated` and `overwrite_features` to `true` for the affected partitions.

4. **Monitor backfill**
   - Use `get_backfill_job_status` and standard pipeline monitoring.
   - Check DQ metrics post-backfill.

5. **Post-backfill actions**
   - For critical changes, retrain models or re-run scoring for affected periods.

### 5.3 Safety Guidelines

- Never backfill the entire history without testing on a smaller window first.
- Keep a backup of previous curated/feature data (via snapshots or versioning) before large backfills.

---

## 6. Training & Scoring Operations

### 6.1 Training

**Routine training cycle:**
1. Ensure data and features are complete and healthy for the training window (DQ checks).
2. Use `get_features_for_training` to build the training dataset.
3. Train model and log feature versions and label definitions used.
4. Store model artifacts with references to dataset and pipeline versions.

### 6.2 Batch Scoring

**Routine scoring cycle:**
1. Ensure ingestion and feature pipelines for the scoring date/time have completed.
2. Use `get_features_for_inference` to retrieve features.
3. Run inference and store predictions with metadata (model version, feature versions).
4. Monitor predictions for anomalies if applicable.

---

## 7. Incident Management

### 7.1 Severity Levels

- **SEV-1 (Critical):**
  - No new data ingested for current/previous trading day.
  - Batch scoring cannot run.
- **SEV-2 (Major):**
  - Partial data loss for some symbols or features; workarounds exist but degrade performance.
- **SEV-3 (Minor):**
  - DQ warnings, minor delays, or cosmetic issues.

### 7.2 General Incident Response Steps

1. Identify scope and severity.
2. Stabilize system:
   - Stop non-critical jobs if they worsen the problem.
3. Diagnose root cause (logs, metrics, recent changes).
4. Apply fix or workaround (e.g., manual backfill, config change).
5. Validate recovery (DQ metrics, end-to-end tests on sample period).
6. Document incident details and lessons learned.

---

## 8. Change Management

- For changes to ingestion logic, DQ rules, or feature definitions:
  - Update design docs (LLD, data model, API contracts) if necessary.
  - Add/adjust tests (especially regression tests).
  - Roll out changes in lower environment or on a small data window first.
  - Monitor closely after deployment for unexpected behavior.

---

## 9. Quick Reference

- **Backfill command:** `trigger_backfill(dataset_name, symbols, start, end, options)`.
- **Check DQ:** `get_data_quality_summary(dataset_name, start_date, end_date)`.
- **Check dataset health:** `get_dataset_health(dataset_name, as_of_date)`.
- **Get curated bars:** `get_curated_bars(symbols, start, end, frequency)`.
- **Get training features:** `get_features_for_training(...)`.
- **Get inference features:** `get_features_for_inference(...)`.

Update this runbook as you learn from real incidents and refine the pipeline.