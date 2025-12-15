# Progress Tracking for Jobs

## Overview

The job system now includes progress tracking capabilities for long-running jobs like optimization and walk-forward analysis. Progress is stored in the job status and displayed via a shared progress bar component in the UI.

### Visual Example

When an optimization job is running, the UI displays:

```
┌─────────────────────────────────────────┐
│ ↻ Active Job: abc123def456             │
│ Log: runs/abc123def456/artifacts/run.log│
│                                         │
│ [RUNNING] Running                       │
│ Started: 2025-12-14T20:30:00Z          │
│                                         │
│ Run 45/100                              │
│ ████████████░░░░░░░░░░░░░░ 45%         │
└─────────────────────────────────────────┘
```

## Architecture

### Backend Components

1. **JobStatus Type** (`src/jobs/types.py`)
   - Added `progress` field (0.0 to 1.0)
   - Added `progress_message` field (optional descriptive text)

2. **Progress Update Function** (`src/jobs/store.py`)
   - `update_progress(job_id, progress, message)` - Updates job progress without changing state
   - Thread-safe: reads current status, updates progress fields, writes back
   - Only updates if job is in RUNNING state

### Job Handlers

Progress tracking is implemented in:

1. **Optimize Job** (`src/jobs/handlers/optimize_job.py`)
   - Tracks progress through grid search iterations
   - Updates every ~1% of total runs (max 100 updates)
   - Message format: `"Run {current}/{total}"`

2. **Walk-Forward Job** (`src/jobs/handlers/walkforward_job.py`)
   - Tracks progress through parameter sets and folds
   - Updates every ~2% of total runs (max 50 updates)
   - Message format: `"Param {param_idx}/{n_params}, Fold {fold_idx}/{n_folds}"`

### UI Components

1. **Progress Bar Component** (`src/ui/components.py`)
   - `render_progress_bar(st, progress, message)` - Renders styled progress bar
   - Uses existing CSS classes from the component system
   - Shows percentage and optional message

2. **Integration** (`src/ui/page_modules/backtest_page.py`)
   - Progress bar appears when job is RUNNING and has progress data
   - Automatically refreshes when user clicks "Refresh" button
   - Styled consistently with status badges and other UI elements

## Auto-Refresh During Job Execution

**The UI automatically refreshes every 2 seconds while a job is running.** This uses the `streamlit-autorefresh` package to poll job status and update the progress bar in real-time.

**How it works:**
1. Start a job (optimization or walk-forward)
2. The page automatically refreshes every 2 seconds
3. Progress bar updates with the latest progress
4. When the job completes, auto-refresh stops and results are displayed

If `streamlit-autorefresh` is not installed, the feature gracefully degrades (no auto-refresh, but manual page refresh still works).

## Usage Example

### In Job Handler

```python
from src.jobs import store

def run(job_id: str, request: SomeRequest) -> None:
    total_items = 100
    
    for i in range(total_items):
        # Do work
        process_item(i)
        
        # Update progress periodically
        if i % max(1, total_items // 100) == 0:
            progress = (i + 1) / total_items
            store.update_progress(job_id, progress, f"Processing {i+1}/{total_items}")
```

### In UI

```python
from src.ui import components

if job_status.state == "RUNNING" and job_status.progress is not None:
    components.render_progress_bar(
        st,
        progress=job_status.progress,
        message=job_status.progress_message,
    )
```

## Design Decisions

1. **Update Frequency**: Limited to prevent excessive I/O
   - Optimize: ~100 updates max
   - Walk-forward: ~50 updates max
   
2. **Atomic Updates**: Progress updates use read-modify-write pattern
   - Ensures other status fields are preserved
   - Only updates if job is still RUNNING

3. **Optional Fields**: Progress tracking is optional
   - Jobs without progress tracking continue to work
   - UI gracefully handles missing progress data

4. **Simple State Machine**: Progress doesn't affect job state
   - State transitions (QUEUED → RUNNING → SUCCEEDED/FAILED) unchanged
   - Progress is metadata within RUNNING state

## Future Enhancements

- Add progress tracking to backtest job (if multi-bar backtests become common)
- Add progress tracking to training job
- Consider adding ETA estimation based on progress rate
- Add progress persistence for job recovery after crashes
