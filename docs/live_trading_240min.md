# Live Trading with 240min Timeframe

## Problem
On a 240min (4-hour) timeframe, the predictor needs **24 bars** for warmup (`warmup_extra=20` + `tsteps=5` - 1). Without pre-seeding, this means:
- **6 days** of waiting before the first prediction
- Every restart = another 6-day wait

## Solution
Pre-seed the predictor buffer from historical CSV data on startup.

## Workflow

### One-time setup (or after long gaps)
Update your historical data to include recent bars:

```bash
python -m src.update_live_data --frequency 240min
```

This will:
1. Fetch new minute data from IBKR (incremental)
2. Clean and deduplicate
3. Resample to 240min
4. Save to `data/processed/nvda_240min.csv`

### Start live session
```bash
python -m src.ibkr_live_session --symbol NVDA --frequency 240min
```

On startup, the predictor will:
1. Load last 25 bars from `data/processed/nvda_240min.csv`
2. Subscribe to live bars from IBKR
3. Make predictions **immediately** when the first new live bar completes

## How it works

### Before (cold start)
```
Bar 1:  No prediction (warmup 1/24)
Bar 2:  No prediction (warmup 2/24)
...
Bar 24: No prediction (warmup 24/24)
Bar 25: First prediction ✓  [~6 days elapsed]
```

### After (warm start)
```
Startup: Load 25 historical bars from CSV
Bar 1:   First prediction ✓  [~4 hours elapsed]
```

## Implementation details

### LivePredictor.warmup_from_csv()
New method in `src/live_predictor.py` that pre-populates the internal buffer with historical bars:

```python
predictor = LivePredictor.from_config(...)
predictor.warmup_from_csv("data/processed/nvda_240min.csv", "240min")
```

### Data consistency
The historical CSV comes from the same data pipeline used for training:
- `RAW_DATA_CSV` (minute data) → clean → resample → `nvda_240min.csv`
- Predictor sees the exact same bars it would if it waited for live updates

### Restart behavior
After a restart:
1. Run `update_live_data` to fetch any missing bars since last run
2. Start live session → warmup loads most recent 25 bars
3. Live bars continue from where you left off

## Frequency-specific warmup times

| Frequency | Bars needed | Time (cold) | Time (warm) |
|-----------|-------------|-------------|-------------|
| 60min     | 24          | ~1 day      | ~1 hour     |
| 240min    | 24          | ~6 days     | ~4 hours    |
| 1440min   | 24          | ~24 days    | ~1 day      |

## Troubleshooting

### "Historical CSV not found"
Run `python -m src.update_live_data --frequency 240min` first.

### "Loaded 0 bars"
Your CSV might be empty or corrupted. Check:
```bash
head data/processed/nvda_240min.csv
```

Should have columns: `DateTime, Open, High, Low, Close`

### Still seeing warmup delay
Check the startup logs for:
```
[live] Predictor warmed up with 25 bars from data/processed/nvda_240min.csv
```

If this line is missing, warmup didn't run. Check file paths.

## Advanced: adjust warmup size

To reduce memory/startup time for very large buffers, adjust `warmup_extra` in `LivePredictorConfig`:

```python
# Default: 20 extra bars for SMA_21
config = LivePredictorConfig(frequency="240min", tsteps=5, warmup_extra=20)

# Aggressive: minimum needed for current features
config = LivePredictorConfig(frequency="240min", tsteps=5, warmup_extra=5)
```

Trade-off: smaller `warmup_extra` = less history for rolling features like SMA_21, which may reduce initial prediction quality.
