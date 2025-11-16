import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.ingestion.curated_minute import (
    run_transform_minute_bars,
    get_raw_bars,
    get_curated_bars,
)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Set up temporary raw and curated paths for curated-minute tests.

    We monkeypatch the module-level paths in ``src.ingestion.curated_minute`` so
    tests do not touch real project data under ``data/``.
    """
    from src import ingestion as ingestion_pkg
    from src.ingestion import curated_minute as cm

    raw_path = tmp_path / "data" / "raw" / "nvda_minute.csv"
    curated_dir = tmp_path / "data" / "processed"
    curated_path = curated_dir / "nvda_minute_curated.csv"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)

    # Patch module-level constants used by the helpers
    monkeypatch.setattr(cm, "RAW_DATA_CSV", str(raw_path))
    monkeypatch.setattr(cm, "PROCESSED_DATA_DIR", str(curated_dir))
    monkeypatch.setattr(cm, "CURATED_MINUTE_PATH", str(curated_path))

    # Also expose through the package for convenience if needed
    monkeypatch.setattr(ingestion_pkg, "CURATED_MINUTE_PATH", str(curated_path), raising=False)

    return {
        "raw_path": raw_path,
        "curated_path": curated_path,
    }


def test_run_transform_minute_bars_creates_curated_file(tmp_paths):
    raw_path = tmp_paths["raw_path"]
    curated_path = tmp_paths["curated_path"]

    # Create a simple raw CSV with the expected columns plus an extra column
    df_raw = pd.DataFrame(
        [
            {
                "DateTime": "2024-01-01 09:30:00",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Extra": 1,
            },
            {
                "DateTime": "2024-01-01 09:31:00",
                "Open": 100.5,
                "High": 102.0,
                "Low": 100.0,
                "Close": 101.5,
                "Extra": 2,
            },
        ]
    )
    df_raw.to_csv(raw_path, index=False)

    # Run transform; it should clean (via clean_raw_minute_data) and then
    # write a curated CSV with only the canonical OHLC columns.
    result_path = run_transform_minute_bars("NVDA")

    assert result_path == str(curated_path)
    assert curated_path.exists()

    df_curated = pd.read_csv(curated_path)
    # Extra column should have been dropped
    assert list(df_curated.columns) == ["DateTime", "Open", "High", "Low", "Close"]
    assert len(df_curated) == 2


def test_get_raw_and_curated_bars_time_filtering(tmp_paths):
    raw_path = tmp_paths["raw_path"]
    curated_path = tmp_paths["curated_path"]

    # Raw and curated both use DateTime for filtering
    base_time = datetime(2024, 1, 1, 9, 30)
    raw_rows = []
    curated_rows = []
    for i in range(5):
        ts = base_time + timedelta(minutes=i)
        row = {
            "DateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": 100.0 + i,
            "High": 101.0 + i,
            "Low": 99.0 + i,
            "Close": 100.5 + i,
        }
        raw_rows.append(row)
        curated_rows.append(row)

    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    pd.DataFrame(curated_rows).to_csv(curated_path, index=False)

    start = base_time + timedelta(minutes=1)
    end = base_time + timedelta(minutes=4)

    df_raw = get_raw_bars("NVDA", start=start, end=end)
    df_curated = get_curated_bars("NVDA", start=start, end=end)

    # Expect rows for minutes 1, 2, 3
    assert len(df_raw) == 3
    assert len(df_curated) == 3
    assert df_raw["DateTime"].min() == start
    assert df_raw["DateTime"].max() == base_time + timedelta(minutes=3)
    assert df_curated["DateTime"].min() == start
    assert df_curated["DateTime"].max() == base_time + timedelta(minutes=3)