"""Data quality checks and reporting for raw NVDA minute data.

This module provides small, explicit checks over the raw minute-level CSV
used by the rest of the pipeline, plus helpers to compute a simple
0–100 quality KPI and format a human-readable report.

All checks are *read-only* and have no side effects.
"""

from __future__ import annotations

from typing import List, Dict

import json
import os

import numpy as np
import pandas as pd

from src.config import RAW_DATA_CSV
from src.data_processing import GAP_ANALYSIS_OUTPUT_JSON


CheckResult = Dict[str, object]


def _get_exchange_trading_days(
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    calendar_name: str = "NASDAQ",
) -> set[pd.Timestamp]:
    """Return expected trading days for a US equity exchange calendar.

    We prefer an official exchange calendar (e.g. NASDAQ/NYSE) when the
    optional dependency :mod:`pandas_market_calendars` is available. When it is
    not installed, we fall back to a simple business-day calendar (``freq="B"``)
    so the check still behaves sensibly without extra packages.
    """

    try:  # Lazy import to avoid hard dependency.
        import pandas_market_calendars as mcal  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        # Fallback: generic business days (no explicit holiday awareness).
        return set(pd.date_range(first_ts.normalize(), last_ts.normalize(), freq="B"))

    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=first_ts.date(), end_date=last_ts.date())
    # ``schedule.index`` is a DatetimeIndex of session opens; normalize to dates.
    return set(pd.to_datetime(schedule.index.normalize()))


def _status_from_bool(ok: bool, warn: bool = False) -> str:
    """Map boolean condition to a status string.

    - ok=True  -> "pass"
    - ok=False and warn=True  -> "warn"
    - ok=False and warn=False -> "fail"
    """

    if ok:
        return "pass"
    if warn:
        return "warn"
    return "fail"


def analyze_raw_minute_data(csv_path: str = RAW_DATA_CSV) -> List[CheckResult]:
    """Run a battery of data-quality checks on the raw minute CSV.

    Returns a list of dictionaries; each has at least the keys::

        id, category, description, status, details

    where ``status`` is one of ``{"pass", "warn", "fail"}``.
    """

    results: List[CheckResult] = []

    # ----------------------------------------------------------------------------------
    # File-level & structural checks
    # ----------------------------------------------------------------------------------
    if not os.path.exists(csv_path):
        results.append(
            {
                "id": "file_exists",
                "category": "Structure",
                "description": f"File exists at {csv_path}",
                "status": "fail",
                "details": "File not found.",
            }
        )
        return results

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    results.append(
        {
            "id": "file_size",
            "category": "Structure",
            "description": "File non-empty",
            "status": _status_from_bool(n_rows > 0),
            "details": f"{n_rows} rows, {n_cols} columns",
        }
    )

    # Handle legacy files where the datetime column header is "index" rather than
    # "DateTime". This mirrors the behaviour in clean_raw_minute_data.
    if "DateTime" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "DateTime"})

    required_cols = ["DateTime", "Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    results.append(
        {
            "id": "required_columns",
            "category": "Structure",
            "description": f"Required columns present: {required_cols}",
            "status": "fail" if missing else "pass",
            "details": f"Missing: {missing}" if missing else "All present.",
        }
    )
    if missing:
        # Without required columns, later checks are not meaningful.
        return results

    # ----------------------------------------------------------------------------------
    # Time coverage & continuity
    # ----------------------------------------------------------------------------------
    # Parse DateTime and drop rows where parsing fails.
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    n_bad_dt = int(df["DateTime"].isna().sum())
    results.append(
        {
            "id": "datetime_parse",
            "category": "Structure",
            "description": "DateTime column parseable as datetime",
            "status": _status_from_bool(n_bad_dt == 0, warn=True),
            "details": f"{n_bad_dt} rows with unparseable DateTime.",
        }
    )

    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    if df.empty:
        # No rows left after dropping bad DateTime values.
        results.append(
            {
                "id": "no_rows_after_datetime_filter",
                "category": "Structure",
                "description": "Any rows remain after filtering invalid DateTime",
                "status": "fail",
                "details": "All rows were dropped due to invalid DateTime values.",
            }
        )
        return results

    first_ts = df["DateTime"].iloc[0]
    last_ts = df["DateTime"].iloc[-1]
    span_days = (last_ts - first_ts).days
    results.append(
        {
            "id": "time_range",
            "category": "Time coverage",
            "description": "First/last timestamp and span",
            "status": "pass",
            "details": f"{first_ts} 		 {last_ts} (span {span_days} days)",
        }
    )

    dup_count = int(df["DateTime"].duplicated().sum())
    results.append(
        {
            "id": "duplicates",
            "category": "Structure",
            "description": "Duplicate DateTime rows",
            "status": _status_from_bool(dup_count == 0, warn=True),
            "details": f"{dup_count} duplicate timestamps.",
        }
    )

    # Missing trading days, distinguishing official exchange holidays from
    # genuinely missing sessions. We use an exchange calendar when available
    # (NASDAQ by default) and otherwise fall back to a generic business-day
    # calendar.
    trading_days = set(pd.to_datetime(df["DateTime"].dt.normalize().unique()))
    generic_bdays = set(
        pd.date_range(first_ts.normalize(), last_ts.normalize(), freq="B")
    )
    exchange_days = _get_exchange_trading_days(first_ts, last_ts, calendar_name="NASDAQ")

    # Days that are business days in a generic calendar but *not* in the
    # exchange calendar are most likely *official holidays*.
    holiday_like_days = sorted(generic_bdays - exchange_days)
    if holiday_like_days:
        results.append(
            {
                "id": "exchange_holidays",
                "category": "Time coverage",
                "description": "Days that look like business days but are holidays on the exchange calendar",
                "status": "pass",
                "details": f"{len(holiday_like_days)} holiday-like business days (info only); first few: {holiday_like_days[:5]}",
            }
        )

    # Days that are expected trading sessions on the exchange calendar but are
    # entirely absent from our data are true gaps.
    missing_trading_days = sorted(exchange_days - trading_days)
    results.append(
        {
            "id": "missing_trading_days",
            "category": "Time coverage",
            "description": "Missing trading days between first/last (holiday-aware NASDAQ calendar)",
            "status": _status_from_bool(len(missing_trading_days) == 0, warn=True),
            "details": f"{len(missing_trading_days)} missing trading days; first few: {missing_trading_days[:5]}",
        }
    )

    # Intra-day gaps > 1 minute.
    df["delta"] = df["DateTime"].diff()
    same_day = df["DateTime"].dt.date == df["DateTime"].shift(1).dt.date
    intraday_gaps = df.loc[same_day & (df["delta"] > pd.Timedelta(minutes=1)), "delta"]
    n_gaps = int(intraday_gaps.shape[0])
    max_gap = intraday_gaps.max() if n_gaps else pd.Timedelta(0)
    results.append(
        {
            "id": "intraday_gaps",
            "category": "Continuity",
            "description": "Intra-day gaps greater than 1 minute",
            "status": _status_from_bool(n_gaps == 0, warn=True),
            "details": f"{n_gaps} intra-day gaps; max gap {max_gap}.",
        }
    )

    # ----------------------------------------------------------------------------------
    # Missing values & basic validity
    # ----------------------------------------------------------------------------------
    price_cols = ["Open", "High", "Low", "Close"]
    nan_counts = df[price_cols].isna().sum()
    total_nans = int(nan_counts.sum())
    nan_details = ", ".join(f"{col}: {int(cnt)}" for col, cnt in nan_counts.items())
    results.append(
        {
            "id": "nan_values",
            "category": "Missing values",
            "description": "NaNs in OHLC price columns",
            "status": _status_from_bool(total_nans == 0, warn=True),
            "details": f"Total NaNs: {total_nans} ({nan_details})",
        }
    )

    negatives = (df[price_cols] < 0).sum().sum()
    zeros = (df[price_cols] == 0).sum().sum()
    results.append(
        {
            "id": "price_sign",
            "category": "Validity",
            "description": "No negative or zero OHLC prices",
            "status": _status_from_bool(negatives == 0 and zeros == 0, warn=True),
            "details": f"Negative values: {int(negatives)}, zeros: {int(zeros)}.",
        }
    )

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    mask_ok = (l <= o) & (l <= c) & (o <= h) & (c <= h) & (l <= h)
    ohlc_bad = int((~mask_ok).sum())
    results.append(
        {
            "id": "ohlc_consistency",
            "category": "Validity",
            "description": "OHLC bounds are internally consistent",
            "status": _status_from_bool(ohlc_bad == 0, warn=True),
            "details": f"{ohlc_bad} rows with inconsistent OHLC values.",
        }
    )

    # ----------------------------------------------------------------------------------
    # Outliers & distribution sanity
    # ----------------------------------------------------------------------------------
    df["ret"] = df["Close"].pct_change()
    ret_std = float(df["ret"].std(skipna=True)) if not df["ret"].dropna().empty else 0.0
    if ret_std > 0:
        k = 6.0
        outliers = df[df["ret"].abs() > k * ret_std]
        n_out = int(outliers.shape[0])
        max_pos = float(outliers["ret"].max()) if n_out else 0.0
        max_neg = float(outliers["ret"].min()) if n_out else 0.0
        results.append(
            {
                "id": "return_outliers",
                "category": "Outliers",
                "description": f"Bars with |return| > {k} * std of returns",
                "status": _status_from_bool(n_out == 0, warn=True),
                "details": f"{n_out} outlier bars; max {max_pos:.4f}, min {max_neg:.4f}.",
            }
        )

    # ----------------------------------------------------------------------------------
    # Existing gap analysis summary, if available
    # ----------------------------------------------------------------------------------
    if os.path.exists(GAP_ANALYSIS_OUTPUT_JSON):
        try:
            with open(GAP_ANALYSIS_OUTPUT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                n_gaps_json = len(data)
                durations = [g.get("duration") for g in data if isinstance(g, dict) and "duration" in g]
                max_dur = max(durations) if durations else None
                details = f"{n_gaps_json} gaps from analyze_gaps.py; max duration {max_dur}."
            else:
                details = "Gap analysis JSON is not a list; structure unknown."
            results.append(
                {
                    "id": "gap_analysis_json",
                    "category": "Continuity",
                    "description": "Existing gap-analysis summary (from analyze_gaps.py)",
                    "status": "pass",
                    "details": details,
                }
            )
        except Exception as exc:  # pragma: no cover - best-effort summary
            results.append(
                {
                    "id": "gap_analysis_json",
                    "category": "Continuity",
                    "description": "Existing gap-analysis summary (from analyze_gaps.py)",
                    "status": "warn",
                    "details": f"Could not read {GAP_ANALYSIS_OUTPUT_JSON}: {exc}",
                }
            )

    return results


def ensure_raw_data_quality(
    *,
    csv_path: str = RAW_DATA_CSV,
    min_score: float = 0.0,
    allow_failures: bool = False,
) -> tuple[list[CheckResult], Dict[str, object]]:
    """Run raw-minute checks and optionally enforce a quality threshold.

    This is a small orchestration wrapper intended for pipeline entrypoints
    (e.g. daily agents, data-processing CLIs). It always returns the full
    ``(checks, kpi)`` tuple; callers may choose whether to treat failures as
    hard errors.

    Parameters
    ----------
    csv_path:
        Path to the raw minute CSV; defaults to :data:`RAW_DATA_CSV`.
    min_score:
        Minimum acceptable ``score_0_100`` from :func:`compute_quality_kpi`.
        Defaults to ``0.0`` so that existing flows can adopt this helper
        without changing behaviour.
    allow_failures:
        When ``False`` and either ``kpi['score_0_100'] < min_score`` or any
        check has ``status == 'fail'``, a :class:`ValueError` is raised.
    """

    checks = analyze_raw_minute_data(csv_path)
    kpi = compute_quality_kpi(checks)

    score = float(kpi.get("score_0_100", 0.0))
    n_fail = int(kpi.get("n_fail", 0))

    if not allow_failures and (score < float(min_score) or n_fail > 0):
        # Keep the exception message concise; callers can log the full report
        # separately via :func:`format_quality_report`.
        raise ValueError(
            f"Raw data quality below threshold: score={score:.1f}, "
            f"failures={n_fail}, min_score={min_score:.1f}"
        )

    return checks, kpi


def get_missing_trading_days(
    csv_path: str = RAW_DATA_CSV,
    calendar_name: str = "NASDAQ",
) -> list[pd.Timestamp]:
    """Return a sorted list of missing exchange trading days (holiday-aware).

    This is a programmatic helper for the UI/backfill logic. It uses the same
    NASDAQ/NYSE-style calendar as :func:`analyze_raw_minute_data` and returns
    only those dates that are expected trading sessions but absent from the
    raw CSV.
    """

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []

    df = pd.read_csv(csv_path)
    if "DateTime" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "DateTime"})

    if "DateTime" not in df.columns:
        return []

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    if df.empty:
        return []

    first_ts = df["DateTime"].iloc[0]
    last_ts = df["DateTime"].iloc[-1]

    trading_days = set(pd.to_datetime(df["DateTime"].dt.normalize().unique()))
    exchange_days = _get_exchange_trading_days(first_ts, last_ts, calendar_name=calendar_name)

    missing_trading_days = sorted(exchange_days - trading_days)
    return list(missing_trading_days)


def validate_hourly_ohlc(df: pd.DataFrame, *, context: str = "hourly_ohlc") -> None:
    """Lightweight sanity checks for resampled OHLC frames.

    This is intentionally strict but small. It is designed to be called from
    :mod:`src.data` so that all downstream consumers (training, evaluation,
    backtests, UI) get consistent guarantees without duplicating checks.

    The function **raises ValueError** on hard violations (missing columns,
    empty frame) and prints nothing; callers can decide how to handle the
    exception at the boundary (CLI/UI/logging).
    """

    required_cols = {"Time", "Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{context}: DataFrame is empty.")

    # Basic type / ordering checks.
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        raise ValueError(f"{context}: 'Time' column must be datetime-like.")

    # Ensure time is strictly increasing to avoid surprises in downstream
    # windowing and backtests.
    if not df["Time"].is_monotonic_increasing:
        raise ValueError(f"{context}: 'Time' must be monotonic increasing.")

    # NaNs in OHLC are treated as hard errors at this stage; earlier pipeline
    # stages (gap filling, cleaning) are responsible for handling them.
    ohlc = df[["Open", "High", "Low", "Close"]]
    if ohlc.isna().any().any():
        raise ValueError(f"{context}: NaNs detected in OHLC columns.")


def validate_feature_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    context: str = "feature_frame",
) -> None:
    """Validate an engineered feature frame used as model input.

    Expectations:

    - Contains a ``Time`` column (datetime-like).
    - Contains at least the requested ``feature_cols``.
    - Contains no NaNs in ``feature_cols``.
    """

    if "Time" not in df.columns:
        raise ValueError(f"{context}: missing 'Time' column.")

    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        raise ValueError(f"{context}: 'Time' column must be datetime-like.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: feature columns missing from frame: {missing}"
        )

    if df[feature_cols].isna().any().any():
        raise ValueError(f"{context}: NaNs detected in feature columns.")


def compute_quality_kpi(checks: List[CheckResult]) -> Dict[str, object]:
    """Compute a simple 0–100 quality score and counts from check results.

    The scoring rule is intentionally simple and transparent:

    - pass -> weight 1.0
    - warn -> weight 0.6
    - fail -> weight 0.0

    The final score is ``100 * mean(weight)`` across all checks.
    """

    n_total = len(checks)
    n_pass = sum(1 for c in checks if c.get("status") == "pass")
    n_warn = sum(1 for c in checks if c.get("status") == "warn")
    n_fail = sum(1 for c in checks if c.get("status") == "fail")

    if n_total == 0:
        score = 0.0
    else:
        weights = {"pass": 1.0, "warn": 0.6, "fail": 0.0}
        total_weight = sum(weights.get(str(c.get("status")), 0.0) for c in checks)
        score = 100.0 * total_weight / float(n_total)

    return {
        "score_0_100": float(score),
        "n_total": int(n_total),
        "n_pass": int(n_pass),
        "n_warn": int(n_warn),
        "n_fail": int(n_fail),
    }


def format_quality_report(
    checks: List[CheckResult],
    kpi: Dict[str, object] | None = None,
    *,
    dataset_name: str = "Raw minute data",
) -> str:
    """Render a plain-text / Markdown-like report for the given checks.

    This is used by the UI to provide a downloadable summary.
    """

    if kpi is None:
        kpi = compute_quality_kpi(checks)

    lines: list[str] = []
    title = f"{dataset_name} quality report"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    score = kpi.get("score_0_100", 0.0)
    n_total = kpi.get("n_total", 0)
    n_pass = kpi.get("n_pass", 0)
    n_warn = kpi.get("n_warn", 0)
    n_fail = kpi.get("n_fail", 0)

    lines.append(f"Overall quality score: {score:.1f} / 100")
    lines.append("")
    lines.append("Summary of checks:")
    lines.append(f"- Total checks: {n_total}")
    lines.append(f"- Passed:       {n_pass}")
    lines.append(f"- Warnings:     {n_warn}")
    lines.append(f"- Failed:       {n_fail}")
    lines.append("")

    if n_fail:
        lines.append("There are FAILED checks that should be addressed before trusting the data.")
        lines.append("")
    elif n_warn:
        lines.append("All checks passed or have minor warnings. Review warnings for potential issues.")
        lines.append("")
    else:
        lines.append("All checks passed.")
        lines.append("")

    # Detailed section
    lines.append("Detailed checks:")
    lines.append("")

    # Sort by category then id for stable output.
    checks_sorted = sorted(
        checks,
        key=lambda c: (str(c.get("category", "")), str(c.get("id", ""))),
    )

    for c in checks_sorted:
        cid = c.get("id", "<unknown>")
        cat = c.get("category", "")
        desc = c.get("description", "")
        status = c.get("status", "")
        details = c.get("details", "")

        lines.append(f"- [{status.upper()}] {cid} ({cat})")
        lines.append(f"  {desc}")
        if details:
            # Indent details slightly for readability.
            for line in str(details).splitlines() or [""]:
                lines.append(f"    -> {line}")
        lines.append("")

    return "\n".join(lines)