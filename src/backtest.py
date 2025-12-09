"""Command-line entrypoint for running a simple backtest.

This is a thin wrapper around `backtest_engine.run_backtest` intended for
local, Phase 0 experiments. It:

- Loads OHLC data for the configured frequency from `data/processed/`.
- Builds a default `StrategyConfig` and `BacktestConfig`.
- Uses a naive prediction provider (Close + k * typical_range) for now.

Later this can be extended to use real model predictions.
"""
from __future__ import annotations

import os
import json

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig, BacktestResult, PredictionProvider, run_backtest
from src.data import load_hourly_ohlc
from src.config import (
    FREQUENCY,
    RISK_PER_TRADE_PCT,
    REWARD_RISK_RATIO,
    K_SIGMA_ERR,
    K_ATR_MIN_TP,
    INITIAL_EQUITY,
    COMMISSION_PER_UNIT_PER_LEG,
    MIN_COMMISSION_PER_ORDER,
    BACKTEST_DEFAULT_START_DATE,
    BACKTEST_DEFAULT_END_DATE,
    ROWS_AHEAD,
    TSTEPS,
    get_hourly_data_csv_path,
    get_latest_best_model_path,
    get_predictions_csv_path,
)
from src.strategy import StrategyConfig
from src.predict import (
    build_prediction_context,
    predict_sequence_batch,
)
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler
from src.bias_correction import (
    apply_rolling_bias_and_amplitude_correction,
    compute_rolling_residual_sigma,
)

PREDICTIONS_DIR = "backtests"
# Default rolling window for bias-correction layer in backtests.
BIAS_CORRECTION_WINDOW = 100


def _load_predictions_csv(path: str) -> pd.DataFrame:
    """Load a per-bar predictions CSV with at least Time/predicted_price.

    This helper is shared by CSV-based backtests, paper trading, and notebooks
    so that we have a single, well-defined way to parse and deduplicate
    predictions.
    """

    df = pd.read_csv(path)
    if "predicted_price" not in df.columns:
        raise ValueError(
            f"Predictions CSV {path!r} must contain a 'predicted_price' column; "
            f"found columns={list(df.columns)}",
        )

    # Be permissive about the timestamp column name: Time or DateTime.
    time_col = None
    for cand in ("Time", "DateTime"):
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        if df[time_col].duplicated().any():
            dup_count = int(df[time_col].duplicated().sum())
            print(
                f"[backtest] WARNING: {dup_count} duplicate {time_col} rows in predictions; "
                "keeping first occurrence.",
                flush=True,
            )
            df = df.sort_values(time_col).drop_duplicates(subset=[time_col], keep="first")
    return df


def _align_predictions_to_data(preds_df: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    """Align ``predicted_price`` from ``preds_df`` to ``data``.

    When possible we align on ``Time``; otherwise we fall back to positional
    alignment, trimming or padding with NaNs to match ``len(data)``.
    """

    n = len(data)
    if n == 0:
        return np.zeros(0, dtype=float)

    series: np.ndarray | None = None

    if "Time" in data.columns:
        # Align by Time when present in both frames.
        time_col: str | None = None
        for cand in ("Time", "DateTime"):
            if cand in preds_df.columns:
                time_col = cand
                break

        if time_col is not None:
            left = data[["Time"]].copy()
            left["Time"] = pd.to_datetime(left["Time"])
            right = preds_df[[time_col, "predicted_price"]].copy()
            right[time_col] = pd.to_datetime(right[time_col])
            right = right.rename(columns={time_col: "Time"})

            merged = pd.merge(
                left,
                right,
                on="Time",
                how="left",
            )
            series = merged["predicted_price"].to_numpy(dtype=float)

    if series is None:
        # Positional fallback: trim or pad predictions to match data length.
        base = preds_df["predicted_price"].to_numpy(dtype=float)
        if len(base) >= n:
            series = base[:n]
        else:
            pad = np.full(shape=(n - len(base),), fill_value=np.nan, dtype=float)
            series = np.concatenate([base, pad])

    return series


def _make_csv_prediction_provider(preds_df: pd.DataFrame, data: pd.DataFrame) -> PredictionProvider:
    """Build a prediction provider from a predictions DataFrame.

    The provider returns ``predicted_price`` aligned to ``data`` by Time when
    possible, or by index/length as a fallback. NaNs or out-of-range accesses
    fall back to using the bar's Close so that backtests remain robust.
    """

    n = len(data)
    if n == 0:
        return lambda i, row: float(row["Close"])  # pragma: no cover - defensive

    series = _align_predictions_to_data(preds_df, data)

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        if i < 0 or i >= len(series):
            return float(row["Close"])
        val = series[i]
        if not np.isfinite(val):
            return float(row["Close"])
        return float(val)

    return provider


def _compute_atr_series(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute a rolling ATR-like volatility measure from OHLC data.

    This implements a standard True Range / ATR calculation over ``window``
    bars of the *current timeframe* (e.g. 14 hourly bars for 60min data).

    The first ``window - 1`` bars will contain NaNs; callers can either
    ignore those bars (no trading during warmup) or rely on earlier history
    (e.g. prior trading hours that have been forward-filled) so that by the
    time the session starts, ATR(14) is already defined.
    """

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)

    prev_close = close.shift(1)

    tr_high_low = (high - low).abs()
    tr_high_close = (high - prev_close).abs()
    tr_low_close = (low - prev_close).abs()

    true_range = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)

    # Rolling ATR over the given window; NaN for the first ``window - 1`` bars.
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr


def _estimate_atr_like(data: pd.DataFrame, window: int = 14) -> float:
    """Estimate a single ATR-like scalar for sizing and risk.

    This computes a rolling ATR via :func:`_compute_atr_series` and returns the
    mean of the non-NaN values. If no finite values are available (e.g. too few
    bars), it falls back to ``1.0`` to avoid division-by-zero.
    """

    atr_series = _compute_atr_series(data, window=window)
    cleaned = atr_series.dropna()
    return float(cleaned.mean()) if not cleaned.empty else 1.0


def _make_model_prediction_provider(data: pd.DataFrame, frequency: str) -> tuple[PredictionProvider, np.ndarray]:
    """Return a prediction provider using the LSTM model.

    This implementation builds a reusable prediction context (model + scaler +
    feature metadata), performs feature engineering over the entire dataset
    once, and then runs batched predictions over all sliding windows.

    Predictions are aligned to the *end* of each TSTEPS window and then
    denormalized back to price space. We also checkpoint the final per-bar
    predictions to CSV so they can be reused or inspected later.
    """

    n = len(data)
    if n == 0:
        return (lambda i, row: float(row["Close"])), np.zeros(0, dtype=np.float32)  # pragma: no cover - defensive

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        PREDICTIONS_DIR,
        f"nvda_{frequency}_model_predictions_checkpoint.csv",
    )

    # If a full-length checkpoint exists, load and reuse it *only* when it is
    # shape- and time-aligned with the current data. Otherwise, fall back to a
    # fresh prediction run so that misaligned checkpoints cannot silently turn
    # off trading.
    if os.path.exists(checkpoint_path):
        try:
            checkpoint_df = pd.read_csv(checkpoint_path)
            if "predicted_price" in checkpoint_df.columns and len(checkpoint_df) >= n:
                series = checkpoint_df["predicted_price"].iloc[:n].to_numpy(dtype=float)

                # If the checkpoint also contains a per-bar model_error_sigma
                # column, reuse it so that repeated model-mode runs (and CSV
                # replays of this checkpoint) see the same sigma series that
                # was used when the checkpoint was first created.
                if "model_error_sigma" in checkpoint_df.columns and len(checkpoint_df) >= n:
                    sigma_series = checkpoint_df["model_error_sigma"].iloc[:n].to_numpy(dtype=np.float32)
                else:
                    # Backwards-compatible behaviour for older checkpoints:
                    # use zeros and let the caller fall back to ATR.
                    sigma_series = np.zeros(n, dtype=np.float32)

                is_aligned = True
                if "Time" in data.columns and "Time" in checkpoint_df.columns:
                    try:
                        data_times = pd.to_datetime(data["Time"]).reset_index(drop=True)
                        ckpt_times = pd.to_datetime(checkpoint_df["Time"]).iloc[:n].reset_index(drop=True)
                        # Require both endpoints to match; this is cheap and
                        # robust enough for our single-symbol use case.
                        if not (
                            len(ckpt_times) == n
                            and data_times.iloc[0] == ckpt_times.iloc[0]
                            and data_times.iloc[-1] == ckpt_times.iloc[-1]
                        ):
                            is_aligned = False
                    except Exception:
                        # Any parsing/shape issues -> treat as misaligned.
                        is_aligned = False

                if is_aligned:
                    print(
                        f"Loaded {len(series)} model predictions from checkpoint at {checkpoint_path}",
                        flush=True,
                    )

                    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
                        # Defensive: if checkpoint contains NaNs or we access
                        # out of range, fall back to the bar's Close so that
                        # the backtest continues instead of silently skipping
                        # trades.
                        if i < 0 or i >= len(series):
                            return float(row["Close"])
                        val = series[i]
                        if not np.isfinite(val):
                            return float(row["Close"])
                        return float(val)

                    return provider, sigma_series
        except Exception:
            # Ignore corrupted/partial checkpoints and fall back to fresh computation.
            pass

    # Build prediction context: model + scaler + feature metadata.
    print("Building prediction context and running batched model predictions...", flush=True)
    ctx = build_prediction_context(frequency=frequency, tsteps=TSTEPS)

    df = data.copy()
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])

    # Feature engineering over the entire dataset.
    df_featured = add_features(df, ctx.features_to_use)

    # Normalize using the stored scaler.
    feature_cols = [c for c in df_featured.columns if c != "Time"]
    df_normalized = apply_standard_scaler(df_featured, feature_cols, ctx.scaler_params)

    # Batched predictions over all sliding windows (on feature-engineered data).
    # The model outputs forward log-return predictions on Open prices.
    preds_log = predict_sequence_batch(ctx, df_normalized)

    # Align log-return predictions to the feature-engineered DataFrame first. For
    # M rows and T steps, we have len(preds_log) = M - T + 1 predictions,
    # corresponding to feature rows indices [T-1, T, ..., M-1]. Prepend NaNs for
    # the warmup period on the FEATURED data.
    m = len(df_featured)
    preds_log_full = np.full(shape=(m,), fill_value=np.nan, dtype=np.float32)
    if len(preds_log) > 0:
        preds_log_full[ctx.tsteps - 1 : ctx.tsteps - 1 + len(preds_log)] = preds_log

    # Map log returns back to prices using the raw Open series from the
    # feature-engineered DataFrame.
    base_open = df_featured["Open"].to_numpy(dtype=float)
    denorm_feat = np.full_like(preds_log_full, np.nan, dtype=np.float32)
    mask = np.isfinite(preds_log_full) & np.isfinite(base_open)
    denorm_feat[mask] = base_open[mask] * np.exp(preds_log_full[mask])

    # Now align predictions back to the ORIGINAL raw data length n by joining on
    # Time when available. This accounts for any rows dropped during feature
    # engineering (e.g. due to rolling windows).
    if "Time" in df.columns and "Time" in df_featured.columns:
        preds_df = pd.DataFrame({
            "Time": df_featured["Time"].reset_index(drop=True),
            "predicted_price": denorm_feat,
        })
        merged = pd.merge(
            df[["Time"]].reset_index(drop=True),
            preds_df,
            on="Time",
            how="left",
        )
        denorm_full = merged["predicted_price"].to_numpy(dtype=np.float32)
        times_full = merged["Time"]
    else:
        # Positional fallback: if lengths match, use directly; otherwise pad or
        # truncate to match the raw data length.
        if m >= n:
            denorm_full = denorm_feat[:n]
        else:
            pad = np.full(shape=(n - m,), fill_value=np.nan, dtype=np.float32)
            denorm_full = np.concatenate([pad, denorm_feat])
        times_full = df["Time"] if "Time" in df.columns else pd.Series([pd.NaT] * n)

    # As a final safeguard, ensure prediction and time arrays match the raw data length.
    if len(denorm_full) < n:
        pad_len = n - len(denorm_full)
        denorm_full = np.concatenate([
            denorm_full,
            np.full(shape=(pad_len,), fill_value=np.nan, dtype=np.float32),
        ])
    elif len(denorm_full) > n:
        denorm_full = denorm_full[:n]

    if len(times_full) < n:
        # Pad missing times with NaT at the end.
        pad_len = n - len(times_full)
        times_full = pd.concat([times_full, pd.Series([pd.NaT] * pad_len)], ignore_index=True)
    elif len(times_full) > n:
        times_full = times_full.iloc[:n]

    # Apply a rolling bias-correction layer in price space. For log-return
    # models we avoid mixing units and therefore ignore any global
    # ``mean_residual`` computed during training (which is in log-return
    # space). The correction below relies purely on local price residuals.
    mean_residual = 0.0

    usable_len = max(0, n - ROWS_AHEAD)
    residual_sigma_series = np.zeros(shape=(n,), dtype=np.float32)
    if usable_len > 0:
        preds_trunc = denorm_full[:usable_len].astype(float)
        # Use the original ``data`` (not the feature-engineered ``df``) for
        # actual prices so that shapes are consistent with ``n``.
        if "Open" in data.columns:
            acts_trunc = data["Open"].shift(-ROWS_AHEAD).to_numpy(dtype=float)[:usable_len]
        else:
            acts_trunc = data["Close"].shift(-ROWS_AHEAD).to_numpy(dtype=float)[:usable_len]

        mask = np.isfinite(preds_trunc) & np.isfinite(acts_trunc)
        if mask.any():
            first = int(np.argmax(mask))
            preds_seq = preds_trunc[first:]
            acts_seq = acts_trunc[first:]

            # Bias-only correction in price space. We disable amplitude scaling
            # here (enable_amplitude=False) to avoid distorting the model's
            # natural volatility in price space, which was empirically causing
            # large deviations between checkpointed predictions and the
            # underlying price series in 2025.
            corrected_seq = apply_rolling_bias_and_amplitude_correction(
                predictions=preds_seq,
                actuals=acts_seq,
                window=BIAS_CORRECTION_WINDOW,
                global_mean_residual=mean_residual,
                enable_amplitude=False,
            )
            denorm_full[first:first + len(corrected_seq)] = corrected_seq

            # Rolling residual sigma in price space, aligned to the same window.
            sigma_seq = compute_rolling_residual_sigma(
                predictions=corrected_seq,
                actuals=acts_seq,
                window=BIAS_CORRECTION_WINDOW,
            )
            residual_sigma_series[first:first + len(sigma_seq)] = sigma_seq.astype(np.float32)

    # If we never computed any residuals, fall back to zeros.
    if not np.isfinite(residual_sigma_series).any():
        residual_sigma_series[:] = 0.0

    # Write a single checkpoint CSV with per-bar predictions and residual
    # sigma aligned to raw data so that future runs (and CSV-mode replays) can
    # reuse both without recomputation.
    checkpoint_df = pd.DataFrame({
        "Time": times_full,
        "predicted_price": denorm_full,
        "model_error_sigma": residual_sigma_series,
    })
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"Wrote model prediction checkpoint to {checkpoint_path}", flush=True)

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        if i < 0 or i >= len(denorm_full):
            return float(row["Close"])
        val = denorm_full[i]
        if np.isnan(val):
            # Warmup zone or missing alignment: fall back to Close.
            return float(row["Close"])
        return float(val)

    return provider, residual_sigma_series


def run_backtest_on_dataframe(
    data: pd.DataFrame,
    initial_equity: float = INITIAL_EQUITY,
    frequency: Optional[str] = None,
    prediction_mode: str = "model",
    commission_per_unit_per_leg: float | None = None,
    min_commission_per_order: float | None = None,
    predictions_csv: Optional[str] = None,
    # Optional overrides for strategy parameters (use config defaults when None).
    risk_per_trade_pct: float | None = None,
    reward_risk_ratio: float | None = None,
    # Shared filters (used as base)
    k_sigma_err: float | None = None,
    k_atr_min_tp: float | None = None,
    # Side-specific overrides (take precedence when provided)
    k_sigma_long: float | None = None,
    k_sigma_short: float | None = None,
    k_atr_long: float | None = None,
    k_atr_short: float | None = None,
    enable_longs: bool | None = None,
    allow_shorts: bool | None = None,
) -> BacktestResult:
    """Run a backtest on an in-memory DataFrame using default settings.

    This is primarily for tests and notebook experiments.

    Supported ``prediction_mode`` values:

    - ``"model"``: run the LSTM model on-the-fly via
      :func:`_make_model_prediction_provider`.
    - ``"csv"``: use a per-bar predictions CSV (``predicted_price`` column)
      aligned on ``Time``.
    - ``"naive"``: simple Close-based baseline (used only in legacy tests).
    """

    freq = frequency or FREQUENCY

    # Compute a rolling ATR(14) on the active timeframe (e.g. 14 hourly bars
    # for 60min data). We will feed this per-bar series into the backtest
    # engine so that filters and sizing react to current volatility instead of
    # a single global scalar.
    atr_series = _compute_atr_series(data, window=14)

    # Strategy defaults come from STRATEGY_DEFAULTS so that risk and noise
    # parameters are centralized in config.py. Allow callers to override
    # individual knobs for optimization / UI workflows.
    base_sigma = float(k_sigma_err) if k_sigma_err is not None else K_SIGMA_ERR
    base_atr = float(k_atr_min_tp) if k_atr_min_tp is not None else K_ATR_MIN_TP

    k_sigma_long_eff = float(k_sigma_long) if k_sigma_long is not None else base_sigma
    k_sigma_short_eff = float(k_sigma_short) if k_sigma_short is not None else base_sigma
    k_atr_long_eff = float(k_atr_long) if k_atr_long is not None else base_atr
    k_atr_short_eff = float(k_atr_short) if k_atr_short is not None else base_atr

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=float(risk_per_trade_pct) if risk_per_trade_pct is not None else RISK_PER_TRADE_PCT,
        reward_risk_ratio=float(reward_risk_ratio) if reward_risk_ratio is not None else REWARD_RISK_RATIO,
        k_sigma_long=k_sigma_long_eff,
        k_sigma_short=k_sigma_short_eff,
        k_atr_long=k_atr_long_eff,
        k_atr_short=k_atr_short_eff,
        enable_longs=True if enable_longs is None else bool(enable_longs),
        allow_shorts=False if allow_shorts is None else bool(allow_shorts),
    )

    # Use the mean ATR as a scalar proxy for backwards compatibility, but
    # prefer the full per-bar series when running the engine.
    atr_like = float(atr_series.dropna().mean()) if not atr_series.dropna().empty else 1.0

    bt_cfg = BacktestConfig(
        initial_equity=initial_equity,
        strategy_config=strat_cfg,
        model_error_sigma=atr_like,  # placeholder until real residuals are wired
        fixed_atr=atr_like,
        commission_per_unit_per_leg=commission_per_unit_per_leg
        if commission_per_unit_per_leg is not None
        else COMMISSION_PER_UNIT_PER_LEG,
        min_commission_per_order=min_commission_per_order
        if min_commission_per_order is not None
        else MIN_COMMISSION_PER_ORDER,
    )

    model_error_sigma_series = atr_series

    # Choose prediction provider based on mode.
    if prediction_mode == "csv":
        if not predictions_csv:
            raise ValueError(
                "prediction_mode='csv' requires predictions_csv path (per-bar "
                "predictions with at least 'Time' and 'predicted_price' columns).",
            )

        # Special-case: if the requested CSV is exactly the default model
        # checkpoint path for this frequency, reuse the model-mode provider and
        # sigma logic. `_make_model_prediction_provider` will load the
        # checkpoint (without recomputing predictions) and return the same
        # provider and sigma series that model-mode would use.
        default_checkpoint = os.path.join(PREDICTIONS_DIR, f"nvda_{freq}_model_predictions_checkpoint.csv")
        try:
            if os.path.abspath(predictions_csv) == os.path.abspath(default_checkpoint):
                provider, model_error_sigma_series = _make_model_prediction_provider(data, frequency=freq)
            else:
                preds_df = _load_predictions_csv(predictions_csv)

                # Heuristic: detect whether this CSV already contains bias-corrected
                # model predictions from our own pipeline (i.e. a checkpoint) to avoid
                # applying the bias-correction layer twice.
                csv_basename = os.path.basename(predictions_csv)
                already_corrected = False
                if "model_predictions_checkpoint" in csv_basename:
                    already_corrected = True
                elif "already_corrected" in preds_df.columns:
                    try:
                        already_corrected = bool(pd.Series(preds_df["already_corrected"]).astype(bool).any())
                    except Exception:
                        already_corrected = False

                has_sigma_column = "model_error_sigma" in preds_df.columns

                # Align CSV predictions to the raw data.
                aligned = _align_predictions_to_data(preds_df, data).astype(float)
                n = len(data)
                denorm_full = aligned.copy()

                # If a per-bar sigma column is present, align it using the same
                # Time-based logic as prices.
                sigma_aligned = None
                if has_sigma_column:
                    tmp = preds_df.copy()
                    # Reuse the same alignment helper by temporarily mapping the sigma
                    # column to the expected name.
                    tmp["predicted_price"] = tmp["model_error_sigma"]
                    sigma_aligned = _align_predictions_to_data(tmp, data).astype(float)

                if already_corrected:
                    # Predictions are assumed to be in price space and already passed
                    # through the bias-correction layer (e.g. checkpoint written by
                    # `_make_model_prediction_provider`). In this case we avoid
                    # re-estimating residual sigma and prefer any explicit sigma series
                    # stored alongside the predictions.
                    if sigma_aligned is not None:
                        residual_sigma_series = sigma_aligned
                    else:
                        # Fall back to ATR as a coarse proxy when no sigma column is
                        # available in an already-corrected CSV.
                        residual_sigma_series = atr_series.copy()
                else:
                    # Apply the same bias-correction + residual-sigma estimation that
                    # model-mode uses so that filters see comparable signals.
                    residual_sigma_series = np.zeros(shape=(n,), dtype=np.float32)
                    usable_len = max(0, n - ROWS_AHEAD)
                    if usable_len > 0:
                        preds_trunc = denorm_full[:usable_len].astype(float)
                        if "Open" in data.columns:
                            acts_trunc = data["Open"].shift(-ROWS_AHEAD).to_numpy(dtype=float)[:usable_len]
                        else:
                            acts_trunc = data["Close"].shift(-ROWS_AHEAD).to_numpy(dtype=float)[:usable_len]

                        mask = np.isfinite(preds_trunc) & np.isfinite(acts_trunc)
                        if mask.any():
                            first = int(np.argmax(mask))
                            preds_seq = preds_trunc[first:]
                            acts_seq = acts_trunc[first:]

                            corrected_seq = apply_rolling_bias_and_amplitude_correction(
                                predictions=preds_seq,
                                actuals=acts_seq,
                                window=BIAS_CORRECTION_WINDOW,
                                global_mean_residual=0.0,
                                enable_amplitude=False,
                            )
                            denorm_full[first:first + len(corrected_seq)] = corrected_seq

                            sigma_seq = compute_rolling_residual_sigma(
                                predictions=corrected_seq,
                                actuals=acts_seq,
                                window=BIAS_CORRECTION_WINDOW,
                            )
                            residual_sigma_series[first:first + len(sigma_seq)] = sigma_seq.astype(np.float32)

                    if not np.isfinite(residual_sigma_series).any():
                        residual_sigma_series[:] = 0.0

                def provider(i: int, row: pd.Series) -> float:  # type: ignore[override]
                    if i < 0 or i >= len(denorm_full):
                        return float(row["Close"])
                    val = denorm_full[i]
                    if np.isnan(val):
                        return float(row["Close"])
                    return float(val)

                model_error_sigma_series = residual_sigma_series
        except Exception:
            # Fallback to naive Close-based provider if anything goes wrong
            # reading or interpreting the CSV. This keeps the backtest
            # runnable even with partially corrupted files.
            def provider(i: int, row: pd.Series) -> float:  # type: ignore[override]
                return float(row["Close"])

            model_error_sigma_series = atr_series.copy()
    elif prediction_mode == "model":
        provider, model_error_sigma_series = _make_model_prediction_provider(data, frequency=freq)
    elif prediction_mode == "naive":
        # Legacy/simple baseline: use Close as the predicted price.
        def provider(i: int, row: pd.Series) -> float:  # type: ignore[override]
            return float(row["Close"])
    else:
        raise ValueError(
            f"Unsupported prediction_mode: {prediction_mode!r}. "
            "Supported modes are 'model', 'csv', and 'naive'.",
        )

    # If the residual sigma series is effectively flat/degenerate, fall back to
    # using ATR as a proxy so that k_sigma and k_atr filters remain effective.
    if prediction_mode in {"model", "csv"}:
        try:
            import numpy as _np  # local alias to avoid polluting module namespace

            arr = _np.asarray(model_error_sigma_series, dtype=float)
            finite = arr[_np.isfinite(arr)]
            if finite.size == 0:
                # No finite information about model error -> use ATR as proxy.
                model_error_sigma_series = atr_series
            else:
                sigma_max = float(_np.nanmax(finite))

                # Compute a relative threshold w.r.t ATR so that we only treat
                # truly "collapsed" sigma series (e.g. all zeros up to numerical
                # noise) as degenerate and fall back to ATR.
                atr_arr = _np.asarray(atr_series, dtype=float)
                atr_finite = atr_arr[_np.isfinite(atr_arr)]
                atr_max = float(_np.nanmax(atr_finite)) if atr_finite.size > 0 else None

                use_atr = False
                if sigma_max <= 0.0:
                    use_atr = True
                elif atr_max is not None and atr_max > 0.0:
                    # If sigma is several orders of magnitude smaller than ATR,
                    # treat it as effectively zero and fall back to ATR.
                    if sigma_max < 1e-3 * atr_max:
                        use_atr = True

                if use_atr:
                    model_error_sigma_series = atr_series
        except Exception:
            model_error_sigma_series = atr_series

    # Ensure model_error_sigma_series is a pandas Series so that
    # ``backtest_engine`` can index it consistently.
    if not isinstance(model_error_sigma_series, pd.Series):
        model_error_sigma_series = pd.Series(
            model_error_sigma_series,
            index=data.index,
        )

    # Feed per-bar ATR into the engine so that entry filters and sizing use
    # ATR(14) on the current timeframe. For model-based predictions we provide
    # a separate per-bar ``model_error_sigma_series`` computed from rolling
    # residuals; for other modes we fall back to ATR as a proxy.
    return run_backtest(
        data,
        provider,
        bt_cfg,
        atr_series=atr_series,
        model_error_sigma_series=model_error_sigma_series,
    )


def _apply_date_range(
    data: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Slice ``data`` to a [start_date, end_date] window based on ``Time``.

    Returns (sliced_data, date_from_str, date_to_str).
    If no Time column is present or no dates are provided anywhere, the
    original DataFrame and an index-based range are returned.
    """

    if "Time" not in data.columns or data.empty:
        # Fall back to index-based labelling only.
        date_from = "idx0"
        date_to = f"idx{len(data) - 1}" if len(data) > 0 else "idx-empty"
        return data, date_from, date_to

    # Effective dates: CLI args override config defaults.
    eff_start = start_date or BACKTEST_DEFAULT_START_DATE
    eff_end = end_date or BACKTEST_DEFAULT_END_DATE

    time_series = pd.to_datetime(data["Time"])

    mask = pd.Series(True, index=data.index)
    if eff_start is not None:
        start_ts = pd.to_datetime(eff_start)
        mask &= time_series >= start_ts
    if eff_end is not None:
        end_ts = pd.to_datetime(eff_end)
        mask &= time_series <= end_ts

    sliced = data.loc[mask].copy()
    if sliced.empty:
        # If the requested window produces no rows, fall back to the original
        # data but make it clear in the labels.
        return data, "invalid_start", "invalid_end"

    sliced.reset_index(drop=True, inplace=True)
    ts_sliced = pd.to_datetime(sliced["Time"])
    date_from = ts_sliced.iloc[0].strftime("%Y%m%d")
    date_to = ts_sliced.iloc[-1].strftime("%Y%m%d")
    return sliced, date_from, date_to


def _compute_time_based_years(data: pd.DataFrame) -> float:
    """Best-effort estimate of the span in years covered by ``data``.

    If a ``Time`` column is present and parseable as datetimes, use it;
    otherwise return 0.0 to signal "unknown".
    """

    if "Time" not in data.columns or data.empty:
        return 0.0

    try:
        time_series = pd.to_datetime(data["Time"])
    except Exception:  # pragma: no cover - defensive
        return 0.0

    if len(time_series) < 2:
        return 0.0

    span_seconds = (time_series.iloc[-1] - time_series.iloc[0]).total_seconds()
    if span_seconds <= 0:
        return 0.0

    seconds_per_year = 365.25 * 24 * 60 * 60
    return span_seconds / seconds_per_year


def _compute_backtest_metrics(
    result: BacktestResult,
    initial_equity: float,
    data: pd.DataFrame,
) -> dict:
    """Compute basic risk/return metrics from a ``BacktestResult``.

    Metrics are intentionally simple but aligned with the docs/runbook:
    - total_return
    - cagr (if a time span can be inferred)
    - max_drawdown
    - sharpe_ratio (simple, using per-bar returns)
    - win_rate
    - profit_factor
    - equity_price_corr (correlation of equity and underlying returns)
    """

    equity_curve = np.asarray(result.equity_curve, dtype=float)
    if equity_curve.size == 0:
        return {}

    final_equity = float(equity_curve[-1])
    total_return = final_equity / float(initial_equity) - 1.0 if initial_equity > 0 else 0.0

    # Max drawdown based on running peak.
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0

    # Simple per-bar returns for Sharpe.
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if returns.size > 1 and np.std(returns) > 0:
        # Estimate annualization factor using time-based span when possible.
        years = _compute_time_based_years(data)
        if years > 0:
            periods_per_year = returns.size / years
        else:
            # Fallback: assume ~252 * 6.5 hours of trading, scaled by FREQUENCY.
            try:
                freq_minutes = int(str(FREQUENCY).replace("min", ""))
                bars_per_day = int(6.5 * 60 / max(freq_minutes, 1))
                periods_per_year = 252 * max(bars_per_day, 1)
            except Exception:  # pragma: no cover - defensive fallback
                periods_per_year = 252.0

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Trade-level stats.
    wins = [t.pnl for t in result.trades if t.pnl > 0]
    losses = [t.pnl for t in result.trades if t.pnl < 0]
    n_trades = len(result.trades)
    win_rate = (len(wins) / n_trades) if n_trades > 0 else 0.0

    gross_profit = float(sum(wins)) if wins else 0.0
    gross_loss = float(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

    # CAGR based on time span when available; otherwise 0 to avoid misleading output.
    years = _compute_time_based_years(data)
    if years > 0 and final_equity > 0 and initial_equity > 0:
        cagr = (final_equity / float(initial_equity)) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Correlation between equity returns and underlying price returns.
    equity_price_corr = 0.0
    try:
        if "Close" in data.columns and len(data) > 1 and returns.size > 0:
            prices = data["Close"].astype(float).to_numpy()
            price_returns = np.diff(prices) / prices[:-1]
            # Align lengths: use the shorter of the two series.
            m = min(len(price_returns), len(returns))
            if m > 1 and np.std(price_returns[:m]) > 0 and np.std(returns[:m]) > 0:
                equity_price_corr = float(np.corrcoef(price_returns[:m], returns[:m])[0, 1])
    except Exception:  # pragma: no cover - defensive
        equity_price_corr = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "equity_price_corr": equity_price_corr,
    }


def run_backtest_for_ui(
    frequency: str | None = None,
    initial_equity: float = INITIAL_EQUITY,
    prediction_mode: str = "csv",
    start_date: str | None = None,
    end_date: str | None = None,
    predictions_csv: str | None = None,
    # Optional strategy parameter overrides from UI / optimization.
    risk_per_trade_pct: float | None = None,
    reward_risk_ratio: float | None = None,
    k_sigma_err: float | None = None,
    k_atr_min_tp: float | None = None,
    k_sigma_long: float | None = None,
    k_sigma_short: float | None = None,
    k_atr_long: float | None = None,
    k_atr_short: float | None = None,
    enable_longs: bool | None = None,
    allow_shorts: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convenience wrapper for interactive UIs.

    This mirrors the CLI behaviour in :func:`main` but returns in-memory
    DataFrames instead of writing CSVs or printing to stdout.

    Returns ``(equity_df, trades_df, metrics)`` where:
    - ``equity_df`` has columns ["Time" or "step", "equity", "price"].
    - ``trades_df`` is a tabular view of individual trades (may be empty).
    - ``metrics`` is the same dict produced by :func:`_compute_backtest_metrics`,
      with a few extra metadata fields (period, n_trades).
    """

    freq = frequency or FREQUENCY

    # Load OHLC data in the same way as the CLI.
    csv_path = get_hourly_data_csv_path(freq)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. Please run 'python -m src.data_pipeline' "
            "to generate the necessary data files."
        )

    data_full = load_hourly_ohlc(freq)

    # Apply optional date range slicing based on Time column and config/inputs.
    data, date_from, date_to = _apply_date_range(
        data_full,
        start_date=start_date,
        end_date=end_date,
    )

    # If CSV mode is requested and no explicit predictions_csv is provided,
    # default to the standard NVDA path for the chosen frequency.
    eff_predictions_csv = predictions_csv
    if prediction_mode == "csv" and eff_predictions_csv is None:
        eff_predictions_csv = get_predictions_csv_path("nvda", freq)

    result = run_backtest_on_dataframe(
        data,
        initial_equity=initial_equity,
        frequency=freq,
        prediction_mode=prediction_mode,
        predictions_csv=eff_predictions_csv,
        risk_per_trade_pct=risk_per_trade_pct,
        reward_risk_ratio=reward_risk_ratio,
        k_sigma_err=k_sigma_err,
        k_atr_min_tp=k_atr_min_tp,
        k_sigma_long=k_sigma_long,
        k_sigma_short=k_sigma_short,
        k_atr_long=k_atr_long,
        k_atr_short=k_atr_short,
        enable_longs=enable_longs,
        allow_shorts=allow_shorts,
    )

    metrics = _compute_backtest_metrics(
        result,
        initial_equity=initial_equity,
        data=data,
    )

    # Equity curve DataFrame.
    if "Time" in data.columns:
        time_values = pd.to_datetime(data["Time"])
        time_col_name = "Time"
    else:
        time_values = pd.RangeIndex(len(data))
        time_col_name = "step"

    n = min(len(time_values), len(result.equity_curve), len(data))
    equity_df = pd.DataFrame(
        {
            time_col_name: time_values[:n],
            "equity": list(result.equity_curve)[:n],
            "price": data["Close"].astype(float).iloc[:n].to_list(),
        }
    )

    # Trades DataFrame (may be empty).
    trades_records = [
        {
            "entry_index": t.entry_index,
            "exit_index": t.exit_index,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": t.size,
            "direction": t.direction,
            "commission": t.commission,
            "gross_pnl": t.gross_pnl,
            "pnl": t.pnl,
        }
        for t in result.trades
    ]
    trades_df = pd.DataFrame(trades_records)

    # Attach a few helpful metadata fields to metrics for display.
    enriched_metrics = dict(metrics)
    enriched_metrics["period"] = f"{date_from} -> {date_to}"
    enriched_metrics["n_trades"] = len(result.trades)
    enriched_metrics["final_equity"] = result.final_equity

    return equity_df, trades_df, enriched_metrics


def _plot_price_and_equity_with_trades(
    data: pd.DataFrame,
    result: BacktestResult,
    symbol: str = "NVDA",
    freq: str | None = None,
    k_sigma_err: float | None = None,
    k_atr_min_tp: float | None = None,
    risk_per_trade_pct: float | None = None,
    reward_risk_ratio: float | None = None,
    quiet: bool = False,
) -> str | None:
    """Visualize NVDA price, equity curve, and trades on a single figure.

    - Price (Close) on the primary y-axis.
    - Equity curve on a secondary y-axis (different scale).
    - Buy and sell points marked with different icons and connected by a thin line.

    This is intended for interactive inspection after each CLI backtest run.
    It is safe to call even when matplotlib is not installed (no-op with a message).
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional dependency
        print("matplotlib is not available; skipping backtest plot.", flush=True)
        return None

    if data.empty or not result.equity_curve:
        return None

    # X-axis: prefer explicit timestamps if present, otherwise use bar indices.
    if "Time" in data.columns:
        try:
            times = pd.to_datetime(data["Time"])  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            times = pd.RangeIndex(len(data))
    else:
        times = pd.RangeIndex(len(data))

    prices = data["Close"].astype(float)
    equity = pd.Series(result.equity_curve, dtype=float)

    # Ensure alignment between equity curve and price/time vectors.
    n = min(len(prices), len(equity))
    prices = prices.iloc[:n]
    equity = equity.iloc[:n]
    if hasattr(times, "iloc"):
        times = times.iloc[:n]
    else:
        times = times[:n]

    fig, ax_price = plt.subplots(figsize=(12, 6))
    ax_equity = ax_price.twinx()

    # Transparent background so exported PNGs can be overlaid.
    fig.patch.set_alpha(0.0)
    ax_price.set_facecolor("none")
    ax_equity.set_facecolor("none")

    # Plot NVDA price.
    ax_price.plot(times, prices, color="tab:blue", label=f"{symbol} Close")
    ax_price.set_ylabel(f"{symbol} price", color="tab:blue")
    ax_price.tick_params(axis="y", labelcolor="tab:blue")

    # Plot equity curve on secondary axis.
    ax_equity.plot(times, equity, color="tab:orange", label="Equity")
    ax_equity.set_ylabel("Equity", color="tab:orange")
    ax_equity.tick_params(axis="y", labelcolor="tab:orange")

    # Mark trades: buys with green upward triangles, sells with red downward triangles,
    # connecting each entry/exit pair with a thin gray line.
    buy_labeled = False
    sell_labeled = False
    for t in result.trades:
        # Guard against out-of-range indices.
        if not (0 <= t.entry_index < len(prices) and 0 <= t.exit_index < len(prices)):
            continue

        if hasattr(times, "iloc"):
            t_entry = times.iloc[t.entry_index]
            t_exit = times.iloc[t.exit_index]
        else:  # pragma: no cover - fallback for non-Index-like sequences
            t_entry = times[t.entry_index]
            t_exit = times[t.exit_index]

        # Entry marker (buy).
        ax_price.scatter(
            t_entry,
            t.entry_price,
            marker="^",
            color="green",
            s=40,
            zorder=5,
            label="Buy" if not buy_labeled else None,
        )
        buy_labeled = True

        # Exit marker (sell).
        ax_price.scatter(
            t_exit,
            t.exit_price,
            marker="v",
            color="red",
            s=40,
            zorder=5,
            label="Sell" if not sell_labeled else None,
        )
        sell_labeled = True

        # Thin line connecting entry and exit.
        ax_price.plot(
            [t_entry, t_exit],
            [t.entry_price, t.exit_price],
            color="gray",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )

    ax_price.set_xlabel("Time" if "Time" in data.columns else "Bar index")
    ax_price.grid(True, alpha=0.3)
    ax_price.set_title(f"{symbol} price and equity curve with trades")

    # Combined legend from both axes.
    lines_p, labels_p = ax_price.get_legend_handles_labels()
    lines_e, labels_e = ax_equity.get_legend_handles_labels()
    if lines_p or lines_e:
        ax_price.legend(lines_p + lines_e, labels_p + labels_e, loc="upper left")

    fig.tight_layout()

    # Resolve metadata for filename components, with sensible fallbacks.
    freq_str = str(freq or FREQUENCY)
    k_sigma_val = float(k_sigma_err if k_sigma_err is not None else K_SIGMA_ERR)
    k_atr_val = float(k_atr_min_tp if k_atr_min_tp is not None else K_ATR_MIN_TP)
    risk_pct_val = float(risk_per_trade_pct if risk_per_trade_pct is not None else RISK_PER_TRADE_PCT)
    rr_val = float(reward_risk_ratio if reward_risk_ratio is not None else REWARD_RISK_RATIO)

    # Date range for the backtest, if available.
    if "Time" in data.columns:
        try:
            time_series = pd.to_datetime(data["Time"])
            date_from = time_series.iloc[0].strftime("%Y%m%d")
            date_to = time_series.iloc[-1].strftime("%Y%m%d")
        except Exception:  # pragma: no cover - defensive
            date_from = "unknown"
            date_to = "unknown"
    else:
        date_from = "idx0"
        date_to = f"idx{len(data) - 1}"

    # Save the figure under backtests/ with a descriptive name so that multiple
    # runs can be visually compared over time.
    out_dir = Path(PREDICTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{symbol.upper()}-"
        f"{freq_str}-"
        f"{k_sigma_val:.2f}-"
        f"{k_atr_val:.2f}-"
        f"{risk_pct_val:.4f}-"
        f"{rr_val:.2f}-"
        f"{date_from}-"
        f"{date_to}.png"
    )
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)
    if not quiet:
        print(f"Saved backtest plot to {out_path}", flush=True)
    return str(out_path)


def _print_report(
    symbol: str,
    freq: str,
    period_str: str,
    n_trades: int,
    initial_equity: float,
    final_equity: float,
    metrics: dict,
    plot_path: str | None = None,
) -> None:
    """Print a clean, formatted backtest report to console."""
    width = 55
    line = "=" * width

    print()
    print(line)
    print(f"  BACKTEST REPORT: {symbol} {freq}")
    print(f"  {period_str}")
    print(line)
    print(f"  {'Trades:':<20} {n_trades:>10}")
    print(f"  {'Win Rate:':<20} {metrics.get('win_rate', 0) * 100:>9.1f}%")
    print(f"  {'Total Return:':<20} {metrics.get('total_return', 0) * 100:>+9.1f}%")
    print(f"  {'Max Drawdown:':<20} {metrics.get('max_drawdown', 0) * 100:>9.1f}%")
    print(f"  {'Sharpe Ratio:':<20} {metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"  {'Profit Factor:':<20} {metrics.get('profit_factor', 0):>10.2f}")
    print(f"  {'CAGR:':<20} {metrics.get('cagr', 0) * 100:>9.1f}%")
    print(f"  {'Eq-Price Corr:':<20} {metrics.get('equity_price_corr', 0):>10.2f}")
    print(line)
    print(f"  {'Equity:':<20} {initial_equity:,.0f} -> {final_equity:,.0f}")
    if plot_path:
        print(f"  Plot: {plot_path}")
    print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple LSTM-based trading backtest.")
    parser.add_argument(
        "--frequency",
        type=str,
        default=FREQUENCY,
        help="Resample frequency to use (e.g. '15min', '60min'). Defaults to config FREQUENCY.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="Initial account equity for the backtest.",
    )
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["csv", "model"],
        default="csv",
        help=(
            "Prediction source: 'csv' (precomputed predictions) or 'model' "
            "(LSTM on the fly). CSV mode is recommended for most experiments."
        ),
    )
    parser.add_argument(
        "--commission-per-unit-per-leg",
        type=float,
        default=0.005,
        help=(
            "Commission per unit per leg (entry or exit). "
            "Total round-trip commission = commission_per_unit_per_leg * size * 2. "
            "Set this from your broker's fee schedule (e.g. IBKR) once."
        ),
    )
    parser.add_argument(
        "--min-commission-per-order",
        type=float,
        default=1.0,
        help=(
            "Minimum commission per order (per leg) in account currency. "
            "Total minimum round-trip commission = 2 * min_commission_per_order."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) for the backtest window.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for the backtest window.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Optional override for the input OHLC CSV. When omitted, the NVDA "
            "file under data/processed/ for the chosen frequency is used."
        ),
    )
    parser.add_argument(
        "--export-trades-csv",
        type=str,
        default=None,
        help="If set, write the trade log to this CSV path.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a per-bar predictions CSV with at least "
            "'predicted_price' (and optionally 'Time') columns. "
            "Used when --prediction-mode=csv."
        ),
    )
    parser.add_argument(
        "--export-equity-csv",
        type=str,
        default=None,
        help="If set, write the equity curve to this CSV path.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a clean summary report with key figures. Plot is always saved.",
    )

    args = parser.parse_args()

    freq = args.frequency
    initial_equity = float(args.initial_equity)
    prediction_mode = args.prediction_mode
    commission_per_unit_per_leg = float(args.commission_per_unit_per_leg)
    min_commission_per_order = float(args.min_commission_per_order)
    predictions_csv = args.predictions_csv

    if args.csv_path:
        csv_path = args.csv_path
        data_full = pd.read_csv(csv_path)
    else:
        csv_path = get_hourly_data_csv_path(freq)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"'{csv_path}' not found. Please run 'python -m src.data_pipeline' "
                "to generate the necessary data files, or provide a valid "
                "--csv-path argument."
            )
        # Use centralized data helper to load the default OHLC CSV.
        data_full = load_hourly_ohlc(freq)

    # Basic sanity check for required columns.
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(data_full.columns)
    if missing:
        raise SystemExit(f"Data file {csv_path} is missing required columns: {missing}")

    # Apply optional date range slicing based on Time column and config/CLI.
    data, date_from, date_to = _apply_date_range(
        data_full,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    n_bars = len(data)
    if not args.report:
        print(f"Running backtest on {n_bars} bars at {freq}...", flush=True)

    # Default predictions CSV when using CSV mode and no explicit path is given.
    eff_predictions_csv = predictions_csv
    if prediction_mode == "csv" and eff_predictions_csv is None:
        eff_predictions_csv = get_predictions_csv_path("nvda", freq)

    result = run_backtest_on_dataframe(
        data,
        initial_equity=initial_equity,
        frequency=freq,
        prediction_mode=prediction_mode,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
        predictions_csv=eff_predictions_csv,
    )

    if not args.report:
        print("Backtest complete. Computing metrics...", flush=True)
    metrics = _compute_backtest_metrics(result, initial_equity=initial_equity, data=data)

    # Determine period string for output.
    if "Time" in data.columns and n_bars > 0:
        try:
            time_series = pd.to_datetime(data["Time"])
            start_time = time_series.iloc[0]
            end_time = time_series.iloc[-1]
            period_str = f"{start_time.strftime('%Y-%m-%d')} -> {end_time.strftime('%Y-%m-%d')}"
        except Exception:  # pragma: no cover - best-effort parsing
            period_str = f"{date_from} -> {date_to}"
    else:
        period_str = f"{date_from} -> {date_to}"

    if not args.report:
        # Concise summary output (legacy mode).
        print(
            f"Backtest summary | freq={freq} | bars={n_bars} | period={period_str} | "
            f"equity: {initial_equity:.2f} -> {result.final_equity:.2f} | trades={len(result.trades)}"
        )

        if metrics:
            print(
                "Metrics | "
                f"total_ret={metrics['total_return'] * 100:5.2f}% | "
                f"CAGR={metrics['cagr'] * 100:5.2f}% | "
                f"max_dd={metrics['max_drawdown'] * 100:5.2f}% | "
                f"Sharpe={metrics['sharpe_ratio']:5.2f} | "
                f"win_rate={metrics['win_rate'] * 100:5.2f}% | "
                f"PF={metrics['profit_factor']:5.2f} | "
                f"corr={metrics.get('equity_price_corr', 0):5.2f}"
            )

    # Optional exports.
    if args.export_trades_csv and result.trades:
        trades_df = pd.DataFrame(
            [
                {
                    "entry_index": t.entry_index,
                    "exit_index": t.exit_index,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "direction": t.direction,
                    "commission": t.commission,
                    "gross_pnl": t.gross_pnl,
                    "pnl": t.pnl,
                }
                for t in result.trades
            ]
        )
        trades_df.to_csv(args.export_trades_csv, index=False)

    if args.export_equity_csv:
        equity_df = pd.DataFrame({
            "step": list(range(len(result.equity_curve))),
            "equity": result.equity_curve,
        })
        equity_df.to_csv(args.export_equity_csv, index=False)

    # Interactive visualization: NVDA price + equity curve + trades.
    # This will no-op (with a message) if matplotlib is not installed.
    plot_path = _plot_price_and_equity_with_trades(
        data,
        result,
        symbol="NVDA",
        freq=freq,
        k_sigma_err=K_SIGMA_ERR,
        k_atr_min_tp=K_ATR_MIN_TP,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        reward_risk_ratio=REWARD_RISK_RATIO,
        quiet=args.report,
    )

    # Print clean report when --report flag is set.
    if args.report:
        _print_report(
            symbol="NVDA",
            freq=freq,
            period_str=period_str,
            n_trades=len(result.trades),
            initial_equity=initial_equity,
            final_equity=result.final_equity,
            metrics=metrics,
            plot_path=plot_path,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
