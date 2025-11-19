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

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig, BacktestResult, PredictionProvider, run_backtest
from src.config import FREQUENCY, TSTEPS, get_hourly_data_csv_path
from src.trading_strategy import StrategyConfig
from src.predict import (
    build_prediction_context,
    predict_sequence_batch,
)
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler

PREDICTIONS_DIR = "backtests"


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


def _make_naive_prediction_provider(offset_multiple: float, atr_like: float) -> PredictionProvider:
    """Return a simple prediction provider for experimentation.

    For each bar, prediction = Close + offset_multiple * atr_like.
    This ensures a consistent positive edge in tests/experiments without
    calling the real model.
    """

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        return float(row["Close"]) + offset_multiple * atr_like

    return provider


def _make_model_prediction_provider(data: pd.DataFrame, frequency: str) -> PredictionProvider:
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
        return lambda i, row: float(row["Close"])  # pragma: no cover - defensive

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        PREDICTIONS_DIR,
        f"nvda_{frequency}_model_predictions_checkpoint.csv",
    )

    # If a full-length checkpoint exists, load and reuse it.
    if os.path.exists(checkpoint_path):
        try:
            checkpoint_df = pd.read_csv(checkpoint_path)
            if "predicted_price" in checkpoint_df.columns and len(checkpoint_df) >= n:
                series = checkpoint_df["predicted_price"].iloc[:n].to_numpy(dtype=float)
                print(
                    f"Loaded {len(series)} model predictions from checkpoint at {checkpoint_path}",
                    flush=True,
                )

                def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
                    return float(series[i])

                return provider
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
    preds_normalized = predict_sequence_batch(ctx, df_normalized)

    # Align predictions to the feature-engineered DataFrame first. For M rows and
    # T steps, we have len(preds_normalized) = M - T + 1 predictions,
    # corresponding to feature rows indices [T-1, T, ..., M-1]. Prepend NaNs for
    # the warmup period on the FEATURED data.
    m = len(df_featured)
    preds_feat_full = np.full(shape=(m,), fill_value=np.nan, dtype=np.float32)
    if len(preds_normalized) > 0:
        preds_feat_full[ctx.tsteps - 1 : ctx.tsteps - 1 + len(preds_normalized)] = preds_normalized

    # Denormalize using the stored scaler (Open as target) on the FEATURED data.
    denorm_feat = preds_feat_full * ctx.std_vals["Open"] + ctx.mean_vals["Open"]

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

    # Write a single checkpoint CSV with per-bar predictions aligned to raw data.
    checkpoint_df = pd.DataFrame({
        "Time": times_full,
        "predicted_price": denorm_full,
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

    return provider


def _load_predictions_csv(csv_path: str) -> pd.DataFrame:
    """Load a per-bar predictions CSV.

    Expected columns at minimum:
    - ``Time``: timestamp matching the OHLC data.
    - ``predicted_price``: model-predicted price for the target horizon.
    """
    df = pd.read_csv(os.path.normpath(csv_path))
    if "predicted_price" not in df.columns:
        raise ValueError(f"Predictions CSV {csv_path} must contain 'predicted_price' column.")
    return df


def _make_csv_prediction_provider(preds_df: pd.DataFrame, data: pd.DataFrame) -> PredictionProvider:
    """Return a prediction provider backed by a predictions DataFrame.

    If a ``Time`` column is present in both dataframes, we align on Time;
    otherwise we assume positional alignment.
    """
    if "Time" in data.columns and "Time" in preds_df.columns:
        merged = pd.merge(
            data[["Time"]].reset_index(drop=True),
            preds_df[["Time", "predicted_price"]].reset_index(drop=True),
            on="Time",
            how="left",
        )
        series = merged["predicted_price"].fillna(method="ffill").fillna(method="bfill").to_numpy()
    else:
        # Positional fallback.
        series = preds_df["predicted_price"].to_numpy()
        if len(series) < len(data):
            # Pad by repeating last prediction.
            pad_value = series[-1]
            series = np.concatenate([series, np.full(len(data) - len(series), pad_value)])
        elif len(series) > len(data):
            series = series[: len(data)]

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        return float(series[i])

    return provider


def run_backtest_on_dataframe(
    data: pd.DataFrame,
    initial_equity: float = 10_000.0,
    frequency: Optional[str] = None,
    prediction_mode: str = "naive",
    commission_per_unit_per_leg: float = 0.005,
    min_commission_per_order: float = 1.0,
    predictions_csv: Optional[str] = None,
) -> BacktestResult:
    """Run a backtest on an in-memory DataFrame using default settings.

    This is primarily for tests and notebook experiments.
    """

    freq = frequency or FREQUENCY

    # Compute a rolling ATR(14) on the active timeframe (e.g. 14 hourly bars
    # for 60min data). We will feed this per-bar series into the backtest
    # engine so that filters and sizing react to current volatility instead of
    # a single global scalar.
    atr_series = _compute_atr_series(data, window=14)

    # Strategy defaults tuned to be more permissive with realistic noise:
    # - Risk 1% of equity per trade
    # - Require predicted move to clear only 0.75 * sigma_err
    # - Require TP at least 0.5 * ATR away
    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,   # 1% of equity per trade
        reward_risk_ratio=2.0,
        k_sigma_err=0.0,           # ignore model error margin for now
        k_atr_min_tp=0.25,         # require TP only 0.25 * ATR away
    )

    # Use the mean ATR as a scalar proxy for backwards compatibility, but
    # prefer the full per-bar series when running the engine.
    atr_like = float(atr_series.dropna().mean()) if not atr_series.dropna().empty else 1.0

    bt_cfg = BacktestConfig(
        initial_equity=initial_equity,
        strategy_config=strat_cfg,
        model_error_sigma=atr_like,  # placeholder until real residuals are wired
        fixed_atr=atr_like,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
    )

    if prediction_mode == "naive":
        provider = _make_naive_prediction_provider(offset_multiple=2.0, atr_like=atr_like)
    elif prediction_mode == "model":
        provider = _make_model_prediction_provider(data, frequency=freq)
    elif prediction_mode == "csv":
        if not predictions_csv:
            raise ValueError("prediction_mode='csv' requires predictions_csv path")
        preds_df = _load_predictions_csv(predictions_csv)
        provider = _make_csv_prediction_provider(preds_df, data)
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    # Feed per-bar ATR into the engine so that entry filters and sizing use
    # ATR(14) on the current timeframe. We also reuse the same ATR series as a
    # proxy for model_error_sigma, consistent with the previous scalar
    # behavior, but now varying over time.
    return run_backtest(
        data,
        provider,
        bt_cfg,
        atr_series=atr_series,
        model_error_sigma_series=atr_series,
    )


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

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def _plot_price_and_equity_with_trades(
    data: pd.DataFrame,
    result: BacktestResult,
    symbol: str = "NVDA",
) -> None:
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
        return

    if data.empty or not result.equity_curve:
        return

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
    plt.show()


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
        choices=["naive", "model", "csv"],
        default="naive",
        help=(
            "Prediction source: 'naive' (Close + k*ATR), 'model' (LSTM on the fly), "
            "or 'csv' (precomputed predictions CSV)."
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

    args = parser.parse_args()

    freq = args.frequency
    initial_equity = float(args.initial_equity)
    prediction_mode = args.prediction_mode
    commission_per_unit_per_leg = float(args.commission_per_unit_per_leg)
    min_commission_per_order = float(args.min_commission_per_order)
    predictions_csv = args.predictions_csv

    if args.csv_path:
        csv_path = args.csv_path
    else:
        csv_path = get_hourly_data_csv_path(freq)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"'{csv_path}' not found. Please run 'python -m src.data_pipeline' "
                "to generate the necessary data files, or provide a valid "
                "--csv-path argument."
            )
    data = pd.read_csv(csv_path)

    # Basic sanity check for required columns.
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(data.columns)
    if missing:
        raise SystemExit(f"Data file {csv_path} is missing required columns: {missing}")

    n_bars = len(data)
    print(f"Running backtest on {n_bars} bars at {freq}...", flush=True)

    result = run_backtest_on_dataframe(
        data,
        initial_equity=initial_equity,
        frequency=freq,
        prediction_mode=prediction_mode,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
        predictions_csv=predictions_csv,
    )

    print("Backtest complete. Computing metrics...", flush=True)
    metrics = _compute_backtest_metrics(result, initial_equity=initial_equity, data=data)

    # Concise summary output.
    if "Time" in data.columns:
        try:
            time_series = pd.to_datetime(data["Time"])
            start_time = time_series.iloc[0]
            end_time = time_series.iloc[-1]
            period_str = f"{start_time} -> {end_time}"
        except Exception:  # pragma: no cover - best-effort parsing
            period_str = "unknown period"
    else:
        period_str = "unknown period"

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
            f"PF={metrics['profit_factor']:5.2f}"
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
    _plot_price_and_equity_with_trades(data, result, symbol="NVDA")


if __name__ == "__main__":  # pragma: no cover
    main()
