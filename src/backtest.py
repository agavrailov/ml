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
from src.predict import predict_future_prices

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

    For now this calls ``predict_future_prices`` in a sliding-window fashion,
    computing a single prediction for each bar once enough history is
    available. This is slower than a fully vectorized approach but keeps the
    implementation simple and faithful to the online usage.
    """

    preds: list[float] = []
    n = len(data)
    if n == 0:
        return lambda i, row: float(row["Close"])  # pragma: no cover - defensive

    # Emit a simple progress indicator while generating model-based predictions,
    # since this can be relatively slow for large datasets.
    progress_step = max(n // 1000, 1)  # ~0.1% increments

    for i in range(n):
        if i % progress_step == 0:
            pct = (i / n) * 100.0
            print(f"Preparing model predictions: {pct:5.1f}% ({i}/{n} bars)", flush=True)

        window = data.iloc[: i + 1]
        try:
            p = float(predict_future_prices(window, frequency=frequency))
        except Exception:
            # Not enough history yet or model/scaler issues – treat as no signal.
            p = float(window["Close"].iloc[-1])
        preds.append(p)

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        return float(preds[i])

    return provider


def _load_predictions_csv(csv_path: str) -> pd.DataFrame:
    """Load a per-bar predictions CSV.

    Expected columns at minimum:
    - ``Time``: timestamp matching the OHLC data.
    - ``predicted_price``: model-predicted price for the target horizon.
    """
    df = pd.read_csv(csv_path)
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


if __name__ == "__main__":  # pragma: no cover
    main()
