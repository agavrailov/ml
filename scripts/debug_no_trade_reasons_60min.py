"""Diagnostic script to understand why no new trades are opened after a cutoff.

This is intentionally kept separate from the core backtest engine to avoid
changing production logic. It mirrors the entry logic for the NVDA 60min
backtest and prints compact diagnostics about "no-trade" reasons after the
last filled trade.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from src.backtest import _compute_atr_series, _make_model_prediction_provider
from src.config import (
    K_ATR_MIN_TP,
    K_SIGMA_ERR,
    RISK_PER_TRADE_PCT,
    REWARD_RISK_RATIO,
)
from src.strategy import StrategyConfig, StrategyState


DATA_CSV = "data/processed/nvda_60min.csv"
TRADES_CSV = "backtests/debug_trades.csv"
EQUITY_CSV = "backtests/debug_equity.csv"
FREQUENCY = "60min"


def _load_last_trade_index(trades_path: str) -> int | None:
    trades = pd.read_csv(trades_path)
    if trades.empty:
        return None

    last = trades.iloc[-1]
    exit_idx = last.get("exit_index")
    if isinstance(exit_idx, (int, float)) and not math.isnan(exit_idx):
        return int(exit_idx)
    entry_idx = last.get("entry_index")
    if isinstance(entry_idx, (int, float)) and not math.isnan(entry_idx):
        return int(entry_idx)
    return None


def _build_strategy_config() -> StrategyConfig:
    """Rebuild the same StrategyConfig that the backtest CLI uses by default."""

    return StrategyConfig(
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        reward_risk_ratio=REWARD_RISK_RATIO,
        k_sigma_long=K_SIGMA_ERR,
        k_sigma_short=K_SIGMA_ERR,
        k_atr_long=K_ATR_MIN_TP,
        k_atr_short=K_ATR_MIN_TP,
        min_position_size=1.0,
        enable_longs=True,
        allow_shorts=False,
    )


def _classify_strategy_no_trade_reason(state: StrategyState, cfg: StrategyConfig) -> str:
    """Mirror compute_tp_sl_and_size but return a coarse reason instead of a plan.

    This only covers the *strategy-level* gates once the basic numeric checks
    (NaNs, non-positive ATR/sigma) have passed.
    """

    if state.has_open_position:
        return "in_position"

    if state.current_price <= 0:
        return "feature_nan"

    price = float(state.current_price)
    predicted_return = (state.predicted_price / price) - 1.0
    if predicted_return == 0.0:
        return "below_threshold"  # flat signal

    sigma_return = state.model_error_sigma / price if price > 0.0 else 0.0
    atr_return = state.atr / price if price > 0.0 else 0.0

    # Long-only for this setup; shorts are disabled.
    if predicted_return > 0.0:
        if not cfg.enable_longs:
            return "below_threshold"

        usable_return = predicted_return - cfg.k_sigma_long * sigma_return
        if usable_return <= 0.0:
            return "below_threshold"  # predicted move not big enough vs noise

        if sigma_return > 0.0:
            snr = usable_return / sigma_return
            if snr < cfg.k_atr_long:
                return "below_threshold"  # fails SNR / TP distance filter

        tp_dist = usable_return * price
        if tp_dist <= 0.0:
            return "below_threshold"

        if cfg.reward_risk_ratio <= 0.0:
            return "risk_limit"

        stop_dist = tp_dist / cfg.reward_risk_ratio
        if stop_dist <= 0.0:
            return "risk_limit"

        max_risk_notional = cfg.risk_per_trade_pct * state.account_equity
        if max_risk_notional <= 0.0:
            return "risk_limit"

        risk_per_unit = stop_dist
        if risk_per_unit <= 0.0:
            return "risk_limit"

        size = max_risk_notional / risk_per_unit
        if size < cfg.min_position_size:
            return "insufficient_cash"  # effectively position-size too small

        return "ok"  # strategy would accept a trade here

    # For this config, shorts are disabled; any bearish signal is effectively rejected.
    return "below_threshold"


def main() -> None:
    data = pd.read_csv(DATA_CSV)
    if "Time" in data.columns:
        data["Time"] = pd.to_datetime(data["Time"])

    trades_last_idx = _load_last_trade_index(TRADES_CSV)
    equity_df = pd.read_csv(EQUITY_CSV)

    if trades_last_idx is None:
        print("No trades found in", TRADES_CSV)
        return

    n = len(data)
    if trades_last_idx >= n:
        print("Last trade index is beyond data length:", trades_last_idx, "n=", n)
        return

    last_trade_time = data["Time"].iloc[trades_last_idx] if "Time" in data.columns else trades_last_idx
    print("LAST_TRADE_INDEX", trades_last_idx)
    print("LAST_TRADE_TIME", last_trade_time)

    # Rebuild per-bar risk series in the same way as the backtest.
    atr_series = _compute_atr_series(data, window=14)
    provider, model_sigma_series = _make_model_prediction_provider(data, frequency=FREQUENCY)

    # Mirror the fallback logic in run_backtest_on_dataframe: if the model
    # residual sigma series is degenerate, fall back to ATR as a proxy.
    arr = np.asarray(model_sigma_series, dtype=float)
    if not np.isfinite(arr).any() or np.nanmax(arr) == 0.0:
        model_sigma_series = atr_series

    if not isinstance(model_sigma_series, pd.Series):
        model_sigma_series = pd.Series(model_sigma_series, index=data.index)

    strat_cfg = _build_strategy_config()

    # Sanity-check equity alignment.
    if "equity" not in equity_df.columns:
        raise SystemExit("Equity CSV is missing 'equity' column: " + EQUITY_CSV)

    if len(equity_df) < n:
        print("Warning: equity curve shorter than data; truncating to", len(equity_df))
        n_effective = len(equity_df)
    else:
        n_effective = n

    # Collect reasons for all bars after the last trade (where backtest
    # remained flat) and a compact view for the first ~20 sessions.
    reasons_counter: Counter[str] = Counter()
    debug_rows: list[dict] = []

    start_idx = trades_last_idx + 1
    end_idx = n_effective - 1  # last index where an entry *could* be opened

    for i in range(start_idx, end_idx):
        row = data.iloc[i]
        predicted_price = float(provider(i, row))
        decision_price = float(row["Close"])
        model_sigma = float(model_sigma_series.iloc[i])
        atr_value = float(atr_series.iloc[i])
        equity = float(equity_df["equity"].iloc[i])

        # Mirror the entry gating in run_backtest.
        if not (np.isfinite(predicted_price) and np.isfinite(decision_price)):
            reason = "signal_nan"
        elif not (np.isfinite(model_sigma) and np.isfinite(atr_value)):
            reason = "feature_nan"
        elif model_sigma <= 0.0 or atr_value <= 0.0:
            reason = "below_threshold"  # unusable risk inputs
        else:
            state = StrategyState(
                current_price=decision_price,
                predicted_price=predicted_price,
                model_error_sigma=model_sigma,
                atr=atr_value,
                account_equity=equity,
                has_open_position=False,
            )
            reason = _classify_strategy_no_trade_reason(state, strat_cfg)

        reasons_counter[reason] += 1

        if len(debug_rows) < 20:
            ts = data["Time"].iloc[i] if "Time" in data.columns else i
            debug_rows.append(
                {
                    "index": i,
                    "time": ts,
                    "price": decision_price,
                    "pred": predicted_price,
                    "sigma": model_sigma,
                    "atr": atr_value,
                    "equity": equity,
                    "reason": reason,
                }
            )

    print("NO_TRADE_REASON_COUNTS_AFTER_LAST_TRADE:")
    for reason, count in reasons_counter.most_common():
        print(f"  {reason}: {count}")

    print("\nFIRST_20_SESSIONS_AFTER_LAST_TRADE:")
    for row in debug_rows:
        print(
            f"idx={row['index']:5d} time={row['time']} price={row['price']:8.2f} "
            f"pred={row['pred']:8.2f} sigma={row['sigma']:7.4f} atr={row['atr']:7.4f} "
            f"eq={row['equity']:10.2f} reason={row['reason']}"
        )


if __name__ == "__main__":
    main()
