"""Multi-symbol portfolio-level backtester.

Simulates concurrent trading across multiple symbols with shared capital
allocation constraints, producing portfolio-level metrics (Sharpe, drawdown,
pairwise PnL correlation).

Key differences from the single-symbol ``backtest_engine.run_backtest()``:
- Shared equity pool across all symbols
- ``CapitalAllocator`` enforces per-symbol and gross exposure caps before sizing
- One position open per symbol at a time (same as single-symbol engine)
- Bar data from all symbols must be time-aligned (inner join on timestamp)

Usage::

    result = run_portfolio_backtest(symbol_data, config)
    print(f"Portfolio Sharpe: {result.portfolio_sharpe():.2f}")
    print(f"Max drawdown:    {result.max_drawdown():.2%}")
    print(result.pairwise_pnl_correlation())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest_engine import Position, Trade, PredictionProvider
from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator
from src.strategy import StrategyConfig, StrategyState, compute_tp_sl_and_size


@dataclass
class PortfolioBacktestConfig:
    """Configuration for a multi-symbol portfolio backtest."""

    symbols: list[str]
    initial_equity: float
    allocation_config: AllocationConfig
    per_symbol_strategy: dict[str, StrategyConfig]

    commission_per_unit_per_leg: float = 0.001
    min_commission_per_order: float = 1.0


@dataclass
class SymbolBarData:
    """Per-symbol inputs aligned to a common timestamp index."""

    ohlc: pd.DataFrame                    # columns: Open, High, Low, Close
    predictions: PredictionProvider        # (i, row) -> predicted_price
    atr_series: Optional[pd.Series] = None
    model_error_sigma_series: Optional[pd.Series] = None

    # Scalar fallbacks when per-bar series are not provided
    fixed_atr: float = 1.0
    fixed_model_error_sigma: float = 0.005


@dataclass
class PortfolioBacktestResult:
    """Results from a multi-symbol portfolio backtest."""

    portfolio_equity_curve: list[float]
    per_symbol_equity: dict[str, list[float]]   # cumulative PnL per symbol
    per_symbol_trades: dict[str, list[Trade]]
    timestamps: list                             # index labels from aligned data

    def portfolio_sharpe(self, bars_per_year: float = 252 * 6.5) -> float:
        """Annualised Sharpe ratio from the portfolio equity curve."""
        eq = np.array(self.portfolio_equity_curve)
        if len(eq) < 2:
            return 0.0
        returns = np.diff(eq) / eq[:-1]
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown (as a negative fraction)."""
        eq = np.array(self.portfolio_equity_curve)
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.min(dd))

    def total_return(self) -> float:
        """Total return as a fraction."""
        if not self.portfolio_equity_curve:
            return 0.0
        return self.portfolio_equity_curve[-1] / self.portfolio_equity_curve[0] - 1.0

    def pairwise_pnl_correlation(self) -> pd.DataFrame:
        """Correlation matrix of per-bar PnL across symbols."""
        pnl_dict: dict[str, list[float]] = {}
        for sym, eq_curve in self.per_symbol_equity.items():
            arr = np.array(eq_curve)
            pnl_dict[sym] = list(np.diff(arr)) if len(arr) > 1 else []

        if not pnl_dict:
            return pd.DataFrame()

        min_len = min(len(v) for v in pnl_dict.values())
        if min_len < 2:
            return pd.DataFrame()

        df = pd.DataFrame({s: v[:min_len] for s, v in pnl_dict.items()})
        return df.corr()

    def per_symbol_sharpe(self, bars_per_year: float = 252 * 6.5) -> dict[str, float]:
        """Annualised Sharpe per symbol."""
        result = {}
        for sym, eq_curve in self.per_symbol_equity.items():
            eq = np.array(eq_curve)
            if len(eq) < 2:
                result[sym] = 0.0
                continue
            returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
            std = np.std(returns)
            result[sym] = float(np.mean(returns) / std * np.sqrt(bars_per_year)) if std > 0 else 0.0
        return result

    def summary(self) -> dict:
        """Compact summary dict for display or serialisation."""
        return {
            "total_return": self.total_return(),
            "portfolio_sharpe": self.portfolio_sharpe(),
            "max_drawdown": self.max_drawdown(),
            "n_bars": len(self.portfolio_equity_curve),
            "per_symbol_trades": {s: len(t) for s, t in self.per_symbol_trades.items()},
            "per_symbol_sharpe": self.per_symbol_sharpe(),
        }


def run_portfolio_backtest(
    symbol_data: dict[str, SymbolBarData],
    config: PortfolioBacktestConfig,
) -> PortfolioBacktestResult:
    """Run a multi-symbol portfolio backtest with shared capital allocation.

    Parameters
    ----------
    symbol_data : dict[str, SymbolBarData]
        Per-symbol OHLC + predictions, pre-aligned to a common timestamp index.
        All DataFrames must have the same length and index.
    config : PortfolioBacktestConfig
        Portfolio-level configuration including allocation constraints.

    Returns
    -------
    PortfolioBacktestResult
    """
    symbols = config.symbols
    allocator = CapitalAllocator(config.allocation_config)

    # Validate all symbol data has the same length
    lengths = {sym: len(sd.ohlc) for sym, sd in symbol_data.items() if sym in symbols}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Symbol data must have the same length after alignment. Got: {lengths}"
        )

    n = next(iter(lengths.values())) if lengths else 0
    if n == 0:
        return PortfolioBacktestResult(
            portfolio_equity_curve=[config.initial_equity],
            per_symbol_equity={s: [0.0] for s in symbols},
            per_symbol_trades={s: [] for s in symbols},
            timestamps=[],
        )

    equity = config.initial_equity
    portfolio_equity_curve: list[float] = []
    per_symbol_pnl: dict[str, list[float]] = {s: [] for s in symbols}
    per_symbol_trades: dict[str, list[Trade]] = {s: [] for s in symbols}
    positions: dict[str, Optional[Position]] = {s: None for s in symbols}

    # Extract timestamps from first symbol
    first_sym = symbols[0]
    timestamps = list(symbol_data[first_sym].ohlc.index)

    for i in range(n):
        bar_pnl: dict[str, float] = {s: 0.0 for s in symbols}

        # ── Phase 1: Check exits on all open positions ──────────────
        for sym in symbols:
            pos = positions[sym]
            if pos is None:
                continue

            sd = symbol_data[sym]
            row = sd.ohlc.iloc[i]
            high = float(row["High"])
            low = float(row["Low"])

            exit_price: Optional[float] = None

            if pos.direction > 0:
                if low <= pos.sl_price:
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    exit_price = pos.tp_price

            if exit_price is not None:
                per_leg = max(
                    config.commission_per_unit_per_leg * pos.size,
                    config.min_commission_per_order,
                )
                commission = per_leg * 2.0
                trade = Trade(
                    entry_index=pos.entry_index,
                    exit_index=i,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size=pos.size,
                    direction=pos.direction,
                    commission=commission,
                )
                bar_pnl[sym] += trade.pnl
                per_symbol_trades[sym].append(trade)
                positions[sym] = None

        # Apply PnL from exits
        equity += sum(bar_pnl.values())

        # ── Phase 2: Consider new entries ───────────────────────────
        if i < n - 1:
            # Compute current open market values for the allocator
            open_market_values: dict[str, float] = {}
            for sym in symbols:
                pos = positions[sym]
                if pos is not None:
                    # Approximate market value as current bar close * size
                    current_close = float(symbol_data[sym].ohlc.iloc[i]["Close"])
                    open_market_values[sym] = abs(current_close * pos.size)

            for sym in symbols:
                if positions[sym] is not None:
                    continue  # already in a position

                sd = symbol_data[sym]
                row = sd.ohlc.iloc[i]
                strategy_cfg = config.per_symbol_strategy.get(sym)
                if strategy_cfg is None:
                    continue

                predicted_price = float(sd.predictions(i, row))
                decision_price = float(row["Close"])

                # Resolve per-bar risk inputs
                if sd.atr_series is not None and i < len(sd.atr_series):
                    atr_value = float(sd.atr_series.iloc[i])
                else:
                    atr_value = sd.fixed_atr

                if sd.model_error_sigma_series is not None and i < len(sd.model_error_sigma_series):
                    model_sigma = float(sd.model_error_sigma_series.iloc[i])
                else:
                    model_sigma = sd.fixed_model_error_sigma

                # Skip if inputs are invalid
                if not (np.isfinite(predicted_price) and np.isfinite(decision_price)):
                    continue
                if not (np.isfinite(model_sigma) and np.isfinite(atr_value)):
                    continue
                if model_sigma <= 0.0 or atr_value <= 0.0:
                    continue

                # Check capital allocation
                available = allocator.available_for(sym, equity, open_market_values)
                if available <= 0:
                    continue

                strat_state = StrategyState(
                    current_price=decision_price,
                    predicted_price=predicted_price,
                    model_error_sigma=model_sigma,
                    atr=atr_value,
                    account_equity=equity,
                    buying_power=available,
                    has_open_position=False,
                )

                trade_plan = compute_tp_sl_and_size(strat_state, strategy_cfg)
                if trade_plan is None:
                    continue

                # Open position at next bar's Open
                next_row = sd.ohlc.iloc[i + 1]
                entry_price = float(next_row["Open"])
                positions[sym] = Position(
                    entry_index=i + 1,
                    entry_price=entry_price,
                    size=trade_plan.size,
                    tp_price=trade_plan.tp_price,
                    sl_price=trade_plan.sl_price,
                    direction=trade_plan.direction,
                )

                # Update open market values for subsequent symbol allocation checks
                open_market_values[sym] = abs(entry_price * trade_plan.size)

        # Record equity
        portfolio_equity_curve.append(equity)
        for sym in symbols:
            per_symbol_pnl[sym].append(bar_pnl[sym])

    # ── Close any positions still open at end ───────────────────
    for sym in symbols:
        pos = positions[sym]
        if pos is None:
            continue
        sd = symbol_data[sym]
        last_row = sd.ohlc.iloc[-1]
        exit_price = float(last_row["Close"])
        per_leg = max(
            config.commission_per_unit_per_leg * pos.size,
            config.min_commission_per_order,
        )
        commission = per_leg * 2.0
        trade = Trade(
            entry_index=pos.entry_index,
            exit_index=n - 1,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            direction=pos.direction,
            commission=commission,
        )
        equity += trade.pnl
        per_symbol_trades[sym].append(trade)
        if portfolio_equity_curve:
            portfolio_equity_curve[-1] = equity

    # Convert per-bar PnL to cumulative equity contribution per symbol
    per_symbol_equity: dict[str, list[float]] = {}
    for sym in symbols:
        initial_alloc = config.initial_equity / len(symbols)
        cumulative = [initial_alloc]
        for pnl in per_symbol_pnl[sym]:
            cumulative.append(cumulative[-1] + pnl)
        per_symbol_equity[sym] = cumulative

    return PortfolioBacktestResult(
        portfolio_equity_curve=portfolio_equity_curve,
        per_symbol_equity=per_symbol_equity,
        per_symbol_trades=per_symbol_trades,
        timestamps=timestamps,
    )


def align_symbol_data(
    symbol_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align multiple symbol DataFrames by timestamp (inner join).

    Each DataFrame must have a datetime-parseable index or a 'Time' column.
    Returns DataFrames with identical index (intersection of all timestamps).
    """
    aligned: dict[str, pd.DataFrame] = {}

    # Ensure all have datetime index
    processed: dict[str, pd.DataFrame] = {}
    for sym, df in symbol_dfs.items():
        df = df.copy()
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        processed[sym] = df

    # Find common timestamps
    common_idx = None
    for sym, df in processed.items():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        return {}

    common_idx = common_idx.sort_values()

    for sym, df in processed.items():
        aligned[sym] = df.loc[common_idx].copy()

    return aligned
