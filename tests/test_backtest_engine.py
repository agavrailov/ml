"""Tests for src.backtest_engine.

These tests use simple synthetic price and prediction series to validate
basic backtest behavior (no trades, TP, SL, equity updates).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_engine import BacktestConfig, BacktestResult, run_backtest
from src.strategy import StrategyConfig


def _constant_prediction_provider(predicted_price: float):
    def _provider(i, row):  # noqa: ANN001
        return predicted_price

    return _provider


def test_no_trades_when_prediction_too_small_vs_noise() -> None:
    # Price barely moves, prediction small relative to model error => no trades.
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.1, 100.3, 100.5],
            "Close": [100.0, 100.2, 100.4, 100.6],
            "High": [100.2, 100.4, 100.6, 100.8],
            "Low": [99.8, 100.0, 100.2, 100.4],
        }
    )

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=0.0,
        k_atr_min_tp=1.0,  # require usable_return / sigma_return >= 1.0
    )
    cfg = BacktestConfig(
        initial_equity=10_000.0,
        strategy_config=strat_cfg,
        model_error_sigma=1.0,  # sigma in price units -> sigma_return = 0.01
        fixed_atr=1.0,
    )

    # Prediction only 0.3 above current price; predicted_return=0.003,
    # sigma_return=0.01 -> snr=0.3 < 1.0, so no trades.
    provider = _constant_prediction_provider(predicted_price=100.3)

    result: BacktestResult = run_backtest(data, provider, cfg)

    assert result.trades == []
    # Equity should remain unchanged at all times.
    assert all(eq == cfg.initial_equity for eq in result.equity_curve)


def test_single_tp_trade_with_positive_pnl() -> None:
    # Construct a simple upward trend where a single trade should hit TP.
    data = pd.DataFrame(
        {
            # Price moves up steadily.
            "Open": [100.0, 100.8, 102.5, 104.5],
            "Close": [100.0, 101.0, 103.0, 105.0],
            "High": [100.5, 102.0, 104.0, 106.0],
            "Low": [99.5, 100.5, 102.0, 104.0],
        }
    )

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=0.0,
        k_atr_min_tp=0.1,
    )
    cfg = BacktestConfig(
        initial_equity=10_000.0,
        strategy_config=strat_cfg,
        model_error_sigma=0.0,
        fixed_atr=0.5,
    )

    # Predict a price far above current Close so that usable_move is large
    # and TP is well within the observed highs.
    def provider(i, row):  # noqa: ANN001
        return float(row["Close"]) + 2.0

    result = run_backtest(data, provider, cfg)

    # We expect at least one trade with positive PnL.
    assert len(result.trades) >= 1
    pnl_values = [t.pnl for t in result.trades]
    assert all(pnl >= 0 for pnl in pnl_values)
    assert result.final_equity >= cfg.initial_equity


def test_single_sl_trade_with_negative_pnl() -> None:
    # Downward move where a long trade should be stopped out.
    data = pd.DataFrame(
        {
            "Open": [100.0, 99.5, 98.5, 97.5],
            "Close": [100.0, 99.0, 98.0, 97.0],
            "High": [100.5, 99.5, 98.5, 97.5],
            "Low": [99.5, 98.5, 97.5, 96.5],
        }
    )

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=0.0,
        k_atr_min_tp=0.1,
    )
    cfg = BacktestConfig(
        initial_equity=10_000.0,
        strategy_config=strat_cfg,
        model_error_sigma=0.0,
        fixed_atr=0.5,
    )

    # Still predict higher prices so the strategy wants to go long,
    # but actual prices fall and should hit SL first.
    def provider(i, row):  # noqa: ANN001
        return float(row["Close"]) + 2.0

    result = run_backtest(data, provider, cfg)

    assert len(result.trades) >= 1
    pnl_values = [t.pnl for t in result.trades]
    assert any(pnl < 0 for pnl in pnl_values)
    assert result.final_equity <= cfg.initial_equity


def test_commission_reduces_pnl_consistently() -> None:
    # Use the same upward-trend scenario but with commission enabled and
    # verify that net PnL is reduced by exactly the commission amount.
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.8, 102.5, 104.5],
            "Close": [100.0, 101.0, 103.0, 105.0],
            "High": [100.5, 102.0, 104.0, 106.0],
            "Low": [99.5, 100.5, 102.0, 104.0],
        }
    )

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=0.0,
        k_atr_min_tp=0.1,
    )
    cfg = BacktestConfig(
        initial_equity=10_000.0,
        strategy_config=strat_cfg,
        model_error_sigma=0.0,
        fixed_atr=0.5,
        commission_per_unit_per_leg=0.005,
        min_commission_per_order=0.0,
    )

    def provider(i, row):  # noqa: ANN001
        return float(row["Close"]) + 2.0

    result = run_backtest(data, provider, cfg)

    assert len(result.trades) >= 1
    for trade in result.trades:
        # Gross - net should equal commission
        assert trade.gross_pnl - trade.pnl == pytest.approx(trade.commission, rel=1e-9)
