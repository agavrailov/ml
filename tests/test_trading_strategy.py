"""Tests for src.strategy.

These tests use synthetic scenarios to validate the MVP strategy logic.

They are aligned with the current implementation in ``src.strategy``, which
operates primarily in *return space* and supports both long and (optionally)
short entries via separate long/short filters.
"""
from __future__ import annotations

import pytest

from src.strategy import StrategyConfig, StrategyState, compute_tp_sl_and_size


@pytest.fixture
def base_config() -> StrategyConfig:
    """Baseline configuration for long-only trading.

    We use the same thresholds for long and short sides; shorts are disabled by
    default in tests unless explicitly enabled.
    """

    return StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_long=1.0,
        k_sigma_short=1.0,
        k_atr_long=0.5,
        k_atr_short=0.5,
        min_position_size=1.0,
        enable_longs=True,
        allow_shorts=False,
    )


def make_state(
    current_price: float = 100.0,
    predicted_price: float = 101.5,
    model_error_sigma: float = 0.5,
    atr: float = 1.0,
    account_equity: float = 10_000.0,
    has_open_position: bool = False,
) -> StrategyState:
    return StrategyState(
        current_price=current_price,
        predicted_price=predicted_price,
        model_error_sigma=model_error_sigma,
        atr=atr,
        account_equity=account_equity,
        has_open_position=has_open_position,
    )


def test_no_trade_when_already_in_position(base_config: StrategyConfig) -> None:
    state = make_state(has_open_position=True)
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_no_trade_when_price_non_positive(base_config: StrategyConfig) -> None:
    state = make_state(current_price=0.0)
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_no_long_trade_when_prediction_not_bullish(base_config: StrategyConfig) -> None:
    # predicted_price < current_price -> negative predicted_return
    state = make_state(predicted_price=99.0)
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_generates_long_trade_plan_when_conditions_met(base_config: StrategyConfig) -> None:
    # Choose values so predicted_return is clearly positive and volatility
    # assumptions are reasonable.
    state = make_state(predicted_price=105.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is not None

    # Long-only: direction should be +1, TP above price, SL below price, size > 0.
    assert plan.direction == 1
    assert plan.tp_price > state.current_price
    assert plan.sl_price < state.current_price
    assert plan.size >= base_config.min_position_size


def test_trade_rejected_when_position_size_below_minimum(base_config: StrategyConfig) -> None:
    # Make equity tiny so resulting size < min_position_size
    cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_long=1.0,
        k_sigma_short=1.0,
        k_atr_long=0.5,
        k_atr_short=0.5,
        min_position_size=10.0,  # require at least 10 units
        enable_longs=True,
        allow_shorts=False,
    )
    state = make_state(account_equity=100.0, predicted_price=105.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, cfg)
    assert plan is None


def test_invalid_config_parameters_reject_trade(base_config: StrategyConfig) -> None:
    # reward_risk_ratio <= 0 should result in no trade
    bad_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=0.0,
        k_sigma_long=1.0,
        k_sigma_short=1.0,
        k_atr_long=0.5,
        k_atr_short=0.5,
        min_position_size=1.0,
        enable_longs=True,
        allow_shorts=False,
    )
    state = make_state(predicted_price=105.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, bad_cfg)
    assert plan is None

    # Zero or negative equity should also reject
    zero_equity_state = make_state(account_equity=0.0, predicted_price=105.0, model_error_sigma=0.5, atr=1.0)
    plan = compute_tp_sl_and_size(zero_equity_state, base_config)
    assert plan is None


def test_short_entries_only_when_enabled(base_config: StrategyConfig) -> None:
    # By default shorts are disabled, so a bearish prediction should not
    # generate a trade when allow_shorts=False.
    bearish_state = make_state(predicted_price=95.0)
    plan = compute_tp_sl_and_size(bearish_state, base_config)
    assert plan is None

    # Enable shorts and verify we can get a short trade plan.
    short_cfg = StrategyConfig(
        risk_per_trade_pct=base_config.risk_per_trade_pct,
        reward_risk_ratio=base_config.reward_risk_ratio,
        k_sigma_long=base_config.k_sigma_long,
        k_sigma_short=base_config.k_sigma_short,
        k_atr_long=base_config.k_atr_long,
        k_atr_short=base_config.k_atr_short,
        min_position_size=base_config.min_position_size,
        enable_longs=base_config.enable_longs,
        allow_shorts=True,
    )

    bearish_state2 = make_state(predicted_price=95.0)
    plan2 = compute_tp_sl_and_size(bearish_state2, short_cfg)

    assert plan2 is not None
    assert plan2.direction == -1
    assert plan2.tp_price < bearish_state2.current_price
    assert plan2.sl_price > bearish_state2.current_price
    assert plan2.size >= short_cfg.min_position_size
