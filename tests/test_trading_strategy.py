"""Tests for src.trading_strategy.

These tests use synthetic scenarios to validate the MVP strategy logic.
"""
from __future__ import annotations

import math

import pytest

from src.trading_strategy import StrategyConfig, StrategyState, compute_tp_sl_and_size


@pytest.fixture
def base_config() -> StrategyConfig:
    return StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=1.0,
        k_atr_min_tp=0.5,
        min_position_size=1.0,
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


def test_no_trade_when_prediction_not_bullish(base_config: StrategyConfig) -> None:
    state = make_state(predicted_price=99.0)  # predicted_move < 0
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_no_trade_when_usable_move_negative(base_config: StrategyConfig) -> None:
    # predicted_move == k_sigma_err * sigma_err -> usable_move == 0
    state = make_state(predicted_price=100.5, model_error_sigma=0.5)
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_no_trade_when_usable_move_smaller_than_atr_threshold(base_config: StrategyConfig) -> None:
    # predicted_move - k_sigma_err * sigma_err = 0.4, but k_atr_min_tp * ATR = 0.5
    # predicted_move = 0.9, sigma_err = 0.5 -> usable_move = 0.4
    # ATR = 1.0, k_atr_min_tp = 0.5 -> min_tp_dist_atr = 0.5
    state = make_state(predicted_price=100.9, model_error_sigma=0.5, atr=1.0)
    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is None


def test_generates_trade_plan_when_conditions_met(base_config: StrategyConfig) -> None:
    # Choose values so usable_move is comfortably above ATR threshold.
    # predicted_move = 2.0, sigma_err = 0.5 -> usable_move = 1.5
    # ATR = 1.0, k_atr_min_tp = 0.5 -> min_tp_dist_atr = 0.5
    state = make_state(predicted_price=102.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, base_config)
    assert plan is not None

    # Basic sanity checks on TP/SL.
    predicted_move = state.predicted_price - state.current_price
    usable_move = predicted_move - base_config.k_sigma_err * state.model_error_sigma
    tp_dist = usable_move
    stop_dist = tp_dist / base_config.reward_risk_ratio

    assert math.isclose(plan.tp_price, state.current_price + tp_dist)
    assert math.isclose(plan.sl_price, state.current_price - stop_dist)

    # Risk per unit should be stop_dist; size chosen so risk_per_unit * size ~= risk_per_trade_pct * equity.
    max_risk_notional = base_config.risk_per_trade_pct * state.account_equity
    expected_size = max_risk_notional / stop_dist
    assert plan.size == pytest.approx(expected_size, rel=1e-6)


def test_trade_rejected_when_position_size_below_minimum(base_config: StrategyConfig) -> None:
    # Make equity tiny so resulting size < min_position_size
    cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=1.0,
        k_atr_min_tp=0.5,
        min_position_size=10.0,  # require at least 10 units
    )
    state = make_state(account_equity=100.0, predicted_price=102.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, cfg)
    assert plan is None


def test_invalid_config_parameters_reject_trade(base_config: StrategyConfig) -> None:
    # reward_risk_ratio <= 0 should result in no trade
    bad_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=0.0,
        k_sigma_err=1.0,
        k_atr_min_tp=0.5,
        min_position_size=1.0,
    )
    state = make_state(predicted_price=102.0, model_error_sigma=0.5, atr=1.0)

    plan = compute_tp_sl_and_size(state, bad_cfg)
    assert plan is None

    # Zero or negative equity should also reject
    zero_equity_state = make_state(account_equity=0.0, predicted_price=102.0, model_error_sigma=0.5, atr=1.0)
    plan = compute_tp_sl_and_size(zero_equity_state, base_config)
    assert plan is None
