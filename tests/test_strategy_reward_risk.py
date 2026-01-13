"""Test reward:risk ratio calculations and validate sane defaults.

This test was added after discovering a critical bug where the code default
reward_risk_ratio was 0.1, causing positions to risk 10x what they expected
to make (inverted risk profile).
"""
import pytest

from src.strategy import StrategyConfig, StrategyState, compute_tp_sl_and_size


def test_reward_risk_ratio_sanity():
    """Verify that reward:risk ratio produces reasonable TP/SL distances."""
    
    cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.5,  # Should make 2.5x what we risk
        k_sigma_long=0.9,
        k_sigma_short=0.9,
        k_atr_long=2.0,
        k_atr_short=2.0,
        enable_longs=True,
        allow_shorts=False,
    )
    
    state = StrategyState(
        current_price=100.0,
        predicted_price=105.0,  # 5% upward move
        model_error_sigma=0.5,  # Small error
        atr=1.0,
        account_equity=10000.0,
        has_open_position=False,
    )
    
    plan = compute_tp_sl_and_size(state, cfg)
    
    assert plan is not None, "Should generate a trade plan for clear long signal"
    
    # Calculate distances
    tp_dist = plan.tp_price - state.current_price
    sl_dist = state.current_price - plan.sl_price
    
    # TP should be ABOVE entry
    assert plan.tp_price > state.current_price, f"TP {plan.tp_price} should be above entry {state.current_price}"
    
    # SL should be BELOW entry
    assert plan.sl_price < state.current_price, f"SL {plan.sl_price} should be below entry {state.current_price}"
    
    # Both distances should be positive
    assert tp_dist > 0, f"TP distance should be positive, got {tp_dist}"
    assert sl_dist > 0, f"SL distance should be positive, got {sl_dist}"
    
    # Calculate actual reward:risk ratio
    actual_rr = tp_dist / sl_dist
    
    # Should be close to configured ratio (within 1% tolerance)
    expected_rr = cfg.reward_risk_ratio
    assert abs(actual_rr - expected_rr) < expected_rr * 0.01, \
        f"Reward:Risk should be ~{expected_rr}:1, got {actual_rr:.2f}:1 (TP={plan.tp_price:.2f}, SL={plan.sl_price:.2f})"
    
    # Sanity check: Should never risk more than we expect to make
    assert tp_dist > sl_dist, \
        f"Should NEVER risk more than we expect to make! TP dist={tp_dist:.2f}, SL dist={sl_dist:.2f}"


def test_inverted_ratio_detection():
    """Document that reward_risk_ratio < 1.0 produces inverted risk profiles.
    
    This is a regression test documenting the bug that was in production:
    The code default was 0.1, causing traders to risk 10x what they expected to make.
    
    The strategy code itself is correct - it just needs sane config values.
    """
    
    # This was the buggy default: reward_risk_ratio = 0.1
    # This means: risk 10x what you expect to make (terrible!)
    cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=0.1,  # INVERTED! Risk 10x the reward
        k_sigma_long=0.9,
        k_sigma_short=0.9,
        k_atr_long=2.0,
        k_atr_short=2.0,
    )
    
    state = StrategyState(
        current_price=100.0,
        predicted_price=105.0,
        model_error_sigma=0.5,
        atr=1.0,
        account_equity=10000.0,
        has_open_position=False,
    )
    
    plan = compute_tp_sl_and_size(state, cfg)
    
    assert plan is not None, "Should generate plan even with bad config"
    
    tp_dist = plan.tp_price - state.current_price
    sl_dist = state.current_price - plan.sl_price
    
    # With reward_risk_ratio=0.1, the formula produces: stop_dist = tp_dist / 0.1 = tp_dist * 10
    # So we EXPECT an inverted risk profile (this documents the bug)
    assert sl_dist > tp_dist, \
        f"With reward_risk_ratio=0.1, we expect inverted profile (bug case). " \
        f"Got TP={tp_dist:.2f}, SL={sl_dist:.2f}"
    
    # The ratio should be approximately 10:1 (risk:reward)
    ratio = sl_dist / tp_dist
    assert abs(ratio - 10.0) < 0.1, \
        f"With reward_risk_ratio=0.1, expect ~10:1 risk:reward ratio, got {ratio:.1f}:1"


def test_code_default_is_sane():
    """Ensure the code default reward_risk_ratio is reasonable."""
    from src.config import STRATEGY_DEFAULTS
    
    # Code default should never be less than 1.5 (bare minimum for viable trading)
    # Ideally >= 2.0
    assert STRATEGY_DEFAULTS.reward_risk_ratio >= 1.5, \
        f"Code default reward_risk_ratio={STRATEGY_DEFAULTS.reward_risk_ratio} is too low! " \
        f"This will cause traders to risk more than they make. Must be >= 1.5, ideally >= 2.0"
    
    # Warn if it's between 1.5 and 2.0 (technically valid but marginal)
    if STRATEGY_DEFAULTS.reward_risk_ratio < 2.0:
        pytest.warns(
            UserWarning,
            match=f"reward_risk_ratio={STRATEGY_DEFAULTS.reward_risk_ratio} is below 2.0"
        )
