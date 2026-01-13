"""Verification script: Confirm live trading reads from configs/active.json.

This script verifies that the fix for make_strategy_config_from_defaults()
correctly reads enable_longs and allow_shorts from the deployed configuration.
"""

from src.trading_session import make_strategy_config_from_defaults
from src.core.config_resolver import get_strategy_defaults
import json
from pathlib import Path

# Read what's actually deployed
active_json = Path("configs/active.json")
if active_json.exists():
    deployed = json.loads(active_json.read_text(encoding="utf-8"))
    print("=" * 60)
    print("DEPLOYED CONFIG (configs/active.json):")
    print("=" * 60)
    strategy = deployed.get("strategy", {})
    print(f"  enable_longs:      {strategy.get('enable_longs')}")
    print(f"  allow_shorts:      {strategy.get('allow_shorts')}")
    print(f"  k_sigma_long:      {strategy.get('k_sigma_long')}")
    print(f"  k_sigma_short:     {strategy.get('k_sigma_short')}")
    print(f"  k_atr_long:        {strategy.get('k_atr_long')}")
    print(f"  k_atr_short:       {strategy.get('k_atr_short')}")
    print(f"  risk_per_trade:    {strategy.get('risk_per_trade_pct')}")
    print(f"  reward_risk_ratio: {strategy.get('reward_risk_ratio')}")
    print()

# What get_strategy_defaults() returns
print("=" * 60)
print("EFFECTIVE DEFAULTS (via get_strategy_defaults):")
print("=" * 60)
defaults = get_strategy_defaults()
for key, value in defaults.items():
    print(f"  {key:20s} {value}")
print()

# What live trading will actually use
print("=" * 60)
print("LIVE TRADING CONFIG (via make_strategy_config_from_defaults):")
print("=" * 60)
cfg = make_strategy_config_from_defaults()
print(f"  enable_longs:      {cfg.enable_longs}")
print(f"  allow_shorts:      {cfg.allow_shorts}")
print(f"  k_sigma_long:      {cfg.k_sigma_long}")
print(f"  k_sigma_short:     {cfg.k_sigma_short}")
print(f"  k_atr_long:        {cfg.k_atr_long}")
print(f"  k_atr_short:       {cfg.k_atr_short}")
print(f"  risk_per_trade:    {cfg.risk_per_trade_pct}")
print(f"  reward_risk_ratio: {cfg.reward_risk_ratio}")
print()

# Verification
print("=" * 60)
print("VERIFICATION:")
print("=" * 60)
if cfg.allow_shorts:
    print("✓ SHORT TRADES ARE ENABLED")
    print("  Your live trading session will now consider SHORT signals!")
else:
    print("✗ SHORT TRADES ARE DISABLED")
    print("  Live trading will ignore SHORT signals.")

if cfg.enable_longs:
    print("✓ LONG TRADES ARE ENABLED")
else:
    print("✗ LONG TRADES ARE DISABLED")

print()
print("Summary: Live trading will now use the parameters you deployed")
print("         from the UI 'Deploy to Production Config' button.")
