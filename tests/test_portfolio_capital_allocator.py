import pytest
from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator


def test_available_when_no_positions():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for("NVDA", total_equity=10_000.0, open_market_values={})
    assert avail == pytest.approx(3_000.0)  # 30% of 10k


def test_blocked_when_symbol_at_cap():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for(
        "NVDA", total_equity=10_000.0,
        open_market_values={"NVDA": 3_100.0}  # already over per-symbol cap
    )
    assert avail == 0.0


def test_gross_exposure_limits_remaining():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.5)
    alloc = CapitalAllocator(cfg)
    # MSFT already uses 70% of equity — only 10% gross headroom remains
    avail = alloc.available_for(
        "NVDA", total_equity=10_000.0,
        open_market_values={"MSFT": 7_000.0}
    )
    assert avail == pytest.approx(1_000.0)  # min(5k per-sym cap, 1k gross headroom)


def test_unknown_symbol_returns_zero():
    cfg = AllocationConfig(symbols=["NVDA"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for("ZZZZ", total_equity=10_000.0, open_market_values={})
    assert avail == 0.0
