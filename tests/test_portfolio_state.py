import json
import pytest
from pathlib import Path
from src.portfolio.state import PortfolioStateManager


def test_write_and_read_roundtrip(tmp_path):
    mgr = PortfolioStateManager(tmp_path / "state.json")
    mgr.write_position("NVDA", quantity=10, market_value=1_500.0)
    mgr.write_position("MSFT", quantity=0, market_value=0.0)
    mgr.write_equity(50_000.0)
    state = mgr.read()
    assert state["total_equity"] == 50_000.0
    assert state["positions"]["NVDA"]["market_value"] == 1_500.0


def test_available_capital_respects_caps(tmp_path):
    from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator
    mgr = PortfolioStateManager(tmp_path / "state.json")
    mgr.write_equity(10_000.0)
    mgr.write_position("NVDA", quantity=10, market_value=2_800.0)
    mgr.write_position("MSFT", quantity=0, market_value=0.0)

    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_per_symbol_pct=0.30)
    alloc = CapitalAllocator(cfg)
    avail = mgr.available_capital_for("MSFT", alloc)
    # gross cap=8000; used=2800; gross_headroom=5200; per_sym_cap=3000 → min=3000
    assert avail == pytest.approx(3_000.0)


def test_load_portfolio_config(tmp_path):
    cfg_path = tmp_path / "portfolio.json"
    cfg_path.write_text(json.dumps({
        "symbols": ["NVDA", "MSFT"],
        "frequency": "60min",
        "tsteps": 5,
        "base_client_id": 10,
        "allocation": {"max_gross_exposure_pct": 0.8, "max_per_symbol_pct": 0.3}
    }))
    from src.portfolio.loader import load_portfolio_config
    cfg = load_portfolio_config(str(cfg_path))
    assert cfg["symbols"] == ["NVDA", "MSFT"]
    assert cfg["allocation"]["max_per_symbol_pct"] == 0.3
