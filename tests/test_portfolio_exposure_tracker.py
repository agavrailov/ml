import pytest
from src.portfolio.exposure_tracker import ExposureTracker, PositionSnapshot


def test_gross_exposure_two_longs():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_200.0),
        PositionSnapshot("MSFT", quantity=5, market_value=2_000.0),
    ], total_equity=10_000.0)
    assert tracker.gross_exposure_pct() == pytest.approx(0.32)


def test_net_exposure_long_and_short():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10,  market_value=1_200.0),
        PositionSnapshot("MSFT", quantity=-5, market_value=-2_000.0),
    ], total_equity=10_000.0)
    assert tracker.net_exposure_pct() == pytest.approx(-0.08)   # (1200-2000)/10000


def test_symbol_exposure():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_500.0),
    ], total_equity=10_000.0)
    assert tracker.symbol_exposure_pct("NVDA") == pytest.approx(0.15)
    assert tracker.symbol_exposure_pct("MSFT") == 0.0


def test_open_market_values_dict():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_500.0),
        PositionSnapshot("MSFT", quantity=-5, market_value=-500.0),
    ], total_equity=10_000.0)
    mv = tracker.open_market_values()
    assert mv["NVDA"] == pytest.approx(1_500.0)
    assert mv["MSFT"] == pytest.approx(-500.0)
