"""Tracks open-position exposure across all symbols."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: float    # positive = long, negative = short
    market_value: float  # signed (negative for short)


class ExposureTracker:
    """Computes gross/net exposure and per-symbol exposure from a position list.

    Updated each poll cycle from the broker snapshot.
    """

    def __init__(self) -> None:
        self._positions: list[PositionSnapshot] = []
        self._total_equity: float = 1.0

    def update(self, positions: list[PositionSnapshot], total_equity: float) -> None:
        """Replace current snapshot. Call once per poll cycle."""
        self._positions = list(positions)
        self._total_equity = max(total_equity, 1.0)

    def gross_exposure_pct(self) -> float:
        """Sum of |market_value| / equity."""
        total = sum(abs(p.market_value) for p in self._positions)
        return total / self._total_equity

    def net_exposure_pct(self) -> float:
        """Sum of signed market_value / equity (long minus short)."""
        total = sum(p.market_value for p in self._positions)
        return total / self._total_equity

    def symbol_exposure_pct(self, symbol: str) -> float:
        """Signed market_value of a single symbol / equity."""
        mv = next(
            (p.market_value for p in self._positions if p.symbol.upper() == symbol.upper()),
            0.0,
        )
        return mv / self._total_equity

    def open_market_values(self) -> dict[str, float]:
        """{symbol: signed_market_value} for all open positions."""
        return {p.symbol.upper(): p.market_value for p in self._positions}
