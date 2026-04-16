"""Per-symbol capital allocation with gross-exposure guardrails."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationConfig:
    symbols: list[str]
    max_gross_exposure_pct: float = 0.80  # max fraction of equity deployed across all positions
    max_per_symbol_pct: float = 0.30      # max fraction of equity in any single symbol


class CapitalAllocator:
    """Computes how much capital a symbol may use for a new position.

    Decision logic (returns the minimum of three constraints):
      1. Per-symbol cap: symbol_cap - current_symbol_exposure
      2. Gross cap:      gross_cap - total_open_exposure
      3. Zero if symbol is not in the configured symbol list
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._cfg = config

    def available_for(
        self,
        symbol: str,
        total_equity: float,
        open_market_values: dict[str, float],
    ) -> float:
        """Return capital available for a new position in `symbol`.

        Args:
            symbol: Symbol to evaluate.
            total_equity: Current net liquidation value of the account.
            open_market_values: {symbol: current_market_value} for all open positions.

        Returns:
            Non-negative float — the maximum capital that can be deployed.
        """
        if symbol.upper() not in [s.upper() for s in self._cfg.symbols]:
            return 0.0

        total_open = sum(abs(v) for v in open_market_values.values())
        symbol_open = abs(open_market_values.get(symbol.upper(), 0.0))

        gross_cap = self._cfg.max_gross_exposure_pct * total_equity
        per_sym_cap = self._cfg.max_per_symbol_pct * total_equity

        gross_headroom = max(0.0, gross_cap - total_open)
        symbol_headroom = max(0.0, per_sym_cap - symbol_open)

        return min(gross_headroom, symbol_headroom)
