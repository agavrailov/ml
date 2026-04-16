"""Per-symbol capital allocation with gross-exposure guardrails.

Supports two sizing modes:
- ``equal``: flat ``max_per_symbol_pct`` cap for every symbol (default)
- ``erc``: Equal Risk Contribution — per-symbol cap is inversely
  proportional to trailing realised volatility, subject to a hard ceiling
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AllocationConfig:
    symbols: list[str]
    max_gross_exposure_pct: float = 0.80  # max fraction of equity deployed across all positions
    max_per_symbol_pct: float = 0.30      # max fraction of equity in any single symbol (equal mode)
    sizing_mode: str = "equal"            # "equal" or "erc"
    erc_lookback_days: int = 20           # rolling vol window for ERC weights
    erc_max_weight: float = 0.25          # hard ceiling per symbol under ERC


class CapitalAllocator:
    """Computes how much capital a symbol may use for a new position.

    Decision logic (returns the minimum of three constraints):
      1. Per-symbol cap: symbol_cap - current_symbol_exposure
      2. Gross cap:      gross_cap - total_open_exposure
      3. Zero if symbol is not in the configured symbol list

    When ``sizing_mode == "erc"``, the per-symbol cap is derived from
    inverse-volatility weights instead of the flat ``max_per_symbol_pct``.
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._cfg = config
        self._erc_weights: dict[str, float] = {}

    # ── ERC weight computation ──────────────────────────────────────

    def update_erc_weights(self, daily_returns: dict[str, pd.Series]) -> dict[str, float]:
        """Compute Equal Risk Contribution weights from trailing volatility.

        w_i = (1 / sigma_i) / sum(1 / sigma_j)

        Weights are clamped to ``erc_max_weight`` and renormalised.
        Symbols with insufficient data or zero vol get equal share of
        remaining weight.

        Args:
            daily_returns: {symbol: pd.Series of daily log returns}.
                Each series should have at least ``erc_lookback_days`` entries.

        Returns:
            {symbol: weight} dict summing to 1.0 (approximately).
        """
        lookback = self._cfg.erc_lookback_days
        max_w = self._cfg.erc_max_weight
        symbols = [s.upper() for s in self._cfg.symbols]

        inv_vols: dict[str, float] = {}
        for sym in symbols:
            rets = daily_returns.get(sym)
            if rets is None or len(rets) < lookback:
                continue
            tail = rets.iloc[-lookback:]
            sigma = float(np.std(tail, ddof=1))
            if sigma > 0:
                inv_vols[sym] = 1.0 / sigma

        if not inv_vols:
            # No vol data: equal weight
            w = 1.0 / max(len(symbols), 1)
            self._erc_weights = {s: w for s in symbols}
            return dict(self._erc_weights)

        # Raw inverse-vol weights
        total_inv = sum(inv_vols.values())
        raw: dict[str, float] = {s: inv_vols.get(s, 0.0) / total_inv for s in symbols}

        # Assign equal share to symbols without vol data
        missing = [s for s in symbols if s not in inv_vols]
        if missing:
            avg_w = np.mean([raw[s] for s in symbols if s in inv_vols])
            for s in missing:
                raw[s] = avg_w

        # Clamp and renormalise
        clamped = {s: min(w, max_w) for s, w in raw.items()}
        total_c = sum(clamped.values())
        if total_c > 0:
            self._erc_weights = {s: w / total_c for s, w in clamped.items()}
        else:
            w = 1.0 / max(len(symbols), 1)
            self._erc_weights = {s: w for s in symbols}

        return dict(self._erc_weights)

    @property
    def erc_weights(self) -> dict[str, float]:
        """Last computed ERC weights (empty until ``update_erc_weights`` is called)."""
        return dict(self._erc_weights)

    # ── Core allocation ─────────────────────────────────────────────

    def _per_symbol_cap_pct(self, symbol: str) -> float:
        """Return the per-symbol cap fraction based on sizing mode."""
        if self._cfg.sizing_mode == "erc" and self._erc_weights:
            return self._erc_weights.get(symbol.upper(), 0.0)
        return self._cfg.max_per_symbol_pct

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
        per_sym_cap = self._per_symbol_cap_pct(symbol) * total_equity

        gross_headroom = max(0.0, gross_cap - total_open)
        symbol_headroom = max(0.0, per_sym_cap - symbol_open)

        return min(gross_headroom, symbol_headroom)
