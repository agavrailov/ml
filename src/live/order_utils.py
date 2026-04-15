"""Order bracket checking and lifecycle event utilities.

These helpers were extracted from src.ibkr_live_session so that
src.live.poll_loop can import them without pulling in the full
live-session runtime (ib_insync, heartbeat thread, reconnect, etc.).
"""
from __future__ import annotations


def _canonical_order_status(status: object) -> str:
    s = str(status or "").strip()
    low = s.lower()
    if "reject" in low:
        return "REJECTED"
    if "cancel" in low:
        return "CANCELLED"
    if "fill" in low:
        return "FILLED"
    if "submit" in low:
        return "SUBMITTED"
    if not s:
        return "UNKNOWN"
    return s.upper()


def check_positions_have_brackets(
    positions: list[object],
    open_orders: list[object],
    *,
    run_id: str,
    symbol: str,
    ib_trades: list[object] | None = None,
) -> list[dict[str, object]]:
    """Check if all positions have associated TP/SL bracket orders.

    Returns a list of warning events for positions that lack bracket orders.

    A position is considered to have brackets if there are exit orders for both
    TP (LIMIT) and SL (STOP) on the opposite side of the position.

    Args:
        positions: List of PositionInfo objects from broker
        open_orders: List of OrderStatus objects from broker
        run_id: Current run ID for logging
        symbol: Symbol to check (for logging)
        ib_trades: Optional list of raw IB trades (for more detailed order type info)
    """
    warnings: list[dict[str, object]] = []

    if not positions:
        return warnings

    # Build order type map from IB trades if available
    order_types: dict[str, str] = {}
    if ib_trades:
        for trade in ib_trades:
            order = getattr(trade, "order", None)
            if order:
                order_id = str(getattr(order, "orderId", ""))
                order_type = str(getattr(order, "orderType", "")).upper()
                if order_id and order_type:
                    order_types[order_id] = order_type

    # Count open orders per symbol and categorize by type
    orders_by_symbol: dict[str, dict[str, list[object]]] = {}
    for order in open_orders:
        order_symbol = str(getattr(order, "symbol", ""))
        if not order_symbol:
            continue

        order_id = str(getattr(order, "order_id", ""))
        side = str(getattr(order, "side", "")).upper()

        # Get order type from IB trades if available, otherwise use heuristic
        if order_id in order_types:
            otype = order_types[order_id]
        else:
            # Fallback: can't reliably determine without IB trades
            otype = "UNKNOWN"

        if order_symbol not in orders_by_symbol:
            orders_by_symbol[order_symbol] = {"LIMIT": [], "STOP": [], "OTHER": []}

        if "LMT" in otype or "LIMIT" in otype:
            orders_by_symbol[order_symbol]["LIMIT"].append(order)
        elif "STP" in otype or "STOP" in otype:
            orders_by_symbol[order_symbol]["STOP"].append(order)
        else:
            orders_by_symbol[order_symbol]["OTHER"].append(order)

    # Check each position
    for pos in positions:
        pos_symbol = str(getattr(pos, "symbol", ""))
        pos_qty = float(getattr(pos, "quantity", 0.0))

        if not pos_symbol or abs(pos_qty) < 0.01:
            continue

        # Get orders for this symbol
        symbol_order_groups = orders_by_symbol.get(pos_symbol, {"LIMIT": [], "STOP": [], "OTHER": []})

        # Look for exit orders (opposite side of position)
        exit_limit_orders = []
        exit_stop_orders = []

        for order in symbol_order_groups["LIMIT"]:
            side = str(getattr(order, "side", "")).upper()
            is_exit = (pos_qty > 0 and side == "SELL") or (pos_qty < 0 and side == "BUY")
            if is_exit:
                exit_limit_orders.append(order)

        for order in symbol_order_groups["STOP"]:
            side = str(getattr(order, "side", "")).upper()
            is_exit = (pos_qty > 0 and side == "SELL") or (pos_qty < 0 and side == "BUY")
            if is_exit:
                exit_stop_orders.append(order)

        has_tp = len(exit_limit_orders) > 0
        has_sl = len(exit_stop_orders) > 0

        # If position lacks both TP and SL, warn
        if not (has_tp and has_sl):
            warnings.append(
                {
                    "type": "position_missing_brackets",
                    "run_id": run_id,
                    "symbol": pos_symbol,
                    "quantity": pos_qty,
                    "has_tp": has_tp,
                    "has_sl": has_sl,
                    "tp_orders_count": len(exit_limit_orders),
                    "sl_orders_count": len(exit_stop_orders),
                    "total_open_orders": sum(len(v) for v in symbol_order_groups.values()),
                }
            )

    return warnings


def derive_order_lifecycle_events(
    prev_state: dict[str, dict[str, object]],
    orders: list[object],
    *,
    run_id: str,
    symbol: str,
    frequency: str,
    where: str,
    bar_time: str | None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Derive order_status/fill events by diffing broker order snapshots.

    Minimal by design:
    - Emit an `order_status` event whenever (status, filled_qty, avg_fill_price) changes.
    - Emit a `fill` event whenever filled_quantity increases.
    """

    next_state: dict[str, dict[str, object]] = {}
    out: list[dict[str, object]] = []

    for o in orders:
        oid = str(getattr(o, "order_id", "") or "")
        if not oid:
            continue

        status_raw = getattr(o, "status", None)
        status = _canonical_order_status(status_raw)
        filled_qty = float(getattr(o, "filled_quantity", 0.0) or 0.0)
        avg_fill_price = getattr(o, "avg_fill_price", None)
        try:
            avg_fill_price_f = float(avg_fill_price) if avg_fill_price is not None else None
        except Exception:
            avg_fill_price_f = None

        prev = prev_state.get(oid) or {}
        prev_status = str(prev.get("status", ""))
        prev_filled = float(prev.get("filled_quantity", 0.0) or 0.0)

        next_state[oid] = {
            "status": status,
            "filled_quantity": filled_qty,
            "avg_fill_price": avg_fill_price_f,
        }

        if status != prev_status or abs(filled_qty - prev_filled) > 1e-9 or (avg_fill_price_f != prev.get("avg_fill_price")):
            out.append(
                {
                    "type": "order_status",
                    "run_id": run_id,
                    "symbol": symbol,
                    "frequency": frequency,
                    "bar_time": bar_time,
                    "where": where,
                    "order_id": oid,
                    "status": status,
                    "filled_quantity": filled_qty,
                    "avg_fill_price": avg_fill_price_f,
                    # Optional passthrough metadata if present on the OrderStatus object.
                    "side": str(getattr(o, "side", "")),
                    "quantity": float(getattr(o, "quantity", 0.0) or 0.0),
                }
            )

        if filled_qty > prev_filled + 1e-9:
            out.append(
                {
                    "type": "fill",
                    "run_id": run_id,
                    "symbol": symbol,
                    "frequency": frequency,
                    "bar_time": bar_time,
                    "where": where,
                    "order_id": oid,
                    "fill_quantity": float(filled_qty - prev_filled),
                    "filled_quantity_total": filled_qty,
                    "avg_fill_price": avg_fill_price_f,
                    "side": str(getattr(o, "side", "")),
                }
            )

    return next_state, out
