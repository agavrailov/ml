"""Send a small test order to IBKR via TWS API (paper account recommended).

This script is intentionally minimal and is meant to verify connectivity
between this repo and an IBKR TWS/IB Gateway instance.

Usage (example):

    python -m src.ibkr_test_order --symbol NVDA --quantity 1 --send

By default the script runs in **dry-run** mode and only logs what it *would*
submit. Passing ``--send`` actually sends the order via ``IBKRBrokerTws``.

IMPORTANT:
- Make sure TWS or IB Gateway is running and logged into a **paper** account.
- Connection parameters (host, port, client id) are taken from ``config.IB`` by
  default but can be overridden via CLI flags.
"""
from __future__ import annotations

import argparse

from src.broker import OrderRequest, OrderType, Side
from src.config import IB as IbConfig
from src.ibkr_broker import IBKRBrokerConfig, IBKRBrokerTws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a small test order to IBKR via TWS API.")
    parser.add_argument("--symbol", type=str, default=IbConfig.contract_details.get("symbol", "NVDA"))
    parser.add_argument("--quantity", type=float, default=1.0, help="Order quantity (shares).")
    parser.add_argument(
        "--side",
        type=str,
        default="BUY",
        choices=["BUY", "SELL"],
        help="Order side (BUY or SELL).",
    )
    parser.add_argument(
        "--order-type",
        type=str,
        default="MKT",
        choices=["MKT", "LMT"],
        help="Order type: MKT (market) or LMT (limit).",
    )
    parser.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help="Limit price for LMT orders (required when --order-type LMT).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=IbConfig.host,
        help="TWS/IB Gateway host (default from config.IB).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=IbConfig.port,
        help="TWS/IB Gateway port (default from config.IB).",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=IbConfig.client_id,
        help="Client ID to use when connecting (default from config.IB).",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send the order. Without this flag the script only prints what it would do.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    side = Side.BUY if args.side.upper() == "BUY" else Side.SELL
    order_type = OrderType.MARKET if args.order_type.upper() == "MKT" else OrderType.LIMIT

    if order_type is OrderType.LIMIT and args.limit_price is None:
        raise SystemExit("--limit-price is required when --order-type LMT")

    req = OrderRequest(
        symbol=args.symbol,
        side=side,
        quantity=float(args.quantity),
        order_type=order_type,
        limit_price=float(args.limit_price) if args.limit_price is not None else None,
        time_in_force="DAY",
    )

    cfg = IBKRBrokerConfig(host=args.host, port=args.port, client_id=args.client_id)
    broker = IBKRBrokerTws(config=cfg)

    print("[ibkr_test_order] Prepared order:")
    print(f"  Symbol:      {req.symbol}")
    print(f"  Side:        {req.side.value}")
    print(f"  Quantity:    {req.quantity}")
    print(f"  Type:        {req.order_type.value}")
    if req.order_type is OrderType.LIMIT:
        print(f"  Limit price: {req.limit_price}")
    print(f"  TWS host:    {args.host}:{args.port} (clientId={args.client_id})")

    if not args.send:
        print("[ibkr_test_order] Dry run only (no order sent). Use --send to actually place the order.")
        return

    print("[ibkr_test_order] Connecting to TWS/IB Gateway and placing order...")
    broker.connect()
    try:
        order_id = broker.place_order(req)
        print(f"[ibkr_test_order] Placed order, broker order id: {order_id}")

        # Optional: fetch a quick snapshot of positions after placement.
        positions = broker.get_positions()
        print("[ibkr_test_order] Current positions (snapshot):")
        for p in positions:
            print(f"  {p.symbol}: quantity={p.quantity}, avg_price={p.avg_price}")
    finally:
        broker.disconnect()


if __name__ == "__main__":  # pragma: no cover
    main()
