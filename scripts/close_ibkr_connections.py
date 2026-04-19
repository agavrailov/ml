"""Close all stale IBKR/TWS client connections.

Iterates through a range of client IDs, connects briefly, and disconnects
cleanly to release any slots that were leaked by prior failed sessions.
"""
import time
import argparse
from ib_insync import IB
from src.config import TWS_HOST, TWS_PORT

DEFAULT_CLIENT_IDS = list(range(1, 35))
CONNECT_TIMEOUT = 4


def close_client(host: str, port: int, client_id: int) -> bool:
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=CONNECT_TIMEOUT, readonly=True)
        if ib.isConnected():
            ib.disconnect()
            print(f"  client {client_id:3d}: connected and closed")
            return True
    except Exception as e:
        msg = str(e)
        if "already in use" in msg or "326" in msg:
            print(f"  client {client_id:3d}: in use by another process (skipped)")
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            print(f"  client {client_id:3d}: no stale connection (timeout)")
        else:
            print(f"  client {client_id:3d}: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=TWS_HOST)
    p.add_argument("--port", type=int, default=TWS_PORT)
    p.add_argument("--ids", nargs="+", type=int, default=DEFAULT_CLIENT_IDS,
                   help="Client IDs to sweep (default: 1-34)")
    args = p.parse_args()

    print(f"Sweeping client IDs {args.ids[0]}-{args.ids[-1]} on {args.host}:{args.port}")
    closed = 0
    for cid in args.ids:
        if close_client(args.host, args.port, cid):
            closed += 1
        time.sleep(0.3)

    print(f"\nDone. Closed {closed} stale connection(s).")


if __name__ == "__main__":
    main()
