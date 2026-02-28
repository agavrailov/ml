"""CLI status checker for live trading sessions.

Provides quick health check without opening Streamlit UI.

Usage:
    python -m scripts.status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Print live session status from status.json."""
    # Assume repo root is parent of scripts/
    repo_root = Path(__file__).resolve().parents[1]
    status_path = repo_root / "ui_state" / "live" / "status.json"

    if not status_path.exists():
        print("❌ No live session running (status.json not found)")
        print(f"   Expected at: {status_path}")
        sys.exit(1)

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to read status.json: {e}")
        sys.exit(1)

    # Extract values
    state = status.get("state", "UNKNOWN")
    conn = status.get("connection", {})
    data_feed = status.get("data_feed", {})
    position = status.get("position", {})
    alerts = status.get("alerts", {})
    kill_switch = status.get("kill_switch", False)
    updated_at = status.get("updated_at", "unknown")

    # Determine health
    is_healthy = (
        state == "TRADING"
        and not kill_switch
        and alerts.get("count", 0) == 0
        and data_feed.get("age_minutes") is not None
        and data_feed.get("age_minutes", 999) < 300  # < 5 hours
    )

    health_icon = "✅" if is_healthy else "⚠️"

    # Print status
    print(f"\n{health_icon} Live Trading Status")
    print("=" * 50)
    print(f"System State:     {state}")
    print(f"Connection:       {conn.get('status', 'UNKNOWN')} ", end="")
    if conn.get("uptime_minutes") is not None:
        print(f"(uptime: {conn['uptime_minutes']:.0f}m)")
    else:
        print()

    last_bar_time = data_feed.get("last_bar_time")
    if last_bar_time:
        age_min = data_feed.get("age_minutes", 0)
        print(f"Last Bar:         {last_bar_time[:19]} ({age_min:.0f} min ago)")
    else:
        print("Last Bar:         (none)")

    print(f"Position:         {position.get('status', 'UNKNOWN')}", end="")
    qty = position.get("quantity")
    if qty:
        print(f" (qty: {qty})")
    else:
        print()

    alert_count = alerts.get("count", 0)
    if alert_count > 0:
        print(f"Alerts:           🔴 {alert_count} ACTIVE")
    else:
        print(f"Alerts:           {alert_count}")

    kill_switch_status = "🔴 ENABLED" if kill_switch else "OFF"
    print(f"Kill Switch:      {kill_switch_status}")

    print(f"\nUpdated:          {updated_at[:19]}")
    print("=" * 50)

    if is_healthy:
        print("\nStatus: HEALTHY ✓")
        sys.exit(0)
    else:
        print("\nStatus: CHECK REQUIRED ⚠️")
        if kill_switch:
            print("  - Kill switch is enabled")
        if alert_count > 0:
            print(f"  - {alert_count} active alerts")
        if state != "TRADING":
            print(f"  - System not trading (state: {state})")
        sys.exit(1)


if __name__ == "__main__":
    main()
