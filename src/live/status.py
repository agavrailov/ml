"""Status file management for live trading sessions.

Provides at-a-glance status without opening UI, via status.json file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StatusManager:
    """Manages status.json file for live session status."""

    def __init__(self, status_file_path: Path):
        """Initialize status manager.

        Args:
            status_file_path: Path to status.json file
        """
        self._path = status_file_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        *,
        state: str,
        connection_info: dict[str, Any],
        last_bar_info: dict[str, Any],
        position_info: dict[str, Any],
        alert_count: int,
        kill_switch_enabled: bool,
    ) -> None:
        """Update status file with current system state.

        Args:
            state: Current system state (e.g., "TRADING", "RECONNECTING")
            connection_info: Connection details (uptime_minutes, last_reconnect)
            last_bar_info: Last bar details (time, age_minutes, expected_next)
            position_info: Position details (status, quantity)
            alert_count: Number of active alerts
            kill_switch_enabled: Whether kill switch is active
        """
        status = {
            "state": state,
            "connection": {
                "status": "CONNECTED" if state == "TRADING" else "DISCONNECTED",
                "uptime_minutes": connection_info.get("uptime_minutes"),
                "last_reconnect": connection_info.get("last_reconnect"),
            },
            "data_feed": {
                "last_bar_time": last_bar_info.get("time"),
                "age_minutes": last_bar_info.get("age_minutes"),
                "expected_next": last_bar_info.get("expected_next"),
            },
            "position": position_info,
            "alerts": {"count": alert_count},
            "kill_switch": kill_switch_enabled,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._write_atomic(status)

    def _write_atomic(self, status: dict[str, Any]) -> None:
        """Write status to file atomically (best-effort).

        Uses write-to-temp-then-rename pattern for atomicity.
        """
        try:
            # Write to temporary file
            temp_path = self._path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

            # Atomic rename (on most filesystems)
            if os.name == "nt":  # Windows
                # Windows requires removing target first
                if self._path.exists():
                    self._path.unlink()
            temp_path.replace(self._path)
        except Exception:
            # Never crash trading loop on status write failure
            pass
