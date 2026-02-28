"""Alert management for live trading sessions.

Provides deduplicat alert tracking with severity levels.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertManager:
    """Manages alerts with deduplication."""

    def __init__(self, alerts_file_path: Path, *, clear_older_than_hours: int = 24):
        """Initialize alert manager.

        Args:
            alerts_file_path: Path to alerts.jsonl file
            clear_older_than_hours: Clear alerts older than this many hours on init
        """
        self._path = alerts_file_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._active_alerts: dict[str, dict] = {}  # key -> alert
        
        # Clear stale alerts on startup
        if clear_older_than_hours > 0:
            self._clear_stale_alerts(hours=clear_older_than_hours)

    def add_alert(
        self,
        *,
        key: str,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
    ) -> None:
        """Add or update an alert.

        Args:
            key: Unique key for deduplication
            severity: Alert severity level
            alert_type: Type/category of alert
            message: Human-readable message
        """
        # Deduplicate: same key = already active
        if key in self._active_alerts:
            return

        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "severity": severity.value,
            "type": alert_type,
            "msg": message,
            "key": key,
        }
        self._active_alerts[key] = alert
        self._append_to_file(alert)

    def clear_alert(self, key: str) -> None:
        """Clear an active alert.

        Args:
            key: Alert key to clear
        """
        if key in self._active_alerts:
            del self._active_alerts[key]

    def count(self) -> int:
        """Get count of active alerts."""
        return len(self._active_alerts)

    def get_critical(self) -> list[dict]:
        """Get list of critical alerts."""
        return [a for a in self._active_alerts.values() if a["severity"] == "CRITICAL"]

    def _append_to_file(self, alert: dict) -> None:
        """Append alert to JSONL file (best-effort).

        Args:
            alert: Alert dictionary to append
        """
        try:
            line = json.dumps(alert, ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
        except Exception:
            # Never crash trading loop on alert write failure
            pass

    def _clear_stale_alerts(self, hours: int) -> None:
        """Remove alerts older than specified hours from file.

        Args:
            hours: Clear alerts older than this many hours
        """
        if not self._path.exists():
            return

        try:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Read all alerts
            alerts = []
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        alert = json.loads(line)
                        alert_time = datetime.fromisoformat(alert["ts"])
                        # Keep alerts newer than cutoff
                        if alert_time >= cutoff:
                            alerts.append(alert)
                    except Exception:
                        continue

            # Rewrite file with only recent alerts
            with self._path.open("w", encoding="utf-8") as f:
                for alert in alerts:
                    f.write(json.dumps(alert, ensure_ascii=False))
                    f.write("\n")
                f.flush()
        except Exception:
            # Never crash on cleanup
            pass
