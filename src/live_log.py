"""Append-only logging utilities for live trading sessions.

Goal
- Provide a crash-safe, concurrency-tolerant way for the live runner to append
  events while the Streamlit UI reads them.

Design
- JSONL (newline-delimited JSON) per run.
- Best-effort atomicity: append a single line, flush, and fsync.
- Readers are tolerant: ignore malformed / partial lines.

This module is intentionally dependency-light (no Streamlit imports).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_ui_state_dir() -> Path:
    # repo_root/src/live_log.py -> repo_root
    return Path(__file__).resolve().parents[1] / "ui_state"


def default_live_dir() -> Path:
    return default_ui_state_dir() / "live"


def create_run_id(ts: datetime | None = None) -> str:
    """Create a sortable run id (UTC)."""

    t = ts or datetime.now(timezone.utc)
    return t.strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def make_log_path(
    *,
    symbol: str,
    frequency: str,
    run_id: str,
    live_dir: Path | None = None,
) -> Path:
    d = live_dir or default_live_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"live_{_safe_slug(symbol)}_{_safe_slug(frequency)}_{_safe_slug(run_id)}.jsonl"


def json_friendly(obj: Any) -> Any:
    """Convert common Python objects into JSON-serializable structures.

    This is intentionally small and pragmatic:
    - dataclasses -> dict
    - datetime -> ISO string
    - Path -> string
    - dict/list/tuple -> recursively converted

    Unknown objects are stringified as a last resort.
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            return obj.replace(tzinfo=timezone.utc).isoformat()
        return obj.astimezone(timezone.utc).isoformat()

    if isinstance(obj, Path):
        return str(obj)

    # dataclass instances
    if hasattr(obj, "__dataclass_fields__"):
        try:
            return json_friendly(asdict(obj))
        except Exception:
            return str(obj)

    if isinstance(obj, dict):
        return {str(k): json_friendly(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_friendly(v) for v in obj]

    return str(obj)


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append a single event to a JSONL file.

    Best-effort crash safety:
    - write a single line
    - flush
    - fsync
    """

    if "ts_utc" not in event:
        event = dict(event)
        event["ts_utc"] = _utc_now_iso()

    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(json_friendly(event), ensure_ascii=False)

    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Some environments/filesystems may not support fsync; ignore.
            pass


def read_events(path: Path, *, max_events: int | None = None) -> list[dict[str, Any]]:
    """Read events from a JSONL file.

    Tolerant reader:
    - Skips blank lines.
    - Ignores lines that aren't valid JSON objects.
    """

    if not path.exists():
        return []

    acc: deque[dict[str, Any]]
    if max_events is None:
        acc = deque()
    else:
        acc = deque(maxlen=int(max_events))

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            acc.append(obj)

    return list(acc)


@dataclass(frozen=True)
class LiveLogInfo:
    path: Path
    symbol: str
    frequency: str
    run_id: str


def _parse_log_filename(name: str) -> LiveLogInfo | None:
    # Expected: live_{symbol}_{frequency}_{run_id}.jsonl
    if not name.startswith("live_") or not name.endswith(".jsonl"):
        return None
    stem = name[:-len(".jsonl")]
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    _, symbol, frequency, run_id = parts[0], parts[1], parts[2], "_".join(parts[3:])
    return LiveLogInfo(path=Path(name), symbol=symbol, frequency=frequency, run_id=run_id)


def list_logs(live_dir: Path | None = None) -> list[LiveLogInfo]:
    """List available live log files, most recent first."""

    d = live_dir or default_live_dir()
    if not d.exists():
        return []

    infos: list[LiveLogInfo] = []
    for p in d.glob("live_*.jsonl"):
        meta = _parse_log_filename(p.name)
        if meta is None:
            continue
        infos.append(
            LiveLogInfo(path=p, symbol=meta.symbol, frequency=meta.frequency, run_id=meta.run_id)
        )

    infos.sort(key=lambda x: x.path.stat().st_mtime, reverse=True)
    return infos


def kill_switch_path(live_dir: Path | None = None) -> Path:
    d = live_dir or default_live_dir()
    return d / "KILL_SWITCH"


def is_kill_switch_enabled(live_dir: Path | None = None) -> bool:
    return kill_switch_path(live_dir).exists()


def set_kill_switch(*, enabled: bool, live_dir: Path | None = None) -> None:
    p = kill_switch_path(live_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if enabled:
        p.write_text(_utc_now_iso(), encoding="utf-8")
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def latest_event(events: Iterable[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for e in reversed(list(events)):
        if e.get("type") == event_type:
            return e
    return None
