#!/usr/bin/env python3
"""
SessionStart hook: reads recent unresolved pipeline failures from
.claude/state/pipeline_failures.jsonl and prints a formatted reminder
to stdout.  Claude Code includes stdout from SessionStart hooks in the
session context, so Claude sees this at the start of every conversation.

Only surfaces failures from the last 14 days to avoid staleness.
Exits silently if there are no recent failures.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


MAX_AGE_DAYS = 14
MAX_FAILURES_SHOWN = 8  # cap to avoid flooding the context


def _load_failures(failures_path: Path) -> list[dict]:
    if not failures_path.exists():
        return []
    records = []
    try:
        with open(failures_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return records


def _is_recent(record: dict, cutoff: datetime) -> bool:
    ts_str = record.get("timestamp", "")
    if not ts_str:
        return True  # include if no timestamp
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= cutoff
    except ValueError:
        return True


def _format_record(r: dict, idx: int) -> str:
    rtype = r.get("type", "unknown")
    ts = r.get("timestamp", "?")[:19]  # trim to seconds
    lines = [f"  [{idx + 1}] {rtype}  ({ts} UTC)"]

    if rtype == "pytest_failure":
        failed = r.get("failed_tests", [])
        if failed:
            lines.append(f"      Tests: {', '.join(failed[:5])}")
        summary = r.get("summary", "")
        if summary:
            lines.append(f"      {summary[:200]}")

    elif rtype == "pipeline_ratio_warning":
        for wl in r.get("warning_lines", [])[:2]:
            lines.append(f"      {wl[:200]}")

    elif rtype == "guardrail_error":
        lines.append(f"      {r.get('exception', '')}: {r.get('message', '')[:200]}")

    cmd = r.get("command", "")
    if cmd:
        lines.append(f"      cmd: {cmd[:120]}")

    return "\n".join(lines)


def main() -> None:
    hook = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    cwd = Path(hook.get("cwd", "."))
    failures_path = cwd / ".claude" / "state" / "pipeline_failures.jsonl"

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    all_records = _load_failures(failures_path)
    recent = [r for r in all_records if _is_recent(r, cutoff)]

    # Deduplicate by (type, summary/message) to avoid spamming identical entries
    seen: set[str] = set()
    unique: list[dict] = []
    for r in reversed(recent):  # newest first
        key = f"{r.get('type')}|{r.get('summary', r.get('message', r.get('warning_lines', '')))}"
        key = key[:120]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique = unique[:MAX_FAILURES_SHOWN]

    if not unique:
        sys.exit(0)

    sep = "-" * 51
    lines = [
        "",
        sep,
        f"  !! {len(unique)} unresolved pipeline failure(s) detected",
        sep,
        "",
    ]
    for i, r in enumerate(unique):
        lines.append(_format_record(r, i))
        lines.append("")

    lines += [
        "  Before proceeding, consider:",
        "  1. Fix the root cause.",
        "  2. Update docs/debugging-heuristics.md with the new pattern.",
        "  3. Write a regression test that would have caught it.",
        "  4. Run pytest to confirm green.",
        sep,
        "",
    ]
    sys.stdout.buffer.write("\n".join(lines).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
