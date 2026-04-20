#!/usr/bin/env python3
"""
PostToolUse hook: captures data-pipeline failures into
.claude/state/pipeline_failures.jsonl for review at the next session.

Detects:
  1. pytest test failures (any FAILED line in output)
  2. pred/Open ratio WARNING from _log_pred_ratio_summary in src/backtest.py
  3. ValueError / RuntimeError from guardrail assertions in the pipeline

Called by Claude Code after every Bash tool call.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_stdin() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _get_output(hook: dict) -> str:
    """Best-effort: extract tool output text from the hook payload."""
    # Claude Code puts tool output in tool_response.output or tool_response.content
    resp = hook.get("tool_response") or {}
    for key in ("output", "content", "result", "text"):
        if isinstance(resp.get(key), str):
            return resp[key]
    # Fallback: stringify the whole response
    return str(resp)


def _append_failure(state_dir: Path, record: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    failures_path = state_dir / "pipeline_failures.jsonl"
    with open(failures_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── patterns ─────────────────────────────────────────────────────────────────

_PYTEST_FAILED_RE = re.compile(r"FAILED\s+([\w/\\.::\-]+)", re.MULTILINE)
_PYTEST_SUMMARY_RE = re.compile(r"(\d+) failed", re.IGNORECASE)
_RATIO_WARN_RE = re.compile(
    r"!!\s*WARNING.*?pred/Open ratio.*?(\d+\.\d+).*?outside", re.IGNORECASE
)
_GUARDRAIL_RE = re.compile(
    r"(ValueError|RuntimeError):\s*(.+?)(?:\n|$)", re.MULTILINE
)


def _detect_pytest_failures(command: str, output: str, ts: str) -> list[dict]:
    if "pytest" not in command:
        return []
    failed_tests = _PYTEST_FAILED_RE.findall(output)
    if not failed_tests and not _PYTEST_SUMMARY_RE.search(output):
        return []

    # Grab short summary block (lines starting with FAILED or containing "==")
    lines = output.splitlines()
    summary = "\n".join(
        l for l in lines
        if l.startswith("FAILED") or ("==" in l and ("failed" in l.lower() or "error" in l.lower()))
    )[:1200]

    return [{
        "type": "pytest_failure",
        "timestamp": ts,
        "command": command[:300],
        "failed_tests": failed_tests[:20],
        "summary": summary,
        "resolved": False,
    }]


def _detect_ratio_warnings(command: str, output: str, ts: str) -> list[dict]:
    matches = _RATIO_WARN_RE.findall(output)
    if not matches:
        return []
    warn_lines = [l for l in output.splitlines() if "WARNING" in l and "ratio" in l.lower()]
    return [{
        "type": "pipeline_ratio_warning",
        "timestamp": ts,
        "command": command[:300],
        "ratio_values": matches,
        "warning_lines": warn_lines[:5],
        "resolved": False,
    }]


def _detect_guardrail_errors(command: str, output: str, ts: str) -> list[dict]:
    """Catch ValueError / RuntimeError raised by our guardrails."""
    matches = _GUARDRAIL_RE.findall(output)
    if not matches:
        return []
    results = []
    for exc_type, msg in matches[:3]:
        # Filter to pipeline-relevant errors only
        if any(kw in msg.lower() for kw in (
            "nan", "scaler", "merge", "time", "alignment", "checkpoint",
            "sha256", "ratio", "warmup", "feature", "sidecar",
        )):
            results.append({
                "type": "guardrail_error",
                "timestamp": ts,
                "command": command[:300],
                "exception": exc_type,
                "message": msg[:400],
                "resolved": False,
            })
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    hook = _load_stdin()

    if hook.get("hook_event_name") != "PostToolUse":
        sys.exit(0)
    if hook.get("tool_name") != "Bash":
        sys.exit(0)

    command: str = (hook.get("tool_input") or {}).get("command", "")
    output: str = _get_output(hook)
    cwd = Path(hook.get("cwd", "."))
    state_dir = cwd / ".claude" / "state"
    ts = datetime.now(timezone.utc).isoformat()

    failures: list[dict] = []
    failures += _detect_pytest_failures(command, output, ts)
    failures += _detect_ratio_warnings(command, output, ts)
    failures += _detect_guardrail_errors(command, output, ts)

    for f in failures:
        _append_failure(state_dir, f)

    # Always exit 0 — PostToolUse hooks must not block
    sys.exit(0)


if __name__ == "__main__":
    main()
