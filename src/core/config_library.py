"""File-based config library (Option 1).

This implements a minimal "config candidates" store for strategy parameters:
- Each candidate is stored as a JSON file under configs/library/{SYMBOL}/{FREQUENCY}/{id}.json
- A manifest (configs/library/manifest.json) indexes candidates for fast UI listing.
- Promotion copies a chosen candidate into configs/active.json (the production config).

Design goals:
- Keep runtime simple: production loads a single active.json.
- Keep UI-friendly: list/filter candidates and promote from a table.
- Be resilient: if the manifest is missing/corrupt, fall back to scanning files.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _get_repo_root() -> Path:
    # src/core/config_library.py -> src/core -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _configs_dir() -> Path:
    return _get_repo_root() / "configs"


def _active_config_path() -> Path:
    return _configs_dir() / "active.json"


def _library_root() -> Path:
    return _configs_dir() / "library"


def _manifest_path() -> Path:
    return _library_root() / "manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write JSON atomically (best-effort) via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _read_json_list(path: Path) -> list[dict[str, Any]] | None:
    # Missing manifest should behave like an empty manifest.
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class CandidateKey:
    symbol: str
    frequency: str

    @classmethod
    def normalize(cls, *, symbol: str, frequency: str) -> "CandidateKey":
        sym = (symbol or "").strip().upper()
        freq = (frequency or "").strip()
        if not sym:
            raise ValueError("symbol must be non-empty")
        if not freq:
            raise ValueError("frequency must be non-empty")
        return cls(symbol=sym, frequency=freq)


def _candidate_dir(key: CandidateKey) -> Path:
    return _library_root() / key.symbol / key.frequency


def _candidate_path(key: CandidateKey, candidate_id: str) -> Path:
    return _candidate_dir(key) / f"{candidate_id}.json"


def _make_candidate_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"cfg_{stamp}_{short}"


def _summarize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}

    # Keep the manifest small and UI-friendly; store a few common KPIs.
    out: dict[str, Any] = {}
    for k in [
        "sharpe_ratio",
        "total_return",
        "cagr",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "n_trades",
        "final_equity",
    ]:
        if k in metrics:
            out[k] = metrics.get(k)
    return out


def _extract_strategy_params(strategy: dict[str, Any]) -> dict[str, float]:
    """Extract the standard strategy params this project supports."""
    keys = [
        "risk_per_trade_pct",
        "reward_risk_ratio",
        "k_sigma_long",
        "k_sigma_short",
        "k_atr_long",
        "k_atr_short",
    ]
    out: dict[str, float] = {}
    for k in keys:
        if k in strategy:
            out[k] = float(strategy[k])
    return out


def save_candidate(
    *,
    symbol: str,
    frequency: str,
    strategy: dict[str, Any],
    label: str | None = None,
    source: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a config candidate and register it in the manifest.

    Returns the manifest row.
    """
    key = CandidateKey.normalize(symbol=symbol, frequency=frequency)

    candidate_id = _make_candidate_id()
    created_at_utc = _utc_now_iso()

    payload = {
        "meta": {
            "id": candidate_id,
            "symbol": key.symbol,
            "frequency": key.frequency,
            "created_at_utc": created_at_utc,
            "label": (label or "").strip() or None,
            "source": (source or "").strip() or None,
        },
        "strategy": _extract_strategy_params(strategy),
    }
    if metrics is not None:
        payload["metrics"] = metrics

    path = _candidate_path(key, candidate_id)
    _write_json_atomic(path, payload)

    rel_path = path.relative_to(_get_repo_root()).as_posix()
    row: dict[str, Any] = {
        "id": candidate_id,
        "symbol": key.symbol,
        "frequency": key.frequency,
        "created_at_utc": created_at_utc,
        "label": payload["meta"]["label"],
        "source": payload["meta"]["source"],
        "path": rel_path,
        **payload["strategy"],
        **_summarize_metrics(metrics),
    }

    # Update manifest.
    manifest = _read_json_list(_manifest_path())
    if manifest is None:
        # Corrupt manifest: rebuild from scratch (best-effort) then append.
        manifest = _scan_manifest_rows()
    manifest.append(row)

    # Sort newest first.
    manifest = sorted(manifest, key=lambda r: r.get("created_at_utc", ""), reverse=True)
    _write_json_atomic(_manifest_path(), manifest)

    return row


def _scan_candidate_files() -> Iterable[Path]:
    root = _library_root()
    if not root.exists() or not root.is_dir():
        return []

    files: list[Path] = []
    for p in root.rglob("*.json"):
        # Skip manifest.json itself.
        if p.name == "manifest.json":
            continue
        files.append(p)
    return files


def _scan_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in _scan_candidate_files():
        data = _read_json_dict(p)
        if not data:
            continue
        meta = data.get("meta")
        strategy = data.get("strategy")
        if not isinstance(meta, dict) or not isinstance(strategy, dict):
            continue

        cid = meta.get("id")
        symbol = meta.get("symbol")
        frequency = meta.get("frequency")
        created_at_utc = meta.get("created_at_utc")
        if not cid or not symbol or not frequency or not created_at_utc:
            continue

        rel_path = p.relative_to(_get_repo_root()).as_posix()

        row: dict[str, Any] = {
            "id": str(cid),
            "symbol": str(symbol),
            "frequency": str(frequency),
            "created_at_utc": str(created_at_utc),
            "label": meta.get("label"),
            "source": meta.get("source"),
            "path": rel_path,
            **_extract_strategy_params(strategy),
        }
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            row.update(_summarize_metrics(metrics))

        rows.append(row)

    return sorted(rows, key=lambda r: r.get("created_at_utc", ""), reverse=True)


def list_candidates(
    *,
    symbol: str | None = None,
    frequency: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List candidates for table display.

    Filters are exact-match (after normalization for symbol).
    """
    manifest = _read_json_list(_manifest_path())
    if manifest is None:
        manifest = _scan_manifest_rows()

    rows = manifest

    if symbol is not None:
        sym = symbol.strip().upper()
        rows = [r for r in rows if str(r.get("symbol", "")).upper() == sym]

    if frequency is not None:
        freq = frequency.strip()
        rows = [r for r in rows if str(r.get("frequency", "")) == freq]

    if limit is not None:
        rows = rows[: max(int(limit), 0)]

    return rows


def load_candidate(candidate_id: str) -> dict[str, Any] | None:
    """Load the full candidate JSON payload."""
    if not candidate_id:
        return None

    manifest = _read_json_list(_manifest_path())
    path_str = None
    if isinstance(manifest, list):
        for row in manifest:
            if row.get("id") == candidate_id:
                path_str = row.get("path")
                break

    if path_str:
        p = _get_repo_root() / Path(path_str)
        data = _read_json_dict(p)
        if data is not None:
            return data

    # Fallback: scan.
    for p in _scan_candidate_files():
        data = _read_json_dict(p)
        if data and isinstance(data.get("meta"), dict) and data["meta"].get("id") == candidate_id:
            return data

    return None


def promote_candidate(candidate_id: str) -> None:
    """Promote a candidate to production by writing configs/active.json."""
    data = load_candidate(candidate_id)
    if data is None:
        raise FileNotFoundError(f"Candidate not found: {candidate_id}")

    # Keep active.json schema stable: include meta + strategy (+metrics optionally).
    active_payload: dict[str, Any] = {
        "meta": {
            **(data.get("meta") if isinstance(data.get("meta"), dict) else {}),
            "promoted_at_utc": _utc_now_iso(),
        },
        "strategy": data.get("strategy", {}),
    }
    if "metrics" in data:
        active_payload["metrics"] = data.get("metrics")

    _write_json_atomic(_active_config_path(), active_payload)


def read_active_config() -> dict[str, Any] | None:
    """Read configs/active.json (best-effort)."""
    return _read_json_dict(_active_config_path())
