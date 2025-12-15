from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.jobs.types import JobStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    # repo_root/src/jobs/store.py -> repo_root
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return repo_root() / "runs"


def run_dir(job_id: str) -> Path:
    return runs_root() / str(job_id)


def request_path(job_id: str) -> Path:
    return run_dir(job_id) / "request.json"


def status_path(job_id: str) -> Path:
    return run_dir(job_id) / "status.json"


def result_path(job_id: str) -> Path:
    return run_dir(job_id) / "result.json"


def artifacts_dir(job_id: str) -> Path:
    return run_dir(job_id) / "artifacts"


def ensure_run_dir(job_id: str) -> Path:
    d = run_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    artifacts_dir(job_id).mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8") or "null")


def write_status(job_id: str, status: JobStatus) -> None:
    ensure_run_dir(job_id)
    write_json(status_path(job_id), status.to_dict())


def read_status(job_id: str) -> JobStatus | None:
    p = status_path(job_id)
    if not p.exists():
        return None
    data = read_json(p)
    if not isinstance(data, dict):
        return None
    try:
        return JobStatus(**data)
    except Exception:
        return None


def write_request(job_id: str, request_obj: Any) -> None:
    ensure_run_dir(job_id)
    write_json(request_path(job_id), request_obj)


def write_result(job_id: str, result_obj: Any) -> None:
    ensure_run_dir(job_id)
    write_json(result_path(job_id), result_obj)


def write_text_artifact(job_id: str, rel_name: str, content: str) -> str:
    """Write a small text artifact under artifacts/ and return its path as string."""
    d = artifacts_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / rel_name
    p.write_text(content, encoding="utf-8")
    return str(p)


def write_df_csv_artifact(job_id: str, rel_name: str, df) -> str:  # noqa: ANN001
    d = artifacts_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / rel_name
    df.to_csv(p, index=False)
    return str(p)


def copy_file_artifact(job_id: str, src_path: str | Path, *, dest_name: str | None = None) -> str:
    """Copy an existing file into artifacts/ and return the destination path as string."""
    import shutil

    d = artifacts_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    src = Path(src_path)
    name = dest_name or src.name
    dest = d / name

    shutil.copy2(src, dest)
    return str(dest)


def update_progress(job_id: str, progress: float, message: str | None = None) -> None:
    """Update job progress without changing its state.
    
    Args:
        job_id: Job identifier
        progress: Progress value from 0.0 to 1.0
        message: Optional descriptive message
    """
    status = read_status(job_id)
    if status and status.state == "RUNNING":
        status.progress = max(0.0, min(1.0, progress))
        status.progress_message = message
        write_status(job_id, status)
