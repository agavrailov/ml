from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from src.core.contracts import BacktestRequest, OptimizeRequest, TrainRequest, WalkForwardRequest
from src.jobs.handlers import backtest_job
from src.jobs.store import read_json, utc_now_iso, write_request, write_status
from src.jobs.types import JobStatus, JobType


def main() -> None:
    p = argparse.ArgumentParser(description="Run a job and persist artifacts under runs/<job_id>.")
    p.add_argument("--job-id", required=True, type=str)
    p.add_argument("--job-type", required=True, choices=[t.value for t in JobType])
    p.add_argument(
        "--request",
        required=True,
        type=str,
        help="Path to a JSON request payload (will be copied into runs/<job_id>/request.json).",
    )

    args = p.parse_args()

    job_id = str(args.job_id)
    job_type = JobType(str(args.job_type))

    created = utc_now_iso()
    write_status(job_id, JobStatus(state="QUEUED", created_at_utc=created))

    try:
        payload = read_json(Path(args.request))
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")

        write_request(job_id, payload)

        started = utc_now_iso()
        write_status(
            job_id,
            JobStatus(state="RUNNING", created_at_utc=created, started_at_utc=started),
        )

        if job_type == JobType.BACKTEST:
            req = BacktestRequest.from_dict(payload)
            backtest_job.run(job_id, req)
        elif job_type == JobType.TRAIN:
            # Lazy import: training pulls in heavy ML deps.
            from src.jobs.handlers import train_job as _train_job

            req = TrainRequest.from_dict(payload)
            _train_job.run(job_id, req)
        elif job_type == JobType.OPTIMIZE:
            from src.jobs.handlers import optimize_job as _optimize_job

            req = OptimizeRequest.from_dict(payload)
            _optimize_job.run(job_id, req)
        elif job_type == JobType.WALKFORWARD:
            from src.jobs.handlers import walkforward_job as _walkforward_job

            req = WalkForwardRequest.from_dict(payload)
            _walkforward_job.run(job_id, req)
        else:
            raise ValueError(f"Unsupported job_type: {job_type}")

        write_status(
            job_id,
            JobStatus(
                state="SUCCEEDED",
                created_at_utc=created,
                started_at_utc=started,
                finished_at_utc=utc_now_iso(),
            ),
        )

    except Exception as exc:  # noqa: BLE001
        write_status(
            job_id,
            JobStatus(
                state="FAILED",
                created_at_utc=created,
                started_at_utc=locals().get("started"),
                finished_at_utc=utc_now_iso(),
                error=repr(exc),
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
