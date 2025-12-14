from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal


class JobType(str, Enum):
    BACKTEST = "backtest"
    TRAIN = "train"
    OPTIMIZE = "optimize"
    WALKFORWARD = "walkforward"
    # Future: PREDICT_CSV = "predict_csv"


JobState = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]


@dataclass
class JobStatus:
    state: JobState
    created_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
