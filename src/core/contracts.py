from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8") or "null")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dump(obj), encoding="utf-8")


@dataclass(frozen=True)
class TrainRequest:
    frequency: str
    tsteps: int
    lstm_units: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    n_lstm_layers: int | None = None
    stateful: bool | None = None
    features_to_use: list[str] | None = None
    train_start_date: str | None = None
    train_end_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainRequest":
        return cls(**d)


@dataclass(frozen=True)
class TrainResult:
    validation_loss: float
    model_filename: str
    bias_correction_filename: str | None = None
    metrics_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestRequest:
    frequency: str
    prediction_mode: Literal["csv", "model", "naive"] = "csv"
    start_date: str | None = None
    end_date: str | None = None
    predictions_csv: str | None = None

    # Strategy parameter overrides
    risk_per_trade_pct: float | None = None
    reward_risk_ratio: float | None = None
    k_sigma_long: float | None = None
    k_sigma_short: float | None = None
    k_atr_long: float | None = None
    k_atr_short: float | None = None
    enable_longs: bool | None = None
    allow_shorts: bool | None = None

    initial_equity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestRequest":
        return cls(**d)


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, Any]
    equity_csv: str
    trades_csv: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizeRequest:
    frequency: str
    start_date: str | None = None
    end_date: str | None = None
    prediction_mode: Literal["csv", "model"] = "csv"
    predictions_csv: str | None = None
    trade_side: Literal["Long only", "Short only", "Long & short"] = "Long only"

    # Each param maps to {start, stop, step} or a fixed value.
    param_grid: dict[str, dict[str, float] | float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizeRequest":
        return cls(**d)


@dataclass(frozen=True)
class OptimizeResult:
    results_csv: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardRequest:
    frequency: str
    symbol: str = "nvda"
    t_start: str | None = None
    t_end: str | None = None
    test_span_months: int = 3
    train_lookback_months: int = 24
    min_lookback_months: int = 18
    first_test_start: str | None = None

    predictions_csv: str | None = None
    parameter_sets: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WalkForwardRequest":
        return cls(**d)


@dataclass(frozen=True)
class WalkForwardResult:
    results_csv: str
    summary_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JsonFile:
    """Small helper for reading/writing JSON payloads.

    Kept here so jobs/UI can share the same conventions without importing
    Streamlit or any heavy dependencies.
    """

    path: str

    def read(self) -> Any:
        return _read_json(Path(self.path))

    def write(self, obj: Any) -> None:
        _write_json(Path(self.path), obj)
