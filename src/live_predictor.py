"""Live (bar-by-bar) LSTM prediction.

This module is intended for live/paper execution loops where we receive one new
OHLC bar at a time (e.g. from IB/TWS) and want to run the trained model once per
bar.

Key points:
- Uses :func:`src.predict.build_prediction_context` to load the latest/best model
  for the configured frequency and tsteps.
- Maintains a rolling buffer large enough for feature engineering warmup
  (e.g. SMA_21 needs 20 extra bars) plus the final ``tsteps`` model window.
- Exposes ``update_and_predict(bar) -> predicted_price``.

The model is trained to predict forward *log returns* on Open prices.
Accordingly, prediction returns a price via:

    predicted_price = Open_t * exp(predicted_log_return)

(plus optional bias correction, when available).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Mapping, Optional, Any

import numpy as np
import pandas as pd

from src.config import FREQUENCY, TSTEPS
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler
from src.predict import PredictionContext, build_prediction_context


@dataclass
class LivePredictorConfig:
    """Configuration for :class:`LivePredictor`.

    warmup_extra:
        Extra bars required beyond ``tsteps`` for feature engineering.
        With the default feature set including SMA_21, we need 20.
    """

    frequency: str = FREQUENCY
    tsteps: int = TSTEPS
    warmup_extra: int = 20
    max_window: int = field(init=False)

    def __post_init__(self) -> None:
        self.max_window = int(self.warmup_extra) + int(self.tsteps)


class LivePredictor:
    """Stateful predictor that runs once per incoming bar."""

    def __init__(self, *, ctx: PredictionContext, config: Optional[LivePredictorConfig] = None) -> None:
        self.ctx = ctx
        self.config = config or LivePredictorConfig(frequency=FREQUENCY, tsteps=ctx.tsteps)

        # Keep raw bars. We store as dict/Series so we can build a DataFrame.
        self._buffer: Deque[dict] = deque(maxlen=self.config.max_window)

    @classmethod
    def from_config(cls, config: Optional[LivePredictorConfig] = None) -> "LivePredictor":
        config = config or LivePredictorConfig()
        ctx = build_prediction_context(frequency=config.frequency, tsteps=config.tsteps)
        return cls(ctx=ctx, config=config)

    def warmup_from_csv(self, csv_path: str, frequency: str) -> int:
        """Pre-populate buffer with last N bars from historical CSV.

        Args:
            csv_path: Path to historical data CSV (e.g. data/processed/nvda_240min.csv)
            frequency: Bar frequency for this CSV (e.g. "240min")

        Returns:
            Number of bars loaded into the buffer.

        This allows the predictor to make predictions immediately on the first
        live bar instead of waiting for max_window bars to accumulate.
        """
        if not os.path.exists(csv_path):
            print(f"[warmup] CSV not found: {csv_path}")
            return 0

        try:
            # Read the CSV - assume columns: DateTime, Open, High, Low, Close
            df = pd.read_csv(csv_path)

            # Convert DateTime to pandas datetime
            if "DateTime" in df.columns:
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df = df.rename(columns={"DateTime": "Time"})
            elif "Time" in df.columns:
                df["Time"] = pd.to_datetime(df["Time"])

            # Take the last max_window bars
            tail = df.tail(self.config.max_window)

            # Convert to buffer format
            loaded = 0
            for _, row in tail.iterrows():
                bar_dict = {
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Time": pd.to_datetime(row["Time"]),
                }
                self._buffer.append(bar_dict)
                loaded += 1

            print(f"[warmup] Loaded {loaded} bars from {csv_path}")
            return loaded

        except Exception as e:
            print(f"[warmup] Failed to load from {csv_path}: {e}")
            return 0

    def _buffer_to_frame(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Time"])
        df = pd.DataFrame(list(self._buffer))
        cols = ["Open", "High", "Low", "Close", "Time"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NaT if c == "Time" else np.nan
        return df[cols]

    def update_and_predict(self, bar: Mapping[str, Any]) -> float:
        """Update rolling buffer with a new bar and return predicted price.

        During warmup (insufficient history for feature engineering + tsteps),
        this returns the latest bar's Close as a neutral placeholder.
        """

        # Normalize incoming bar.
        t = bar.get("Time")
        if t is None:
            t = pd.Timestamp.utcnow()
        t = pd.to_datetime(t)

        row = {
            "Open": float(bar.get("Open")),
            "High": float(bar.get("High")),
            "Low": float(bar.get("Low")),
            "Close": float(bar.get("Close")),
            "Time": t,
        }
        self._buffer.append(row)

        df_raw = self._buffer_to_frame()

        # Need enough raw history to survive add_features(dropna) and still have tsteps.
        if len(df_raw) < self.config.max_window:
            return float(df_raw["Close"].iloc[-1])

        # Feature engineering; add_features drops NA rows due to rolling indicators.
        df_feat = add_features(df_raw.copy(), self.ctx.features_to_use)
        if len(df_feat) < self.ctx.tsteps:
            return float(df_raw["Close"].iloc[-1])

        df_feat_window = df_feat.tail(self.ctx.tsteps)

        feature_cols = [c for c in self.ctx.features_to_use if c != "Time"]
        missing = [c for c in feature_cols if c not in df_feat_window.columns]
        if missing:
            raise ValueError(f"Feature columns missing after engineering: {missing}")

        df_norm = apply_standard_scaler(df_feat_window, feature_cols, self.ctx.scaler_params)
        X = df_norm[feature_cols].to_numpy(dtype=np.float32)[np.newaxis, :, :]  # (1, T, F)

        # Model outputs forward log return on Open.
        preds = self.ctx.model.predict(X, verbose=0)
        log_r = float(np.asarray(preds).reshape(-1)[0])
        log_r += float(getattr(self.ctx, "bias_correction_mean_residual", 0.0) or 0.0)

        base_open = float(df_raw["Open"].iloc[-1])
        predicted_price = base_open * float(np.exp(log_r))
        return float(predicted_price)
