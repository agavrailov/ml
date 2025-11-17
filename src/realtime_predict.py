"""Realtime LSTM prediction helper for Phase 1 Paper Trading.

This module provides a small, stateful helper class that:

- Builds a PredictionContext (model + scaler + feature metadata) once.
- Maintains a rolling window of recent OHLC bars.
- For each new bar, updates the window, runs feature engineering, and
  returns a single-step price prediction.

It is designed to fit into the Phase 1 "Real-Time Strategy Loop" described in
`docs/trading system/trading_system_hld.md`, where the overall flow is:

1. Receive new bar from the Real-Time Data Adapter.
2. Feed it into RealtimePredictor.update_and_predict(...).
3. Feed the prediction + latest features into the strategy and risk engine.
4. Log simulated trades and PnL (paper trading) or forward signals to
   an execution adapter in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Optional
from collections import deque

import numpy as np
import pandas as pd

from src.config import TSTEPS, FREQUENCY
from src.predict import PredictionContext, build_prediction_context
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler


@dataclass
class RealtimePredictorConfig:
    """Configuration for the realtime LSTM predictor.

    Attributes
    ----------
    frequency:
        Resample frequency / bar timeframe (e.g. "60min"). Must match the
        trained model's frequency.
    tsteps:
        Number of timesteps in the LSTM input window. Defaults to TSTEPS
        from src.config.
    warmup_extra:
        Extra bars required for feature engineering beyond ``tsteps``.
        For the current SMA_21-based feature set, this is 20 (21-day window
        minus one).
    max_window:
        Maximum number of recent bars to keep in memory. Should be at least
        ``warmup_extra + tsteps``.
    """

    frequency: str = FREQUENCY
    tsteps: int = TSTEPS
    warmup_extra: int = 20
    max_window: int = field(init=False)

    def __post_init__(self) -> None:
        self.max_window = self.warmup_extra + self.tsteps


class RealtimePredictor:
    """Stateful helper for real-time, bar-by-bar LSTM predictions.

    Usage (conceptual)::

        cfg = RealtimePredictorConfig(frequency="60min")
        predictor = RealtimePredictor.from_config(cfg)

        for new_bar in stream_of_ohlc_bars:
            pred = predictor.update_and_predict(new_bar)
            # pass `pred` + latest state into strategy engine

    Each bar is expected as a mapping or dict with at least:
    - "Open", "High", "Low", "Close"
    - Optional "Time" (timestamp). If provided, it will be preserved and
      used during feature engineering.
    """

    def __init__(
        self,
        ctx: PredictionContext,
        config: Optional[RealtimePredictorConfig] = None,
    ) -> None:
        self.ctx = ctx
        self.config = config or RealtimePredictorConfig(frequency=ctx.features_to_use[0] if ctx.features_to_use else FREQUENCY)

        # Rolling window of raw bars stored as a DataFrame.
        self._buffer: Deque[pd.Series] = deque(maxlen=self.config.max_window)

    @classmethod
    def from_config(cls, config: Optional[RealtimePredictorConfig] = None) -> "RealtimePredictor":
        """Construct a predictor from a config, building a PredictionContext.

        This performs all heavy initialization once at startup.
        """

        config = config or RealtimePredictorConfig()
        ctx = build_prediction_context(frequency=config.frequency, tsteps=config.tsteps)
        return cls(ctx=ctx, config=config)

    def _buffer_to_frame(self) -> pd.DataFrame:
        """Return the current buffer as a DataFrame."""
        if not self._buffer:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Time"])
        df = pd.DataFrame(list(self._buffer))
        # Ensure expected columns exist in a stable order.
        cols = ["Open", "High", "Low", "Close", "Time"]
        for c in cols:
            if c not in df.columns:
                if c == "Time":
                    df[c] = pd.NaT
                else:
                    df[c] = np.nan
        return df[cols]

    def update_and_predict(self, bar: dict) -> float:
        """Add a new bar and return the latest price prediction.

        Parameters
        ----------
        bar:
            Mapping with at least 'Open', 'High', 'Low', 'Close'. It may also
            contain 'Time' (timestamp). Any extra keys are ignored.

        Returns
        -------
        float
            Predicted future price. During warmup (insufficient history), this
            falls back to the bar's Close.
        """

        # Normalize incoming bar into a Series with expected columns.
        expected = ["Open", "High", "Low", "Close", "Time"]
        s = {k: bar.get(k) for k in expected}
        if s["Time"] is not None:
            s["Time"] = pd.to_datetime(s["Time"])
        self._buffer.append(pd.Series(s))

        df_raw = self._buffer_to_frame()

        # Warmup: require enough history for both feature engineering and tsteps.
        if len(df_raw) < self.config.max_window:
            # Not enough data yet; return latest Close as a neutral prediction.
            return float(df_raw["Close"].iloc[-1])

        # Feature engineering on the rolling window.
        df_feat = add_features(df_raw.copy(), self.ctx.features_to_use)

        if len(df_feat) < self.ctx.tsteps:
            return float(df_raw["Close"].iloc[-1])

        # Take last tsteps rows for the current prediction window.
        df_feat_window = df_feat.tail(self.ctx.tsteps)

        feature_cols = [c for c in df_feat_window.columns if c != "Time"]
        df_norm = apply_standard_scaler(df_feat_window, feature_cols, self.ctx.scaler_params)
        X = df_norm[feature_cols].to_numpy(dtype="float32")[np.newaxis, :, :]  # (1, T, F)

        preds_norm = self.ctx.model.predict(X, verbose=0)
        y_norm = float(preds_norm[0, 0])

        # Denormalize using the stored scaler (Open as target).
        y = y_norm * float(self.ctx.std_vals["Open"]) + float(self.ctx.mean_vals["Open"])
        return float(y)
