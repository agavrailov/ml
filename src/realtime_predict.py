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

# This module originally contained an early implementation of bar-by-bar
# prediction. The project now uses :class:`src.live_predictor.LivePredictor`,
# which correctly maps log-return outputs back to price.

from src.live_predictor import LivePredictor as RealtimePredictor
from src.live_predictor import LivePredictorConfig as RealtimePredictorConfig

__all__ = ["RealtimePredictor", "RealtimePredictorConfig"]
