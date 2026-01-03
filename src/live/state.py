"""State machine for live trading session lifecycle.

Provides explicit state tracking to replace scattered boolean flags
like trading_enabled, shutdown_requested, etc.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class SystemState(Enum):
    """Possible states for the live trading system."""

    INITIALIZING = "INITIALIZING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    SUBSCRIBED = "SUBSCRIBED"
    TRADING = "TRADING"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    WAITING_FOR_GATEWAY = "WAITING_FOR_GATEWAY"
    FATAL_ERROR = "FATAL_ERROR"
    STOPPED = "STOPPED"


class StateMachine:
    """Manages system state transitions with logging."""

    def __init__(
        self,
        initial_state: SystemState,
        event_logger: Callable[[dict], None],
        run_id: str,
    ):
        """Initialize state machine.

        Args:
            initial_state: Starting state
            event_logger: Function to log state transitions (JSONL)
            run_id: Current run ID for logging
        """
        self._state = initial_state
        self._log = event_logger
        self._run_id = run_id
        self._transition_times: dict[SystemState, float] = {initial_state: time.time()}

    def transition_to(self, new_state: SystemState, *, reason: str | None = None) -> None:
        """Transition to a new state.

        Args:
            new_state: Target state
            reason: Optional reason for transition (for logging)
        """
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        transition_time = time.time()
        self._transition_times[new_state] = transition_time

        # Calculate time spent in previous state
        time_in_previous_state = None
        if old_state in self._transition_times:
            time_in_previous_state = transition_time - self._transition_times[old_state]

        # Log transition
        self._log(
            {
                "type": "state_transition",
                "run_id": self._run_id,
                "from_state": old_state.value,
                "to_state": new_state.value,
                "reason": reason,
                "time_in_previous_state_seconds": time_in_previous_state,
            }
        )

    @property
    def current(self) -> SystemState:
        """Get current state."""
        return self._state

    def can_trade(self) -> bool:
        """Check if system can execute trades in current state."""
        return self._state == SystemState.TRADING

    def is_running(self) -> bool:
        """Check if system is running (any state except STOPPED/FATAL_ERROR)."""
        return self._state not in {SystemState.STOPPED, SystemState.FATAL_ERROR}

    def time_in_current_state(self) -> float:
        """Get time spent in current state (seconds)."""
        if self._state not in self._transition_times:
            return 0.0
        return time.time() - self._transition_times[self._state]
