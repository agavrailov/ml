"""Reconnect tests for src.ibkr_live_session.

This is a pure-unit test that simulates an IB Gateway restart by firing
ib.disconnectedEvent while the event loop is running.

We assert that:
- the live session attempts to reconnect (second connect call)
- it re-subscribes to keepUpToDate bars (second reqHistoricalData call)
- the broker is constructed with the *same* IB instance (single-IB design)
"""

from __future__ import annotations

import time
import types

import pytest

import src.ibkr_live_session as m


class _Event:
    def __init__(self) -> None:
        self._handlers: list = []

    def __iadd__(self, fn):
        self._handlers.append(fn)
        return self

    def __isub__(self, fn):
        try:
            self._handlers.remove(fn)
        except ValueError:
            pass
        return self

    def fire(self, *args, **kwargs) -> None:  # noqa: ANN001
        for h in list(self._handlers):
            h(*args, **kwargs)


class _FakeBars:
    def __init__(self) -> None:
        self.updateEvent = _Event()


class _FakeIB:
    def __init__(self) -> None:
        self.errorEvent = _Event()
        self.disconnectedEvent = _Event()

        self._connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.req_hist_calls = 0

    def connect(self, host: str, port: int, clientId: int) -> None:  # noqa: N803
        self.connect_calls.append((host, int(port), int(clientId)))
        self._connected = True

    def isConnected(self) -> bool:  # noqa: N802
        return bool(self._connected)

    def reqCurrentTime(self):  # noqa: N802
        if not self._connected:
            raise TimeoutError("not connected")
        return 0

    def disconnect(self) -> None:
        self._connected = False

    def qualifyContracts(self, contract) -> None:  # noqa: ANN001
        return None

    def reqHistoricalData(self, *args, **kwargs):  # noqa: ANN001
        self.req_hist_calls += 1
        return _FakeBars()

    def sleep(self, seconds: float) -> None:
        # No-op to keep the unit test fast.
        return None

    def run(self) -> None:
        # Simulate a disconnect while the event loop is running.
        self._connected = False
        self.disconnectedEvent.fire()

        # Wait briefly for reconnect/resubscribe to happen.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if self._connected and self.req_hist_calls >= 2:
                break
            time.sleep(0.01)

        raise KeyboardInterrupt


class _FakeBroker:
    def __init__(self) -> None:
        self.positions_calls = 0

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_positions(self):
        self.positions_calls += 1
        return []

    def get_open_orders(self):
        return []

    def get_all_orders(self):
        return []

    def get_account_summary(self):
        return {}


def test_reconnect_resubscribes_keep_up_to_date_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure live-session import guard doesn't block the run.
    monkeypatch.setattr(m, "_HAVE_IB", True)

    fake_ib = _FakeIB()
    fake_broker = _FakeBroker()

    monkeypatch.setattr(m, "IB", lambda: fake_ib)
    monkeypatch.setattr(m, "Stock", lambda *args, **kwargs: object())

    # Avoid loading any real ML model in this unit test.
    monkeypatch.setattr(
        m,
        "LivePredictor",
        types.SimpleNamespace(from_config=lambda *args, **kwargs: object()),
    )

    # Ensure broker is constructed with the SAME IB instance (single-IB design).
    def _make_broker(_backend, *, ib=None):  # noqa: ANN001
        assert ib is fake_ib
        return fake_broker

    monkeypatch.setattr(m, "make_broker", _make_broker)

    cfg = m.LiveSessionConfig(symbol="NVDA", frequency="60min", backend="IBKR_TWS", log_to_disk=False)

    with pytest.raises(KeyboardInterrupt):
        m.run_live_session(cfg)

    # Initial connect + reconnect.
    assert len(fake_ib.connect_calls) >= 2

    # Initial subscribe + resubscribe.
    assert fake_ib.req_hist_calls >= 2
