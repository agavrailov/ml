"""Graceful shutdown tests for src.ibkr_live_session.

The real live session uses ib_insync + a broker backend. In practice this often
creates *two* IB connections (one for data, one inside the broker). On Ctrl+C we
must disconnect both to release clientIds.

This test is pure-unit: we monkeypatch IB/Stock/make_broker/LivePredictor so we
can simulate Ctrl+C (KeyboardInterrupt) without any real TWS/IB Gateway.
"""

from __future__ import annotations

import types

import pytest

import src.ibkr_live_session as m


class _Event:
    def __init__(self) -> None:
        self._handlers = []

    def __iadd__(self, fn):
        self._handlers.append(fn)
        return self

    def __isub__(self, fn):
        try:
            self._handlers.remove(fn)
        except ValueError:
            pass
        return self


class _FakeBars:
    def __init__(self) -> None:
        self.updateEvent = _Event()


class _FakeIB:
    def __init__(self) -> None:
        self.errorEvent = _Event()
        self._connected = False
        self.disconnected = False

    def connect(self, host: str, port: int, clientId: int) -> None:  # noqa: N803
        self._connected = True

    def isConnected(self) -> bool:  # noqa: N802
        return bool(self._connected)

    def reqCurrentTime(self):  # noqa: N802
        if not self._connected:
            raise TimeoutError("not connected")
        return 0

    def disconnect(self) -> None:
        self.disconnected = True
        self._connected = False

    def qualifyContracts(self, contract) -> None:  # noqa: ANN001
        return None

    def reqHistoricalData(self, *args, **kwargs):  # noqa: ANN001
        return _FakeBars()

    def run(self) -> None:
        # Simulate operator pressing Ctrl+C.
        raise KeyboardInterrupt

    def sleep(self, seconds: float) -> None:
        return None


class _FakeBroker:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    # Snapshot helpers used by the live session (best-effort).
    def get_positions(self):
        return []

    def get_open_orders(self):
        return []

    def get_all_orders(self):
        return []

    def get_account_summary(self):
        return {}


def test_ctrl_c_disconnects_broker_and_ib(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure live-session import guard doesn't block the run.
    monkeypatch.setattr(m, "_HAVE_IB", True)

    fake_ib = _FakeIB()
    fake_broker = _FakeBroker()

    # Patch IB/Stock to avoid any ib_insync dependency.
    monkeypatch.setattr(m, "IB", lambda: fake_ib)
    monkeypatch.setattr(m, "Stock", lambda *args, **kwargs: object())

    # Avoid loading any real ML model in this unit test.
    monkeypatch.setattr(
        m,
        "LivePredictor",
        types.SimpleNamespace(from_config=lambda *args, **kwargs: object()),
    )

    # Ensure we create a broker that can be disconnected.
    monkeypatch.setattr(m, "make_broker", lambda *args, **kwargs: fake_broker)

    cfg = m.LiveSessionConfig(symbol="NVDA", frequency="60min", backend="SIM", log_to_disk=False)

    with pytest.raises(KeyboardInterrupt):
        m.run_live_session(cfg)

    assert fake_broker.connected is True
    assert fake_broker.disconnected is True
    assert fake_ib.disconnected is True
