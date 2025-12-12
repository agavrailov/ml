"""Unit tests for IB error logging in src.ibkr_live_session.

These tests are pure-unit (no live IBKR connectivity).
"""

from __future__ import annotations

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

    def emit(self, *args, **kwargs):  # noqa: ANN001
        for h in list(self._handlers):
            h(*args, **kwargs)


class _FakeIB:
    def __init__(self) -> None:
        self.errorEvent = _Event()


def test_attach_ib_error_logging_emits_event() -> None:
    ib = _FakeIB()
    events: list[dict] = []

    detach = m._attach_ib_error_logging(
        ib,
        log_fn=lambda e: events.append(dict(e)),
        run_id="RID",
        symbol="NVDA",
        frequency="5min",
    )

    ib.errorEvent.emit(-1, 162, "Historical Market Data Service error message", None)

    assert len(events) == 1
    assert events[0]["type"] == "ib_error"
    assert events[0]["run_id"] == "RID"
    assert events[0]["symbol"] == "NVDA"
    assert events[0]["frequency"] == "5min"
    assert events[0]["error_code"] == 162

    detach()


def test_detach_ib_error_logging_stops_emitting() -> None:
    ib = _FakeIB()
    events: list[dict] = []

    detach = m._attach_ib_error_logging(
        ib,
        log_fn=lambda e: events.append(dict(e)),
        run_id="RID",
        symbol="NVDA",
        frequency="5min",
    )

    detach()
    ib.errorEvent.emit(-1, 162, "Historical Market Data Service error message", None)

    assert events == []
