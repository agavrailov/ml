"""Tests for clientId selection/retry in src.ibkr_live_session.

These tests are pure-unit (no live IBKR connectivity).
"""

from __future__ import annotations

import pytest

from src.ibkr_live_session import _connect_with_unique_client_id


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
    def __init__(
        self,
        fail_ids: set[int],
        *,
        probe_fail_ids: set[int] | None = None,
        async_326_ids: set[int] | None = None,
    ) -> None:
        self.fail_ids = set(fail_ids)
        self.probe_fail_ids = set(probe_fail_ids or set())
        self.async_326_ids = set(async_326_ids or set())
        self.connected = None
        self.calls: list[int] = []
        self.errorEvent = _Event()

    def connect(self, host: str, port: int, clientId: int) -> None:  # noqa: N803
        self.calls.append(int(clientId))
        if int(clientId) in self.fail_ids:
            raise TimeoutError("Unable to connect as the client id is already in use")
        self.connected = (host, port, int(clientId))

    def sleep(self, seconds: float) -> None:
        # Emit a 326 error after connect for specified clientIds.
        if self.connected is None:
            return
        cid = int(self.connected[2])
        if cid in self.async_326_ids:
            # Signature mirrors ib_insync: (reqId, errorCode, errorString, contract)
            self.errorEvent.emit(-1, 326, "client id is already in use", None)
            self.connected = None

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected is not None

    def disconnect(self) -> None:
        self.connected = None

    def reqCurrentTime(self):  # noqa: N802
        # Simulate a connection that "succeeds" but is immediately unusable.
        if self.connected is None:
            raise TimeoutError("not connected")
        cid = int(self.connected[2])
        if cid in self.probe_fail_ids:
            raise TimeoutError("API connection failed")
        return 0


def test_connect_retries_on_timeout_and_picks_next_id() -> None:
    ib = _FakeIB(fail_ids={1, 2})

    cid = _connect_with_unique_client_id(
        ib,
        host="127.0.0.1",
        port=7496,
        preferred_client_id=1,
        max_tries=10,
    )

    assert cid == 3
    assert ib.calls[:3] == [1, 2, 3]
    assert ib.connected == ("127.0.0.1", 7496, 3)


def test_connect_raises_after_exhausting_tries() -> None:
    ib = _FakeIB(fail_ids={1, 2, 3})

    with pytest.raises(RuntimeError) as excinfo:
        _connect_with_unique_client_id(
            ib,
            host="127.0.0.1",
            port=7496,
            preferred_client_id=1,
            max_tries=3,
        )

    assert "tried 1..3" in str(excinfo.value)


def test_connect_retries_when_probe_fails_after_connect() -> None:
    # clientId=1 appears to connect but fails the post-connect probe; we should retry.
    ib = _FakeIB(fail_ids=set(), probe_fail_ids={1})

    cid = _connect_with_unique_client_id(
        ib,
        host="127.0.0.1",
        port=7496,
        preferred_client_id=1,
        max_tries=5,
    )

    assert cid == 2
    assert ib.calls[:2] == [1, 2]
    assert ib.connected == ("127.0.0.1", 7496, 2)


def test_connect_retries_when_async_326_emitted_after_connect() -> None:
    # This models the real-world case: connect() returns, then error 326 arrives.
    ib = _FakeIB(fail_ids=set(), async_326_ids={1})

    cid = _connect_with_unique_client_id(
        ib,
        host="127.0.0.1",
        port=7496,
        preferred_client_id=1,
        max_tries=5,
    )

    assert cid == 2
    assert ib.calls[:2] == [1, 2]
    assert ib.connected == ("127.0.0.1", 7496, 2)
