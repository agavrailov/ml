from __future__ import annotations

from pathlib import Path

from src.live_log import (
    append_event,
    is_kill_switch_enabled,
    make_log_path,
    read_events,
    set_kill_switch,
)


def test_append_and_read_events_round_trip(tmp_path: Path) -> None:
    log_path = make_log_path(symbol="NVDA", frequency="60min", run_id="test", live_dir=tmp_path)

    append_event(log_path, {"type": "run_start", "run_id": "test"})
    append_event(log_path, {"type": "bar", "run_id": "test", "close": 123.0})

    events = read_events(log_path)
    assert len(events) == 2
    assert events[0]["type"] == "run_start"
    assert events[1]["type"] == "bar"
    assert "ts_utc" in events[0]


def test_read_events_ignores_partial_last_line(tmp_path: Path) -> None:
    log_path = make_log_path(symbol="NVDA", frequency="60min", run_id="partial", live_dir=tmp_path)

    append_event(log_path, {"type": "run_start", "run_id": "partial"})

    # Simulate a crash during append (partial JSON line at EOF).
    with log_path.open("a", encoding="utf-8") as f:
        f.write('{"type": "bar"')

    events = read_events(log_path)
    assert len(events) == 1
    assert events[0]["type"] == "run_start"


def test_kill_switch_toggle(tmp_path: Path) -> None:
    assert not is_kill_switch_enabled(tmp_path)

    set_kill_switch(enabled=True, live_dir=tmp_path)
    assert is_kill_switch_enabled(tmp_path)

    set_kill_switch(enabled=False, live_dir=tmp_path)
    assert not is_kill_switch_enabled(tmp_path)
