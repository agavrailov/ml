from __future__ import annotations

from pathlib import Path

import src.ui.app as ui_app


def test_ui_app_delegates_to_legacy_app(monkeypatch):
    called = {}

    def _fake_run_path(path, run_name=None):
        called["path"] = path
        called["run_name"] = run_name
        return {}

    monkeypatch.setattr(ui_app.runpy, "run_path", _fake_run_path)

    ui_app.main()

    assert called["run_name"] == "__main__"

    legacy_path = Path(called["path"]).resolve()
    assert legacy_path.name == "app.py"
    assert legacy_path.parent.name == "src"
