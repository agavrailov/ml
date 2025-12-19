from pathlib import Path


def test_live_page_module_importable() -> None:
    from src.ui.page_modules import live_page

    assert callable(live_page.render_live_tab)


def test_app_delegates_live_tab_to_page_module() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "from src.ui.page_modules import live_page" in text
    assert "live_page.render_live_tab" in text
