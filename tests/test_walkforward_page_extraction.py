from pathlib import Path


def test_walkforward_page_module_importable() -> None:
    from src.ui.page_modules import walkforward_page

    assert callable(walkforward_page.render_walkforward_tab)


def test_app_delegates_walkforward_tab_to_page_module() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "from src.ui.page_modules import walkforward_page" in text
    assert "walkforward_page.render_walkforward_tab" in text
