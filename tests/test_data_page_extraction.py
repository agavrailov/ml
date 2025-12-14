from pathlib import Path


def test_data_page_module_importable() -> None:
    from src.ui.pages import data_page

    assert callable(data_page.render_data_tab)


def test_app_delegates_data_tab_to_page_module() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "from src.ui.pages import data_page" in text
    assert "data_page.render_data_tab" in text
