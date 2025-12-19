from pathlib import Path


def test_experiments_page_module_importable() -> None:
    from src.ui.page_modules import experiments_page

    assert callable(experiments_page.render_experiments_tab)


def test_app_delegates_experiments_tab_to_page_module() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "from src.ui.page_modules import experiments_page" in text
    assert "experiments_page.render_experiments_tab" in text
