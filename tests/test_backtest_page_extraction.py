from pathlib import Path


def test_backtest_page_module_importable() -> None:
    # Import should not pull in Streamlit at import time (render receives `st`).
    from src.ui.page_modules import backtest_page

    assert callable(backtest_page.render_backtest_tab)


def test_app_delegates_backtest_tab_to_page_module() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "from src.ui.page_modules import backtest_page" in text
    assert "backtest_page.render_backtest_tab" in text
