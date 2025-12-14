# UI Architecture

## Navigation Structure

This UI uses **horizontal tabs** for navigation instead of Streamlit's multipage sidebar navigation.

### Why Tabs?

**Pros:**
- ✅ All content loads once - faster tab switching
- ✅ State persists naturally across tabs (no page reloads)
- ✅ Better for workflows that move data between steps
- ✅ Single-page app simplicity

**Trade-offs:**
- ⚠️ No URL routing (tabs are not bookmarkable)
- ⚠️ Browser back/forward doesn't switch tabs

### Directory Structure

```
src/ui/
├── app.py              # Main entry point with st.tabs()
├── page_modules/       # Tab content modules (NOT "pages" - prevents auto-discovery)
│   ├── live_page.py
│   ├── data_page.py
│   ├── experiments_page.py
│   ├── train_page.py
│   ├── backtest_page.py
│   └── walkforward_page.py
├── state.py            # Centralized UI state management
├── formatting.py       # Display helpers
└── registry.py         # Model registry operations
```

### Important: Why `page_modules/` not `pages/`?

Streamlit auto-discovers directories named `pages/` and creates sidebar navigation automatically. By using `page_modules/` instead, we:
- Avoid double navigation (sidebar + tabs)
- Keep clean tab-based UX
- Prevent empty sidebar pages from appearing

### Running the UI

```bash
streamlit run src/ui/app.py
```

### State Management

All UI state is managed through:
- `src/ui/state.py`: Centralized session state with `get_ui_state()`
- Nested sections: `data`, `experiments`, `training`, `strategy`, `backtests`, `optimization`, `walkforward`
- JSON persistence for history tables and parameter grids

### Tab Flow

1. **Live**: Monitor live trading sessions
2. **Data**: Ingest and prepare OHLC data
3. **Experiments**: Hyperparameter search
4. **Train & Promote**: Full model training
5. **Backtest / Strategy**: Test and optimize strategies
6. **Walk-Forward**: Robustness validation

Each tab loads its content from a module in `page_modules/` via a `render_*_tab()` function.
