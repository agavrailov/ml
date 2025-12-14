"""Profile the Streamlit app to identify performance bottlenecks."""

import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def profile_imports():
    """Profile all imports to find slow ones."""
    import time
    
    print("=" * 80)
    print("PROFILING APP IMPORTS")
    print("=" * 80)
    
    # Time each major import
    imports_to_test = [
        ("streamlit", "import streamlit as st"),
        ("pandas", "import pandas as pd"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("numpy", "import numpy as np"),
        ("src.ui.components", "from src.ui import components"),
        ("src.backtest", "from src.backtest import run_backtest_for_ui"),
        ("src.train", "from src.train import train_model"),
        ("src.config", "from src.config import FREQUENCY, TSTEPS"),
        ("src.data_processing", "from src.data_processing import clean_raw_minute_data"),
    ]
    
    results = []
    for name, import_stmt in imports_to_test:
        start = time.perf_counter()
        try:
            exec(import_stmt, globals())
            elapsed = time.perf_counter() - start
            results.append((name, elapsed, "OK"))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((name, elapsed, f"ERROR: {e}"))
    
    # Sort by time
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Module':<30} {'Time (ms)':<12} {'Status'}")
    print("-" * 80)
    for name, elapsed, status in results:
        print(f"{name:<30} {elapsed*1000:>10.1f} ms  {status}")
    
    print(f"\n{'TOTAL IMPORT TIME:':<30} {sum(r[1] for r in results)*1000:>10.1f} ms")
    print("=" * 80)

def profile_app_startup():
    """Profile the app.py module load."""
    import time
    
    print("\n" + "=" * 80)
    print("PROFILING APP STARTUP")
    print("=" * 80)
    
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.perf_counter()
    
    # Import the app module (this triggers all page module imports)
    try:
        from src.ui import app
    except SystemExit:
        pass  # Streamlit may call sys.exit()
    
    elapsed = time.perf_counter() - start
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 slowest functions
    
    print(f"\nApp startup time: {elapsed*1000:.1f} ms\n")
    print("Top 30 functions by cumulative time:")
    print("-" * 80)
    print(s.getvalue())
    print("=" * 80)

def profile_component_rendering():
    """Profile component rendering speed."""
    import time
    import streamlit as st
    import pandas as pd
    from src.ui import components
    
    print("\n" + "=" * 80)
    print("PROFILING COMPONENT RENDERING")
    print("=" * 80)
    
    # Mock streamlit for testing
    class MockSt:
        def markdown(self, *args, **kwargs):
            pass
        def columns(self, n):
            return [MockSt() for _ in range(n)]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def container(self):
            return self
    
    mock_st = MockSt()
    
    # Test metric card rendering
    start = time.perf_counter()
    for i in range(100):
        components.render_metric_card(
            mock_st, 
            label="Test Metric", 
            value="42.5%", 
            color="success",
            icon="📈"
        )
    elapsed = time.perf_counter() - start
    print(f"100 metric cards: {elapsed*1000:.1f} ms ({elapsed*10:.2f} ms per card)")
    
    # Test KPI row rendering
    metrics = [
        {"label": f"Metric {i}", "value": f"{i*10}%", "color": "success", "icon": "📈"}
        for i in range(8)
    ]
    start = time.perf_counter()
    for i in range(100):
        components.render_kpi_row(mock_st, metrics)
    elapsed = time.perf_counter() - start
    print(f"100 KPI rows (8 cards each): {elapsed*1000:.1f} ms ({elapsed*10:.2f} ms per row)")
    
    print("=" * 80)

if __name__ == "__main__":
    profile_imports()
    profile_app_startup()
    profile_component_rendering()
    
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("""
Common causes of slow Streamlit loads:
1. Heavy imports (pandas, matplotlib, tensorflow, etc.)
2. Module-level code execution
3. Large data loading at import time
4. Circular imports
5. Too many page modules being imported

Recommendations:
- Use lazy imports for heavy libraries (import inside functions)
- Move expensive operations inside @st.cache_data decorators
- Minimize module-level code execution
- Consider code splitting for page modules
""")
