"""Streamlit entrypoint.

For now this delegates to the existing monolithic Streamlit app at `src/app.py`.
As we migrate pages into `src/ui/pages/*`, we will replace this delegation with
native UI composition and eventually retire `src/app.py`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    # streamlit run src/ui/app.py -> sys.path[0] is usually src/ui.
    # Ensure repo root is importable so `import src.*` works reliably.
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    _ensure_repo_root_on_syspath()

    legacy_app_path = Path(__file__).resolve().parents[1] / "app.py"
    # Execute legacy Streamlit script in-process.
    runpy.run_path(str(legacy_app_path), run_name="__main__")


# Streamlit runs the script top-to-bottom; keep a simple guard for direct exec.
if __name__ == "__main__":  # pragma: no cover
    main()
