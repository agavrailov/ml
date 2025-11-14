"""Pytest configuration to make the project root importable as a package.

This ensures that ``import src`` and similar absolute imports work when tests
are run from the repository root or other locations.
"""

import os
import sys

# Project root = parent directory of this tests/ folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
