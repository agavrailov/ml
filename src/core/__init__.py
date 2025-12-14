"""Core (pure) library layer.

This package is intended to be UI-agnostic and safe to import from:
- jobs (subprocess workers)
- CLI entrypoints
- tests

It should not import Streamlit or trigger long-running side effects at import time.
"""
