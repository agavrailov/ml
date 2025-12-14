"""Filesystem-backed job runner.

Purpose:
- Run long tasks (train/backtest/optimize/etc.) outside Streamlit.
- Persist inputs, status, and artifacts under runs/<job_id>/.

This keeps UI reruns from interrupting work and makes runs reproducible.
"""
