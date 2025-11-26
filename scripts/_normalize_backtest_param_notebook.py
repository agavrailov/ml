from __future__ import annotations

from pathlib import Path

import nbformat


NB_PATH = Path("notebooks/backtest_param.ipynb")


def main() -> None:
    """Normalize sources in backtest_param.ipynb so no string contains "\n".

    We load the notebook, join each cell's source to a single string, split it
    into logical lines without keeping newline characters, and write the list
    of lines back. This ensures that every element of ``cell["source"]`` is a
    standalone line without embedded newlines, which is friendlier to our
    refactor tooling on Windows.
    """

    nb = nbformat.read(NB_PATH, as_version=4)

    for cell in nb.cells:
        src_obj = cell.get("source", "")
        if isinstance(src_obj, list):
            joined = "".join(src_obj)
        else:
            joined = src_obj
        if joined is None:
            joined = ""

        # Split into lines, dropping newline characters themselves.
        lines = joined.splitlines(keepends=False)
        cell["source"] = lines

    nbformat.write(nb, NB_PATH)


if __name__ == "__main__":
    main()