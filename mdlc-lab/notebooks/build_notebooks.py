"""Generate student and solution notebooks from `notebooks/src/*.py`.

Source format (a subset of the "percent" format):

    # %% [markdown]
    # Markdown lines, each prefixed with "# ".
    # %%
    code ...
    ### BEGIN SOLUTION
    reference implementation
    ### END SOLUTION

In the student notebook each solution block becomes
`raise NotImplementedError("TODO")`, keeping the block's indentation.

Usage: python notebooks/build_notebooks.py
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
SOLUTIONS = HERE.parent / "solutions"

_CELL_RE = re.compile(r"^# %%(?: \[(markdown)\])?\s*$", re.M)
_SOL_RE = re.compile(r"^([ \t]*)### BEGIN SOLUTION\n.*?^[ \t]*### END SOLUTION\n?", re.M | re.S)


def parse(text: str) -> list[tuple[str, str]]:
    parts = _CELL_RE.split(text)
    cells = []
    # parts = [preamble, kind, body, kind, body, ...]
    for kind, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            cells.append(("markdown", body))
        else:
            cells.append(("code", body))
    return cells


def render(cells: list[tuple[str, str]], solution: bool) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, body in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
            continue
        if solution:
            body = re.sub(r"^[ \t]*### (BEGIN|END) SOLUTION\n?", "", body + "\n", flags=re.M).rstrip("\n")
        else:
            body = _SOL_RE.sub(lambda m: f'{m.group(1)}raise NotImplementedError("TODO")\n', body + "\n")
            body = body.rstrip("\n")
        nb.cells.append(nbformat.v4.new_code_cell(body))
    return nb


def main() -> None:
    SOLUTIONS.mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.py")):
        cells = parse(src.read_text())
        name = src.stem + ".ipynb"
        nbformat.write(render(cells, solution=False), HERE / name)
        nbformat.write(render(cells, solution=True), SOLUTIONS / name)
        print(f"built {name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
