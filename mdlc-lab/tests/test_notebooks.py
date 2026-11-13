import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "notebooks" / "build_notebooks.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)

SRC = """# %% [markdown]
# # Title
# %%
def f():
    ### BEGIN SOLUTION
    return 1
    ### END SOLUTION
"""


def test_student_and_solution_versions():
    cells = bn.parse(SRC)
    assert cells[0] == ("markdown", "# Title")
    student = bn.render(cells, solution=False).cells[1].source
    solution = bn.render(cells, solution=True).cells[1].source
    assert student == 'def f():\n    raise NotImplementedError("TODO")'
    assert solution == "def f():\n    return 1"


def test_committed_notebooks_are_up_to_date():
    import nbformat
    for src in (ROOT / "notebooks" / "src").glob("*.py"):
        cells = bn.parse(src.read_text())
        for solution, path in [(False, ROOT / "notebooks" / f"{src.stem}.ipynb"),
                               (True, ROOT / "solutions" / f"{src.stem}.ipynb")]:
            nb = nbformat.read(path, as_version=4)
            expected = [c.source for c in bn.render(cells, solution).cells]
            assert [c.source for c in nb.cells] == expected, f"{path.name} is stale: run make notebooks"
