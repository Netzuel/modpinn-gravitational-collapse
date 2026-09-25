"""Documentation links, protected math, and executable beginner notebooks."""

import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md"] + sorted(
    [*ROOT.joinpath("docs").rglob("*.md"), *ROOT.joinpath("summary").glob("*.md")]
)


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda path: str(path.relative_to(ROOT)))
def test_documentation_local_links_and_math(path):
    text = path.read_text()
    assert text.count("```") % 2 == 0, "Unclosed code or math fence"
    for target in re.findall(r"\]\(([^)]+)\)|(?:src|srcset)=\"([^\"]+)\"", text):
        link = next(part for part in target if part)
        parts = urlsplit(link)
        if not parts.scheme and parts.path:
            assert (path.parent / unquote(parts.path)).exists(), link
    # Protect TeX from Markdown's punctuation escaping and emphasis processing.
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    assert "$$" not in prose, "Use fenced math blocks on GitHub"
    for expression in re.findall(r"\$([^$]+)\$", prose):
        assert expression.startswith("`") and expression.endswith("`"), expression
    expressions = re.findall(r"```math\n(.*?)```|\$`(.*?)`\$", text, re.S)
    for pair in expressions:
        expression = next(part for part in pair if part)
        # GitHub's browser renderer can interpret literal < as HTML, even in
        # fenced math. Balanced TeX alone does not detect this rendering error.
        assert "<" not in expression, r"Use \lt instead of < in GitHub math"
        depth = 0
        for brace in re.findall(r"(?<!\\)[{}]", expression):
            depth += 1 if brace == "{" else -1
            assert depth >= 0, expression
        assert depth == 0, expression


@pytest.mark.parametrize("name", ["quickstart.ipynb", "inference.ipynb"])
def test_basic_notebook_runs_from_fresh_kernel(name, tmp_path):
    path = ROOT / "docs/examples" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs
    manager = KernelManager(kernel_name="python3")
    assert manager.kernel_spec is not None
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(notebook, km=manager, timeout=120)
    try:
        executed = client.execute(
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        )
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert cells and all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    if name == "quickstart.ipynb":
        results = list(tmp_path.glob("datasets/runs/*/metrics.json"))
        assert len(results) == 1
        metrics = json.loads(results[0].read_text())
        assert metrics["epochs_completed"] == 3
        assert metrics["parameter_change_l2"] > 0
    else:
        assert any(
            "image/png" in output.get("data", {}) for cell in cells for output in cell.outputs
        ), "The inference tutorial must display its predicted profiles"
