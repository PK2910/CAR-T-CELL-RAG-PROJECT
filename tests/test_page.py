"""The web page's script: it parses, and its answer renderer passes tests/page_render_checks.js (needs Node.js)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = re.search(r"<script>(.*)</script>", (ROOT / "api/static/index.html").read_text(), re.S).group(1)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


@needs_node
def test_page_script_parses(tmp_path):
    (tmp_path / "page.js").write_text(SCRIPT)
    subprocess.run(["node", "--check", str(tmp_path / "page.js")], check=True)


@needs_node
def test_answer_renderer(tmp_path):
    # The rendering functions only (the rest of the script needs a browser page)
    render = SCRIPT[SCRIPT.index("function escapeHtml"):SCRIPT.index("// ---------- page state")]
    (tmp_path / "render.js").write_text(render + "\nmodule.exports = { renderMarkdown, renderAnswer };\n")
    result = subprocess.run(["node", str(ROOT / "tests/page_render_checks.js"), str(tmp_path / "render.js")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
