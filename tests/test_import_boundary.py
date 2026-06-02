"""
Import Boundary Test — tests/test_import_boundary.py

Asserts the seam contract (locked architectural decision):

    "The live stack (agents/, execution/) may only import from research/
     through research/seam.py. A CI import-boundary test enforces this."

How it works:
    1. Walks every .py file under agents/ and execution/.
    2. Parses each file's AST and extracts all import statements.
    3. Asserts that no import targets research.* UNLESS the target is
       exactly 'research.seam' or 'research.seam.SignalPacket'.

This test must remain fast (<1s) and dependency-free — it uses only
Python's stdlib ast module, no runtime imports of the modules under test.
If this test fails, it means a live-stack module has broken the boundary.
"""
import ast
import os
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

# Root of the repo (two levels up from this file: tests/ → repo root)
REPO_ROOT = Path(__file__).parent.parent

# Directories that constitute the "live stack"
LIVE_STACK_DIRS = [
    REPO_ROOT / "agents",
    REPO_ROOT / "execution",
]

# The only research/ import path permitted in the live stack
ALLOWED_RESEARCH_MODULE = "research.seam"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_python_files(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in directories:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return files


def _extract_research_imports(source: str) -> list[str]:
    """
    Return every research.* module name imported in the given source code.
    Handles both `import X` and `from X import Y` forms.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    research_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("research"):
                    research_imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("research"):
                research_imports.append(module)

    return research_imports


def _is_allowed(module: str) -> bool:
    """Return True if the research import is the permitted seam path."""
    return module == ALLOWED_RESEARCH_MODULE or module.startswith(
        ALLOWED_RESEARCH_MODULE + "."
    )


# ── Test ──────────────────────────────────────────────────────────────────────

def test_live_stack_only_imports_from_seam() -> None:
    """
    No module under agents/ or execution/ may import from research.*
    except research.seam (or research.seam.*).

    Failure means a live-stack module bypassed the boundary.
    Fix: move the import to research/seam.py and expose what's needed there.
    """
    violations: list[str] = []

    live_stack_files = _collect_python_files(LIVE_STACK_DIRS)

    for py_file in live_stack_files:
        source = py_file.read_text(encoding="utf-8")
        research_imports = _extract_research_imports(source)

        for module in research_imports:
            if not _is_allowed(module):
                relative = py_file.relative_to(REPO_ROOT)
                violations.append(
                    f"  {relative}: imports '{module}' — "
                    f"only 'research.seam' is permitted in the live stack."
                )

    assert not violations, (
        "Import boundary violation(s) detected.\n"
        "The live stack may only import from research/ via research/seam.py.\n\n"
        + "\n".join(violations)
    )


def test_research_seam_itself_has_no_internal_research_imports() -> None:
    """
    research/seam.py must not import from other research/ sub-packages.
    It is the boundary file — it must be self-contained.
    """
    seam_file = REPO_ROOT / "research" / "seam.py"
    assert seam_file.exists(), "research/seam.py does not exist."

    source = seam_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("research."), (
                f"research/seam.py imports from '{module}'. "
                "seam.py must be self-contained — it cannot import "
                "from other research/ sub-packages."
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("research."), (
                    f"research/seam.py imports '{alias.name}'. "
                    "seam.py must be self-contained."
                )


def test_live_stack_directories_exist() -> None:
    """Sanity: the directories we're scanning actually exist."""
    for d in LIVE_STACK_DIRS:
        assert d.exists(), f"Expected live-stack directory does not exist: {d}"


def test_boundary_test_scans_at_least_one_file() -> None:
    """Sanity: if this returns 0 files, the test is doing nothing."""
    files = _collect_python_files(LIVE_STACK_DIRS)
    assert len(files) > 0, (
        "No Python files found under agents/ or execution/. "
        "The boundary test is not scanning anything."
    )
