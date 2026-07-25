"""Compilation smoke tests for Python source files."""

from __future__ import annotations

import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile() -> None:
    """All project Python sources should parse before deeper tests run."""
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
    ]
    paths = [
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    for path in paths:
        py_compile.compile(str(path), doraise=True)
