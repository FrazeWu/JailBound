from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmark.safety_eval.io import read_jsonl


ROOT = Path(__file__).resolve().parents[2]


def _controls_module():
    spec = importlib.util.spec_from_file_location(
        "run_fol_controls_test", ROOT / "scripts" / "run_fol_controls.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controls_source_selection_rejects_unconfigured_source() -> None:
    module = _controls_module()

    assert module._selected_sources(("jailbound", "s_eval"), ["jailbound"]) == ("jailbound",)
    with pytest.raises(ValueError, match="unconfigured source"):
        module._selected_sources(("jailbound", "s_eval"), ["other"])


def test_controls_ledger_merges_disjoint_source_workers(tmp_path: Path) -> None:
    module = _controls_module()
    path = tmp_path / "controls.jsonl"

    module._append_control_rows(path, [{"source": "jailbound", "sample_id": "a"}])
    module._append_control_rows(path, [{"source": "s_eval", "sample_id": "b"}])

    assert {(row["source"], row["sample_id"]) for row in read_jsonl(path)} == {
        ("jailbound", "a"),
        ("s_eval", "b"),
    }
