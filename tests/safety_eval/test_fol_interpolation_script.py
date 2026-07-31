from __future__ import annotations

import importlib.util
from pathlib import Path


def test_interpolation_roots_separate_frozen_states_from_outputs(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_fol_interpolation.py"
    spec = importlib.util.spec_from_file_location("fol_interpolation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default_state = tmp_path / "default"
    frozen = tmp_path / "frozen"
    results = tmp_path / "results"

    state_root, output_root = module.resolve_fol_roots(default_state, frozen, results)

    assert state_root == frozen
    assert output_root == results
    assert module.resolve_fol_roots(default_state, None, None) == (default_state, default_state)
