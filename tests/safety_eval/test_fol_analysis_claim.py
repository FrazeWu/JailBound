from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _analysis_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_fol_boundary_test", ROOT / "scripts" / "analyze_fol_boundary.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_payload_is_inconclusive_when_interpolation_has_too_few_paths() -> None:
    module = _analysis_module()

    payload = module._claim_payload(
        h3={
            "jailbound": {"valid_paths": 0, "crossing_paths": 0, "status": "inconclusive"},
            "s_eval": {"valid_paths": 2, "crossing_paths": 1, "status": "inconclusive"},
        },
        failures={"primary": 0, "secondary": 0, "both": 0},
        usable_directions=1088,
        minimum_paths=5,
    )

    assert payload["decision"] == "inconclusive"
    assert payload["reason"] == "interpolation_underpowered"
    assert payload["quality_gates"]["jailbound_valid_paths"] == 0
    assert payload["quality_gates"]["s_eval_valid_paths"] == 2
