from __future__ import annotations

import importlib.util
import json
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


def test_h4_reads_controls_from_state_root_when_outputs_are_separate(tmp_path: Path) -> None:
    module = _analysis_module()
    output_root = tmp_path / "outputs"
    state_root = tmp_path / "states"
    output_root.mkdir()
    state_root.mkdir()
    (state_root / "controls.jsonl").write_text(
        json.dumps({"source": "jailbound", "sample_id": "opaque-id"}) + "\n",
        encoding="utf-8",
    )

    status = module._write_h4(
        tmp_path / "h4.csv",
        root=output_root,
        controls_root=state_root,
        primary_key="primary",
        primary_threshold=0.5,
        target_key="target",
        sources=("jailbound",),
        folds=2,
        seed=7,
    )

    assert status == {"status": "inconclusive", "reason": "ValueError"}
