from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.safety_eval.h1_v3_runtime import (
    H1V3Paths,
    build_h1_v3_contract,
    build_h1_v3_schedule,
    frozen_h1_v2_selection,
    validate_h1_v3_contract,
)


def test_h1_v3_schedule_uses_only_new_radii_and_deterministic_ids() -> None:
    rows = build_h1_v3_schedule(
        sample_ids=("opaque:a",), radii=(0.4, 0.6), attempts=64, seed=7
    )

    assert len(rows) == 128
    assert {row.radius for row in rows} == {0.4, 0.6}
    assert all(row.perturbation_id.startswith("fol-h1-v3:") for row in rows)
    assert rows == build_h1_v3_schedule(
        sample_ids=("opaque:a",), radii=(0.6, 0.4), attempts=64, seed=7
    )


def test_h1_v3_paths_write_only_under_output_root(tmp_path: Path) -> None:
    paths = H1V3Paths(
        source_root=tmp_path / "fol_h1_v2", output_root=tmp_path / "fol_h1_v3"
    )

    assert paths.response_ledger("qwen2_5_7b", "jailbound").is_relative_to(paths.output_root)
    assert not paths.response_ledger("qwen2_5_7b", "jailbound").is_relative_to(paths.source_root)
    assert paths.source_selection().is_relative_to(paths.source_root)


def test_h1_v3_contract_rejects_changed_h1_v2_selection_hash() -> None:
    contract = build_h1_v3_contract(
        source_hash="a" * 64,
        selected_ids=(("jailbound", "opaque"),),
        radii=(0.4, 0.6),
    )

    validate_h1_v3_contract(
        contract,
        source_hash="a" * 64,
        selected_ids=(("jailbound", "opaque"),),
        radii=(0.4, 0.6),
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_h1_v3_contract(
            contract,
            source_hash="b" * 64,
            selected_ids=(("jailbound", "opaque"),),
            radii=(0.4, 0.6),
        )


def test_h1_v3_selection_uses_selected_fol_values_and_allows_candidate_diagnostics(tmp_path: Path) -> None:
    source_root = tmp_path / "fol_h1_v2"
    sources = {}
    for source in ("jailbound", "s_eval"):
        bands = {
            "low": [f"{source}:low:{index}" for index in range(17)],
            "middle": [f"{source}:middle:{index}" for index in range(3)],
            "high": [f"{source}:high:{index}" for index in range(17)],
        }
        ids = [sample_id for values in bands.values() for sample_id in values]
        sources[source] = {
            **bands,
            "fol_by_id": {**{sample_id: float(index) for index, sample_id in enumerate(ids)}, f"{source}:unselected": 99.0},
        }
    path = source_root / "manifests" / "h1_v2_validation_selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"sources": sources}), encoding="utf-8")

    selection = frozen_h1_v2_selection(source_root)

    assert len(selection["jailbound"]["fol_by_id"]) == 37
    assert f"jailbound:unselected" not in selection["jailbound"]["fol_by_id"]
