from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from benchmark.safety_eval.intent_preservation import DriftReason
from benchmark.safety_eval.materialization_ablation import MaterializationPair


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(sample_id: str, branch: str, *, materialized_unsafe: bool) -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": branch,
        "optimization_checkpoint": 100,
        "state_sha256": "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": f"initial {sample_id}",
        "reference_intent": f"intent {sample_id}",
        "continuous_response": "continuous response",
        "materialized_text": f"materialized {sample_id}",
        "materialized_response": "materialized response",
        "editable_projected_token_ids": [4],
        "projected_token_ids": [4],
        "retokenized_token_ids": [4],
        "projection_cosines": [0.9],
        "roundtrip_exact_match": True,
        "projected_length": 1,
        "retokenized_length": 1,
        "max_new_tokens": 512,
        "status": "complete",
        "error": None,
        "judgments": {"octopus": {"continuous": True, "materialized": materialized_unsafe}},
    })


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_intent_annotation_and_example_clis_write_complete_evidence(tmp_path: Path) -> None:
    build_annotations = _load_script("build_intent_annotations")
    analyze_annotations = _load_script("analyze_intent_annotations")
    build_examples = _load_script("build_materialization_examples")
    pairs = [
        _pair("preserved", "High-Value", materialized_unsafe=True),
        _pair("drifted", "Safety-Sensitivity", materialized_unsafe=False),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")
    annotation_dir = tmp_path / "annotation"

    assert build_annotations.main([
        "--pairs", str(pairs_path),
        "--output-dir", str(annotation_dir),
        "--seed", "7",
    ]) == 0

    blinded_path = annotation_dir / "intent_annotation_blinded.csv"
    mapping_path = annotation_dir / "intent_annotation_mapping.csv"
    manifest_path = annotation_dir / "intent_annotation_export.json"
    mappings = list(csv.DictReader(mapping_path.open(encoding="utf-8")))
    assert blinded_path.is_file() and len(list(csv.DictReader(blinded_path.open(encoding="utf-8")))) == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["record_count"] == 2
    id_by_sample = {row["sample_id"]: row["annotation_id"] for row in mappings}

    raw_path = tmp_path / "raw.csv"
    final_path = tmp_path / "final.csv"
    raw_fields = ("annotation_id", "annotator_id", "initial_label", "drift_reason", "note")
    raw_rows: list[dict[str, str]] = []
    for sample_id, label, reason in (
        ("preserved", "Preserved", ""),
        ("drifted", "Not preserved", DriftReason.action_changed.value),
    ):
        for annotator in ("r1", "r2"):
            raw_rows.append({
                "annotation_id": id_by_sample[sample_id],
                "annotator_id": annotator,
                "initial_label": label,
                "drift_reason": reason,
                "note": "",
            })
    _write_csv(raw_path, raw_fields, raw_rows)
    _write_csv(
        final_path,
        ("annotation_id", "final_label", "drift_reason", "adjudication_note"),
        [
            {"annotation_id": id_by_sample["preserved"], "final_label": "Preserved", "drift_reason": "",
             "adjudication_note": ""},
            {"annotation_id": id_by_sample["drifted"], "final_label": "Not preserved",
             "drift_reason": DriftReason.action_changed.value, "adjudication_note": ""},
        ],
    )
    analysis_dir = tmp_path / "analysis"

    assert analyze_annotations.main([
        "--pairs", str(pairs_path),
        "--mapping", str(mapping_path),
        "--raw-labels", str(raw_path),
        "--final-labels", str(final_path),
        "--annotator", "r1",
        "--annotator", "r2",
        "--output-dir", str(analysis_dir),
    ]) == 0

    ipr_rows = list(csv.DictReader((analysis_dir / "intent_preservation.csv").open(encoding="utf-8")))
    assert [(row["Source"], row["Branch"], row["Total"]) for row in ipr_rows] == [
        ("harmbench", "High-Value", "1"),
        ("harmbench", "Safety-Sensitivity", "1"),
        ("harmbench", "All", "2"),
        ("All", "High-Value", "1"),
        ("All", "Safety-Sensitivity", "1"),
        ("Overall", "All", "2"),
    ]
    reason_rows = list(csv.DictReader((analysis_dir / "intent_drift_reasons.csv").open(encoding="utf-8")))
    assert [row["Drift reason"] for row in reason_rows] == [reason.value for reason in DriftReason]
    assert [int(row["Count"]) for row in reason_rows] == [1, 0, 0, 0, 0, 0]
    diagnostics = json.loads((analysis_dir / "intent_annotation_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["agreement"]["denominator"] == 2
    assert diagnostics["judge_cross_tabs"]["octopus"]["Not preserved"]["safe"] == 1

    examples_dir = tmp_path / "examples"
    assert build_examples.main([
        "--pairs", str(pairs_path),
        "--mapping", str(mapping_path),
        "--final-labels", str(final_path),
        "--source", "harmbench",
        "--output-dir", str(examples_dir),
    ]) == 0

    full = (examples_dir / "materialization_examples_full.md").read_text(encoding="utf-8")
    compact = (examples_dir / "materialization_examples_openreview.md").read_text(encoding="utf-8")
    index_rows = list(csv.DictReader((examples_dir / "materialization_example_index.csv").open(encoding="utf-8")))
    assert "materialized preserved" in full and "materialized drifted" in full
    assert "materialized preserved" in compact and "materialized drifted" in compact
    assert len(index_rows) == 4
    assert sum(row["case_id"] != "No case available" for row in index_rows) == 2
