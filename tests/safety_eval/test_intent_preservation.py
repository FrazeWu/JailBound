from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark.safety_eval.intent_preservation import (
    analyze_intent_labels,
    export_blinded_annotations,
    read_final_labels,
    read_raw_labels,
)
from benchmark.safety_eval.materialization_ablation import MaterializationPair


def _pair(
    sample_id: str,
    *,
    preserved_by_judge: bool,
    source: str = "harmbench",
    branch: str = "High-Value",
) -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": source,
        "sample_id": sample_id,
        "branch": branch,
        "optimization_checkpoint": 100,
        "state_sha256": "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": f"initial {sample_id}",
        "reference_intent": f"intent {sample_id}",
        "continuous_response": "continuous",
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
        "judgments": {
            "octopus": {"continuous": True, "materialized": preserved_by_judge}
        },
    })


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_is_deterministic_and_blinded(tmp_path: Path) -> None:
    pairs = [_pair("b", preserved_by_judge=False), _pair("a", preserved_by_judge=True)]

    first = export_blinded_annotations(pairs, seed=20260731, output_dir=tmp_path / "first")
    second = export_blinded_annotations(reversed(pairs), seed=20260731, output_dir=tmp_path / "second")

    assert first.annotation_ids == second.annotation_ids
    blinded = first.blinded_csv.read_text(encoding="utf-8")
    assert blinded.splitlines()[0] == "annotation_id,reference_intent,materialized_prompt"
    assert "harmbench" not in blinded
    assert "High-Value" not in blinded
    assert "octopus" not in blinded
    mapping = first.mapping_csv.read_text(encoding="utf-8")
    assert "materialized_text_hash" in mapping
    assert "sample_id" in mapping and "branch" in mapping


def test_raw_labels_require_complete_two_annotator_coverage(tmp_path: Path) -> None:
    exported = export_blinded_annotations([_pair("a", preserved_by_judge=True)], seed=7, output_dir=tmp_path)
    annotation_id = exported.annotation_ids[0]
    raw = tmp_path / "raw.csv"
    _write_csv(raw, ["annotation_id", "annotator_id", "initial_label", "drift_reason", "note"], [{
        "annotation_id": annotation_id,
        "annotator_id": "r1",
        "initial_label": "Preserved",
        "drift_reason": "",
        "note": "",
    }])

    with pytest.raises(ValueError, match="coverage"):
        read_raw_labels(raw, exported.mapping_csv, expected_annotators=("r1", "r2"))


def test_not_preserved_requires_reason_and_other_requires_note(tmp_path: Path) -> None:
    exported = export_blinded_annotations([_pair("a", preserved_by_judge=True)], seed=7, output_dir=tmp_path)
    annotation_id = exported.annotation_ids[0]
    raw = tmp_path / "raw.csv"
    fields = ["annotation_id", "annotator_id", "initial_label", "drift_reason", "note"]
    rows = [
        {"annotation_id": annotation_id, "annotator_id": "r1", "initial_label": "Not preserved", "drift_reason": "", "note": ""},
        {"annotation_id": annotation_id, "annotator_id": "r2", "initial_label": "Not preserved", "drift_reason": "OTHER", "note": ""},
    ]
    _write_csv(raw, fields, rows)

    with pytest.raises(ValueError, match="drift reason"):
        read_raw_labels(raw, exported.mapping_csv, expected_annotators=("r1", "r2"))

    rows[0]["drift_reason"] = "ACTION_CHANGED"
    _write_csv(raw, fields, rows)
    with pytest.raises(ValueError, match="OTHER requires a note"):
        read_raw_labels(raw, exported.mapping_csv, expected_annotators=("r1", "r2"))


def test_final_labels_reject_unsure_and_compute_count_first_analysis(tmp_path: Path) -> None:
    pairs = [_pair("a", preserved_by_judge=True), _pair("b", preserved_by_judge=False)]
    exported = export_blinded_annotations(pairs, seed=7, output_dir=tmp_path)
    mapping_rows = list(csv.DictReader(exported.mapping_csv.open(encoding="utf-8")))
    id_by_sample = {row["sample_id"]: row["annotation_id"] for row in mapping_rows}
    raw = tmp_path / "raw.csv"
    raw_fields = ["annotation_id", "annotator_id", "initial_label", "drift_reason", "note"]
    raw_rows = [
        {"annotation_id": id_by_sample["a"], "annotator_id": "r1", "initial_label": "Preserved", "drift_reason": "", "note": ""},
        {"annotation_id": id_by_sample["a"], "annotator_id": "r2", "initial_label": "Preserved", "drift_reason": "", "note": ""},
        {"annotation_id": id_by_sample["b"], "annotator_id": "r1", "initial_label": "Not preserved", "drift_reason": "ACTION_CHANGED", "note": ""},
        {"annotation_id": id_by_sample["b"], "annotator_id": "r2", "initial_label": "Preserved", "drift_reason": "", "note": ""},
    ]
    _write_csv(raw, raw_fields, raw_rows)
    final = tmp_path / "final.csv"
    final_fields = ["annotation_id", "final_label", "drift_reason", "adjudication_note"]
    _write_csv(final, final_fields, [
        {"annotation_id": id_by_sample["a"], "final_label": "Preserved", "drift_reason": "", "adjudication_note": ""},
        {"annotation_id": id_by_sample["b"], "final_label": "Not preserved", "drift_reason": "ACTION_CHANGED", "adjudication_note": "reviewed"},
    ])

    raw_labels = read_raw_labels(raw, exported.mapping_csv, expected_annotators=("r1", "r2"))
    final_labels = read_final_labels(final, exported.mapping_csv)
    analysis = analyze_intent_labels(pairs, exported.mapping_csv, raw_labels, final_labels)

    assert analysis.agreement.denominator == 2
    assert analysis.agreement.raw_agreement == pytest.approx(0.5)
    assert analysis.agreement.kappa == pytest.approx(0.0)
    assert analysis.agreement.disagreements == 1
    assert analysis.agreement.unsure_records == 0
    assert analysis.agreement.adjudications_required == 1
    overall = next(row for row in analysis.ipr if row.source == "Overall")
    assert (overall.preserved, overall.total, overall.rate) == (1, 2, pytest.approx(0.5))
    assert analysis.drift_reason_counts == {"ACTION_CHANGED": 1}
    assert analysis.judge_cross_tabs["octopus"] == {
        "Preserved": {"safe": 0, "unsafe": 1},
        "Not preserved": {"safe": 1, "unsafe": 0},
    }


def test_final_labels_require_one_binary_label_per_annotation(tmp_path: Path) -> None:
    exported = export_blinded_annotations([_pair("a", preserved_by_judge=True)], seed=7, output_dir=tmp_path)
    final = tmp_path / "final.csv"
    _write_csv(final, ["annotation_id", "final_label", "drift_reason", "adjudication_note"], [{
        "annotation_id": exported.annotation_ids[0],
        "final_label": "Unsure",
        "drift_reason": "",
        "adjudication_note": "",
    }])

    with pytest.raises(ValueError, match="final label"):
        read_final_labels(final, exported.mapping_csv)


def test_analysis_requires_adjudication_note_for_disagreement_or_any_unsure(tmp_path: Path) -> None:
    pair = _pair("a", preserved_by_judge=True)
    exported = export_blinded_annotations([pair], seed=7, output_dir=tmp_path)
    annotation_id = exported.annotation_ids[0]
    raw_path = tmp_path / "raw.csv"
    raw_fields = ["annotation_id", "annotator_id", "initial_label", "drift_reason", "note"]
    final_path = tmp_path / "final.csv"
    final_fields = ["annotation_id", "final_label", "drift_reason", "adjudication_note"]

    for left, right in (("Preserved", "Not preserved"), ("Unsure", "Unsure")):
        _write_csv(raw_path, raw_fields, [
            {"annotation_id": annotation_id, "annotator_id": "r1", "initial_label": left,
             "drift_reason": "ACTION_CHANGED" if left == "Not preserved" else "", "note": ""},
            {"annotation_id": annotation_id, "annotator_id": "r2", "initial_label": right,
             "drift_reason": "ACTION_CHANGED" if right == "Not preserved" else "", "note": ""},
        ])
        _write_csv(final_path, final_fields, [{
            "annotation_id": annotation_id,
            "final_label": "Preserved",
            "drift_reason": "",
            "adjudication_note": "",
        }])
        raw = read_raw_labels(raw_path, exported.mapping_csv, expected_annotators=("r1", "r2"))
        final = read_final_labels(final_path, exported.mapping_csv)

        with pytest.raises(ValueError, match="adjudication note"):
            analyze_intent_labels([pair], exported.mapping_csv, raw, final)

    _write_csv(final_path, final_fields, [{
        "annotation_id": annotation_id,
        "final_label": "Preserved",
        "drift_reason": "",
        "adjudication_note": "reviewed Unsure labels",
    }])
    raw = read_raw_labels(raw_path, exported.mapping_csv, expected_annotators=("r1", "r2"))
    final = read_final_labels(final_path, exported.mapping_csv)
    agreement = analyze_intent_labels([pair], exported.mapping_csv, raw, final).agreement

    assert agreement.raw_agreement == pytest.approx(1.0)
    assert agreement.disagreements == 0
    assert agreement.unsure_records == 1
    assert agreement.adjudications_required == 1


def test_ipr_contains_all_aggregation_levels_in_deterministic_order(tmp_path: Path) -> None:
    pairs = [
        _pair("h-hv", preserved_by_judge=True, source="harmbench", branch="High-Value"),
        _pair("h-ss", preserved_by_judge=True, source="harmbench", branch="Safety-Sensitivity"),
        _pair("s-hv", preserved_by_judge=True, source="s_eval", branch="High-Value"),
        _pair("s-ss", preserved_by_judge=True, source="s_eval", branch="Safety-Sensitivity"),
    ]
    exported = export_blinded_annotations(pairs, seed=7, output_dir=tmp_path)
    mapping_rows = list(csv.DictReader(exported.mapping_csv.open(encoding="utf-8")))
    raw_path = tmp_path / "raw.csv"
    final_path = tmp_path / "final.csv"
    _write_csv(
        raw_path,
        ["annotation_id", "annotator_id", "initial_label", "drift_reason", "note"],
        [
            {"annotation_id": row["annotation_id"], "annotator_id": annotator, "initial_label": "Preserved",
             "drift_reason": "", "note": ""}
            for row in mapping_rows
            for annotator in ("r1", "r2")
        ],
    )
    _write_csv(
        final_path,
        ["annotation_id", "final_label", "drift_reason", "adjudication_note"],
        [
            {"annotation_id": row["annotation_id"], "final_label": "Preserved", "drift_reason": "",
             "adjudication_note": ""}
            for row in mapping_rows
        ],
    )

    analysis = analyze_intent_labels(
        pairs,
        exported.mapping_csv,
        read_raw_labels(raw_path, exported.mapping_csv, expected_annotators=("r1", "r2")),
        read_final_labels(final_path, exported.mapping_csv),
    )

    assert [(row.source, row.branch, row.total) for row in analysis.ipr] == [
        ("harmbench", "High-Value", 1),
        ("harmbench", "Safety-Sensitivity", 1),
        ("s_eval", "High-Value", 1),
        ("s_eval", "Safety-Sensitivity", 1),
        ("harmbench", "All", 2),
        ("s_eval", "All", 2),
        ("All", "High-Value", 2),
        ("All", "Safety-Sensitivity", 2),
        ("Overall", "All", 4),
    ]
