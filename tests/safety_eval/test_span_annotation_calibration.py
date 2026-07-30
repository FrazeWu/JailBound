from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from benchmark.safety_eval.datasets import RawExample
from benchmark.safety_eval.io import canonical_json, read_jsonl
from benchmark.safety_eval.schema import EditableSpan, EditableSpanRole
from benchmark.safety_eval.span_annotation import FrozenSpanAnnotation


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "calibrate_span_annotation_confidence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "calibrate_span_annotation_confidence", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CalibrationError = _MODULE.CalibrationError
ReviewedAnnotation = _MODULE.ReviewedAnnotation
calibrate_confidence = _MODULE.calibrate_confidence
freeze_reviewed_annotations = _MODULE.freeze_reviewed_annotations
parse_args = _MODULE.parse_args
prepare_review_rows = _MODULE.prepare_review_rows
write_frozen_artifact = _MODULE.write_frozen_artifact


def _review(confidence: float, accepted: bool) -> object:
    return ReviewedAnnotation(confidence=confidence, accepted=accepted)


def test_calibration_selects_maximum_recall_then_lowest_threshold() -> None:
    result = calibrate_confidence(
        (
            _review(0.95, True),
            _review(0.90, True),
            _review(0.85, False),
            _review(0.80, False),
        ),
        target_precision=1.0,
        minimum_selected=1,
    )

    assert result.threshold == 0.90
    assert result.selected_count == 2
    assert result.accepted_count == 2
    assert result.reviewed_count == 4
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_calibration_uses_lowest_threshold_when_recall_ties() -> None:
    result = calibrate_confidence(
        (_review(0.9, True), _review(0.8, False), _review(0.7, False)),
        target_precision=0.3,
        minimum_selected=1,
    )

    assert result.threshold == 0.7
    assert result.recall == 1.0


@pytest.mark.parametrize(
    ("reviews", "target_precision", "minimum_selected", "message"),
    [
        ((ReviewedAnnotation(confidence=0.9, accepted=None),), 0.9, 1, "incomplete"),
        (({"confidence": 0.9, "accepted": 1},), 0.9, 1, "exact boolean"),
        ((_review(0.9, False),), 0.9, 1, "accepted positive"),
        ((_review(0.9, True),), 0.0, 1, "target_precision"),
        ((_review(0.9, True),), 1.1, 1, "target_precision"),
        ((_review(0.9, True),), 0.9, True, "minimum_selected"),
        ((_review(0.9, True),), 0.9, 0, "minimum_selected"),
        ((_review(0.9, True), _review(0.8, False)), 1.0, 2, "qualifies"),
    ],
)
def test_calibration_rejects_invalid_or_unusable_inputs(
    reviews: tuple[object, ...],
    target_precision: float,
    minimum_selected: int,
    message: str,
) -> None:
    with pytest.raises((CalibrationError, ValueError), match=message):
        calibrate_confidence(
            reviews,
            target_precision=target_precision,
            minimum_selected=minimum_selected,
        )


class FixtureAnnotator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def annotate(self, prompt: str) -> FrozenSpanAnnotation:
        self.prompts.append(prompt)
        return FrozenSpanAnnotation(
            spans=(
                EditableSpan(
                    start=0,
                    end=len(prompt),
                    quote=prompt,
                    role=EditableSpanRole.harmful_payload,
                    confidence=0.75,
                    rationale="fixture",
                ),
            ),
            confidence=0.75,
            template_sha256="1" * 64,
            response_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            model="fixture-model",
            revision="fixture-revision",
        )


def _raw(source: str, row: int) -> RawExample:
    return RawExample(
        source=source,
        source_row=row,
        intent=f"intent-{source}-{row}",
        attack_text=f"prompt-{source}-{row}",
        target_text=None,
        source_risk_label=f"risk-{source}",
        source_attack_label=f"attack-{source}",
        source_domain_label=f"domain-{source}",
        language="en",
        preprocessing=("fixture",),
    )


def test_prepare_core_samples_fixed_count_per_source_deterministically() -> None:
    sources = {
        "source_b": tuple(_raw("source_b", row) for row in range(5)),
        "source_a": tuple(_raw("source_a", row) for row in range(5)),
    }
    first_annotator = FixtureAnnotator()
    second_annotator = FixtureAnnotator()

    first = prepare_review_rows(
        sources, per_source=2, seed=17, annotator=first_annotator
    )
    second = prepare_review_rows(
        sources, per_source=2, seed=17, annotator=second_annotator
    )

    assert first == second
    assert len(first) == 4
    assert [row["source"] for row in first] == [
        "source_a",
        "source_a",
        "source_b",
        "source_b",
    ]
    assert first_annotator.prompts == second_annotator.prompts
    assert all(row["accepted"] is None for row in first)
    assert all(row["confidence"] == 0.75 for row in first)
    assert all(row["annotation_model"] == "fixture-model" for row in first)
    assert all(row["seed_intent"].startswith("intent-") for row in first)
    assert all(row["source_hints"]["language"] == "en" for row in first)
    assert all(row["spans"][0]["quote"] == row["prompt"] for row in first)


def test_prepare_core_rejects_short_source_or_invalid_count() -> None:
    with pytest.raises(CalibrationError, match="requires 2 rows"):
        prepare_review_rows(
            {"source": (_raw("source", 0),)},
            per_source=2,
            seed=17,
            annotator=FixtureAnnotator(),
        )
    with pytest.raises(ValueError, match="per_source"):
        prepare_review_rows(
            {"source": (_raw("source", 0),)},
            per_source=True,
            seed=17,
            annotator=FixtureAnnotator(),
        )


def _reviewed_file(path: Path) -> bytes:
    rows = [
        {
            "source": "a",
            "confidence": 0.9,
            "accepted": True,
            "annotation_model": "model-a",
            "annotation_revision": "rev-a",
            "annotation_template_sha256": "1" * 64,
            "annotation_response_sha256": "2" * 64,
        },
        {
            "source": "b",
            "confidence": 0.8,
            "accepted": False,
            "annotation_model": "model-a",
            "annotation_revision": "rev-a",
            "annotation_template_sha256": "1" * 64,
            "annotation_response_sha256": "3" * 64,
        },
    ]
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return payload


def test_freeze_uses_exact_input_hash_and_deterministic_provenance(
    tmp_path: Path,
) -> None:
    reviewed = tmp_path / "reviewed.jsonl"
    exact_bytes = _reviewed_file(reviewed)

    artifact = freeze_reviewed_annotations(
        reviewed, target_precision=1.0, minimum_selected=1
    )

    assert artifact == {
        "accepted_count": 1,
        "annotation_provenance": [
            {
                "model": "model-a",
                "revision": "rev-a",
                "template_sha256": "1" * 64,
            }
        ],
        "artifact_version": "span_annotation_confidence.v1",
        "command": "calibrate_span_annotation_confidence.py freeze",
        "minimum_selected": 1,
        "precision": 1.0,
        "recall": 1.0,
        "reviewed_count": 2,
        "reviewed_sha256": hashlib.sha256(exact_bytes).hexdigest(),
        "selected_count": 1,
        "sources": ["a", "b"],
        "target_precision": 1.0,
        "threshold": 0.9,
    }


@pytest.mark.parametrize("accepted", [None, 1, 0, "true", [], {}])
def test_freeze_rejects_incomplete_or_non_boolean_reviews(
    tmp_path: Path, accepted: object
) -> None:
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text(
        canonical_json({"confidence": 0.9, "accepted": accepted}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="accepted must be an exact boolean"):
        freeze_reviewed_annotations(
            reviewed, target_precision=0.9, minimum_selected=1
        )


def test_frozen_artifact_write_is_atomic_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "artifact.json"
    artifact = {"threshold": 0.9}
    observed: list[Path] = []
    original_replace = Path.replace

    def observe_replace(source: Path, target: Path) -> Path:
        observed.append(source)
        assert source.parent == output.parent
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    write_frozen_artifact(output, artifact)

    assert observed
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_frozen_artifact(output, {"threshold": 0.8})
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_cli_mode_boundaries_match_documented_commands() -> None:
    prepare = parse_args(
        [
            "prepare",
            "--config",
            "config.yaml",
            "--per-source",
            "10",
            "--output",
            "review.jsonl",
        ]
    )
    assert prepare.mode == "prepare"
    assert prepare.per_source == 10

    freeze = parse_args(
        [
            "freeze",
            "--reviewed",
            "review.jsonl",
            "--target-precision",
            "0.95",
            "--minimum-selected",
            "20",
            "--output",
            "confidence.json",
        ]
    )
    assert freeze.mode == "freeze"
    assert freeze.minimum_selected == 20

    with pytest.raises(SystemExit):
        parse_args(["prepare", "--reviewed", "review.jsonl"])
    with pytest.raises(SystemExit):
        parse_args(["freeze", "--config", "config.yaml"])


def test_prepare_rows_can_be_written_as_canonical_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "review.jsonl"
    rows = prepare_review_rows(
        {"source": tuple(_raw("source", row) for row in range(2))},
        per_source=1,
        seed=17,
        annotator=FixtureAnnotator(),
    )
    _MODULE.write_prepared_rows(output, rows)

    assert read_jsonl(output) == rows
