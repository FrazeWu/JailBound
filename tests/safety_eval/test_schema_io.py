from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from benchmark.safety_eval.io import (
    JsonlLedger,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
)
from benchmark.safety_eval.schema import (
    BenchmarkExample,
    CellKey,
    ComputeCounters,
    EditableSpan,
    EditableSpanRole,
    FailureKind,
    OptimizationRecord,
    RecordStatus,
    TransportType,
    V2BenchmarkExample,
)


def example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="advbench:000000:8b42",
        source="advbench",
        source_file="AdvBench/harmful_behaviors.csv",
        source_row=0,
        source_sha256="a" * 64,
        intent="Example intent.",
        attack_text="Example request.",
        target_text="Sure, here is",
        source_risk_label=None,
        source_attack_label="direct_request",
        risk_category="unsafe_unethical",
        threat_domain="education",
        attack_type="direct_request",
        language="en",
        selection_stratum="unsafe_unethical|direct_request",
        selection_seed=20260725,
        prompt_sha256="b" * 64,
        preprocessing=("normalized_newlines",),
    )


def v2_example_payload() -> dict[str, object]:
    return {
        **example().model_dump(mode="json"),
        "schema_version": "reviewer_eval.v2",
        "intent_sha256": "c" * 64,
        "editable_spans": [
            {
                "start": 0,
                "end": 7,
                "quote": "Example",
                "role": "seed_intent",
                "confidence": 0.9,
                "rationale": "The seed intent is stated directly.",
            },
            {
                "start": 8,
                "end": 15,
                "quote": "request",
                "role": "harmful_payload",
                "confidence": 0.75,
                "rationale": "This is the payload-bearing phrase.",
            },
        ],
        "annotator_model": "de-aligned-annotator",
        "annotator_revision": "immutable-revision",
        "annotation_template_sha256": "d" * 64,
        "annotation_response_sha256": "e" * 64,
        "annotation_confidence": 0.75,
    }


def test_v2_benchmark_example_round_trips_complete_record() -> None:
    record = V2BenchmarkExample.model_validate(v2_example_payload())

    assert record.schema_version == "reviewer_eval.v2"
    assert record.editable_spans == (
        EditableSpan(
            start=0,
            end=7,
            quote="Example",
            role=EditableSpanRole.seed_intent,
            confidence=0.9,
            rationale="The seed intent is stated directly.",
        ),
        EditableSpan(
            start=8,
            end=15,
            quote="request",
            role=EditableSpanRole.harmful_payload,
            confidence=0.75,
            rationale="This is the payload-bearing phrase.",
        ),
    )
    serialized = record.model_dump_json()
    assert V2BenchmarkExample.model_validate_json(serialized) == record
    assert TransportType.text.value == "text"
    assert TransportType.embedding.value == "embedding"
    assert {
        FailureKind.annotation.value,
        FailureKind.token_mapping.value,
        FailureKind.objective.value,
        FailureKind.transport.value,
    } == {"annotation", "token_mapping", "objective", "transport"}


@pytest.mark.parametrize(
    "span_update",
    [
        {"start": -1},
        {"end": 0},
        {"start": 7, "end": 7},
        {"quote": ""},
        {"confidence": -0.01},
        {"confidence": 1.01},
    ],
    ids=[
        "negative-start",
        "end-before-start",
        "empty-range",
        "empty-quote",
        "confidence-below-zero",
        "confidence-above-one",
    ],
)
def test_editable_span_rejects_malformed_fields(span_update: dict[str, object]) -> None:
    payload = {
        "start": 0,
        "end": 7,
        "quote": "Example",
        "role": "seed_intent",
        "confidence": 0.9,
        "rationale": "The seed intent is stated directly.",
        **span_update,
    }

    with pytest.raises(ValueError):
        EditableSpan.model_validate(payload)


def test_v2_benchmark_example_rejects_empty_spans() -> None:
    payload = v2_example_payload()
    payload["editable_spans"] = []

    with pytest.raises(ValueError):
        V2BenchmarkExample.model_validate(payload)


def test_v2_benchmark_example_rejects_exact_quote_mismatch() -> None:
    payload = v2_example_payload()
    spans = list(payload["editable_spans"])
    spans[0] = {**spans[0], "quote": "example"}
    payload["editable_spans"] = spans

    with pytest.raises(ValueError, match="quote"):
        V2BenchmarkExample.model_validate(payload)


@pytest.mark.parametrize(
    "second_span",
    [
        {
            "start": 0,
            "end": 7,
            "quote": "Example",
            "role": "attack_instruction",
            "confidence": 0.75,
            "rationale": "Duplicates an earlier span.",
        },
        {
            "start": 6,
            "end": 15,
            "quote": "e request",
            "role": "attack_instruction",
            "confidence": 0.75,
            "rationale": "Overlaps an earlier span.",
        },
    ],
    ids=["unordered", "overlapping"],
)
def test_v2_benchmark_example_rejects_unordered_or_overlapping_spans(
    second_span: dict[str, object],
) -> None:
    payload = v2_example_payload()
    payload["editable_spans"] = [payload["editable_spans"][0], second_span]

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        V2BenchmarkExample.model_validate(payload)


def test_v2_benchmark_example_rejects_span_end_past_attack_text() -> None:
    payload = v2_example_payload()
    spans = list(payload["editable_spans"])
    spans[-1] = {**spans[-1], "end": len(str(payload["attack_text"])) + 1}
    payload["editable_spans"] = spans

    with pytest.raises(ValueError, match="within attack_text"):
        V2BenchmarkExample.model_validate(payload)


def test_v2_benchmark_example_rejects_record_confidence_above_span_minimum() -> None:
    payload = v2_example_payload()
    payload["annotation_confidence"] = 0.9

    with pytest.raises(ValueError, match="minimum span confidence"):
        V2BenchmarkExample.model_validate(payload)


def test_records_forbid_unknown_fields() -> None:
    payload = example().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        BenchmarkExample.model_validate(payload)


def test_records_reject_invalid_sha256() -> None:
    payload = example().model_dump(mode="json")
    payload["source_sha256"] = "A" * 64
    with pytest.raises(ValueError):
        BenchmarkExample.model_validate(payload)


def test_cell_id_changes_when_any_identity_field_changes() -> None:
    base = CellKey(
        dataset_source="advbench",
        sample_manifest_hash="a" * 64,
        optimization_method="pez",
        optimization_budget="updates=100",
        surrogate_model_revision="qwen-rev",
        target_model_revision="qwen-rev",
        decoding_config_hash="b" * 64,
        judge_revision="octopus-rev",
        judge_threshold=0.5,
    )
    changed = base.model_copy(update={"judge_threshold": 0.6})
    assert base.cell_id != changed.cell_id


def test_model_copy_revalidates_updated_records() -> None:
    record = OptimizationRecord(
        schema_version="reviewer_eval.v1",
        run_id="run:example",
        config_hash="a" * 64,
        git_revision="1234567890abcdef",
        cell_id="cell:example",
        sample_id="advbench:000000:example",
        source="advbench",
        method="pez",
        checkpoint=25,
        random_seed=20260725,
        status=RecordStatus.complete,
        failure_kind=None,
        failure_reason=None,
        state_path=None,
        representation="token_ids",
        attack_loss=0.25,
        fol=0.5,
        internal_margin=0.1,
        materialized_prompt=None,
        counters=ComputeCounters(),
    )

    with pytest.raises(ValueError, match="failed records require"):
        record.model_copy(update={"status": RecordStatus.failed})

    updated = record.model_copy(
        update={
            "status": "failed",
            "failure_kind": "generation",
            "failure_reason": "test failure",
        }
    )
    assert updated.status is RecordStatus.failed
    assert updated.failure_kind is not None
    assert updated.failure_kind.value == "generation"


def test_records_accept_serialized_enum_values() -> None:
    payload = {
        "schema_version": "reviewer_eval.v1",
        "run_id": "run:example",
        "config_hash": "a" * 64,
        "git_revision": "1234567890abcdef",
        "cell_id": "cell:example",
        "sample_id": "advbench:000000:example",
        "source": "advbench",
        "method": "pez",
        "checkpoint": 25,
        "random_seed": 20260725,
        "status": "failed",
        "failure_kind": "generation",
        "failure_reason": "test failure",
        "state_path": None,
        "representation": "token_ids",
        "attack_loss": None,
        "fol": None,
        "internal_margin": None,
        "materialized_prompt": None,
        "counters": {},
    }

    record = OptimizationRecord.model_validate(payload)

    assert record.status is RecordStatus.failed
    assert record.failure_kind is not None
    assert record.failure_kind.value == "generation"


def test_jsonl_ledger_does_not_duplicate_terminal_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    ledger = JsonlLedger(path, key_fields=("cell_id", "sample_id", "checkpoint"))
    record = {
        "cell_id": "cell",
        "sample_id": "sample",
        "checkpoint": 25,
        "status": RecordStatus.complete,
    }
    assert ledger.append_once(record) is True
    assert ledger.append_once(record) is False
    assert read_jsonl(path) == [{**record, "status": "complete"}]


def test_jsonl_ledger_rejects_conflicting_duplicate_key(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "records.jsonl", key_fields=("id",))
    assert ledger.append_once({"id": "same", "value": 1})
    with pytest.raises(ValueError, match="conflicting payload"):
        ledger.append_once({"id": "same", "value": 2})


def test_jsonl_ledger_appends_after_a_valid_unterminated_final_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id":"first"}', encoding="utf-8")

    ledger = JsonlLedger(path, key_fields=("id",))
    assert ledger.append_once({"id": "second"})
    assert read_jsonl(path) == [{"id": "first"}, {"id": "second"}]


def test_truncated_final_line_is_preserved_and_recorded_before_repair(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id":"complete"}\n{"id":"truncated"')

    ledger = JsonlLedger(path, key_fields=("id",))
    assert ledger.append_once({"id": "next"})

    assert read_jsonl(path) == [{"id": "complete"}, {"id": "next"}]
    corrupt_path = tmp_path / "records.jsonl.corrupt"
    assert corrupt_path.read_bytes() == b'{"id":"truncated"'
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repairs"] == [
        {
            "corrupt_file": "records.jsonl.corrupt",
            "kind": "truncated_jsonl_final_line",
            "ledger": "records.jsonl",
            "sha256": hashlib.sha256(b'{"id":"truncated"').hexdigest(),
        }
    ]


def test_malformed_nonfinal_jsonl_line_aborts(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id":"complete"}\nnot-json\n{"id":"later"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed non-final"):
        JsonlLedger(path, key_fields=("id",))


def test_atomic_write_json_replaces_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"
    destination.parent.mkdir()
    destination.write_text('{"old":true}\n', encoding="utf-8")

    atomic_write_json(destination, {"b": 2, "a": 1})

    assert destination.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert list(destination.parent.glob("*.tmp")) == []


def test_atomic_write_jsonl_replaces_existing_file_with_canonical_rows(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.jsonl"
    destination.parent.mkdir()
    destination.write_text('{"old":true}\n', encoding="utf-8")

    atomic_write_jsonl(destination, [{"b": 2, "a": 1}, {"id": "next"}])

    assert destination.read_text(encoding="utf-8") == '{"a":1,"b":2}\n{"id":"next"}\n'
    assert list(destination.parent.glob("*.tmp")) == []


def test_canonical_hash_ignores_dictionary_insertion_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
