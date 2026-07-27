from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from benchmark.reviewer_eval.io import (
    JsonlLedger,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
)
from benchmark.reviewer_eval.schema import (
    BenchmarkExample,
    CellKey,
    ComputeCounters,
    OptimizationRecord,
    RecordStatus,
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
