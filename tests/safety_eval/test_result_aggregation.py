from __future__ import annotations

import pytest

from benchmark.safety_eval.result_aggregation import (
    paired_judgment_differences,
    summarize_materializations,
    summarize_judgments,
    write_materialization_summaries,
    write_paired_judgment_differences,
)


def test_judgment_summary_retains_failed_rows_in_itt_denominator() -> None:
    rows = [
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "init", "threshold": 0.5, "status": "complete", "unsafe_label": True},
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "init", "threshold": 0.5, "status": "complete", "unsafe_label": False},
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "init", "threshold": 0.5, "status": "failed", "unsafe_label": False},
    ]

    summary = summarize_judgments(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row.itt_asr.display == "1 / 3 (33.33%)"
    assert row.execution_asr.display == "1 / 2 (50.00%)"
    assert row.failed_count == 1


def test_v2_judgment_summary_separates_text_and_embedding_transport() -> None:
    rows = [
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "text", "judge_key": "primary", "judge_revision": "judge:1", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "a" * 64, "target_revision": "target:1", "target_tokenizer_sha256": "b" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": True},
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "embedding", "judge_key": "primary", "judge_revision": "judge:1", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "c" * 64, "target_revision": "target:1", "target_tokenizer_sha256": "b" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": False},
    ]

    summaries = summarize_judgments(rows)

    assert {(row.transport, row.itt_asr.value) for row in summaries} == {("text", 1.0), ("embedding", 0.0)}


def test_v2_judgment_summary_separates_target_snapshot_history() -> None:
    rows = [
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "text", "judge_key": "primary", "judge_revision": "judge:1", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "a" * 64, "target_revision": "target:old", "target_tokenizer_sha256": "b" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": False},
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "text", "judge_key": "primary", "judge_revision": "judge:1", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "c" * 64, "target_revision": "target:new", "target_tokenizer_sha256": "d" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": True},
    ]

    summaries = summarize_judgments(rows)

    assert len(summaries) == 2
    assert {(row.target_revision, row.target_tokenizer_sha256) for row in summaries} == {
        ("target:old", "b" * 64), ("target:new", "d" * 64),
    }


def test_v2_judgment_summary_separates_judge_snapshot_history() -> None:
    rows = [
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "text", "judge_key": "primary", "judge_revision": "judge:old", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "a" * 64, "target_revision": "target:1", "target_tokenizer_sha256": "b" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": False},
        {"schema_version": "reviewer_eval.v2", "branch": "o_plus", "transport": "text", "judge_key": "primary", "judge_revision": "judge:new", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "s:1", "materialization_sha256": "a" * 64, "target_revision": "target:1", "target_tokenizer_sha256": "b" * 64, "threshold": 0.5, "status": "complete", "unsafe_label": True},
    ]

    summaries = summarize_judgments(rows)

    assert {row.judge_revision for row in summaries} == {"judge:old", "judge:new"}


def test_v2_judgment_csv_retains_execution_identity(tmp_path) -> None:
    from benchmark.safety_eval.result_aggregation import write_judgment_summaries

    path = tmp_path / "judgments" / "primary" / "qwen" / "source" / "method" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":"reviewer_eval.v2","branch":"o_plus","transport":"text","judge_key":"primary","judge_revision":"judge:1","target_key":"qwen","source":"source","method":"method","sample_id":"s:1","materialization_sha256":"' + "a" * 64 + '","target_revision":"target:1","target_tokenizer_sha256":"' + "b" * 64 + '","threshold":0.5,"status":"complete","unsafe_label":true}\n',
        encoding="utf-8",
    )

    output = write_judgment_summaries(tmp_path).read_text(encoding="utf-8")

    assert "Target revision" in output.splitlines()[0]
    assert "target:1" in output
    assert "a" * 64 in output


def test_paired_judgment_differences_uses_shared_sample_ids_and_itt_labels() -> None:
    rows = [
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "init", "sample_id": "a", "threshold": 0.5, "status": "complete", "unsafe_label": False},
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "init", "sample_id": "b", "threshold": 0.5, "status": "complete", "unsafe_label": False},
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "a", "threshold": 0.5, "status": "complete", "unsafe_label": True},
        {"judge_key": "primary", "target_key": "qwen", "source": "source", "method": "method", "sample_id": "b", "threshold": 0.5, "status": "failed", "unsafe_label": False},
    ]

    differences = paired_judgment_differences(rows)

    assert len(differences) == 1
    row = differences[0]
    assert (row.method, row.denominator, row.delta_itt) == ("method", 2, pytest.approx(0.5))
    assert (row.method_only, row.baseline_only, row.mcnemar_pvalue) == (1, 0, pytest.approx(1.0))


def test_write_paired_judgment_differences_writes_count_first_csv(tmp_path) -> None:
    ledger = tmp_path / "judgments" / "primary" / "qwen" / "source" / "init" / "records.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"judge_key":"primary","target_key":"qwen","source":"source","method":"init","sample_id":"a","threshold":0.5,"status":"complete","unsafe_label":false}\n',
        encoding="utf-8",
    )
    method = tmp_path / "judgments" / "primary" / "qwen" / "source" / "method" / "records.jsonl"
    method.parent.mkdir(parents=True)
    method.write_text(
        '{"judge_key":"primary","target_key":"qwen","source":"source","method":"method","sample_id":"a","threshold":0.5,"status":"complete","unsafe_label":true}\n',
        encoding="utf-8",
    )

    destination = write_paired_judgment_differences(tmp_path)

    assert destination.name == "paired_asr.csv"
    assert "1" in destination.read_text(encoding="utf-8")


def test_materialization_summary_excludes_failed_rows_from_numeric_means() -> None:
    rows = [
        {
            "source": "source", "method": "method", "checkpoint": 100,
            "status": "complete", "intent_preserved": True,
            "semantic_similarity_after": 0.9,
            "prefix_projection_cosine": 0.8,
            "seed_projection_cosine": 0.7,
        },
        {
            "source": "source", "method": "method", "checkpoint": 100,
            "status": "failed", "intent_preserved": False,
            "semantic_similarity_after": 0.0,
            "prefix_projection_cosine": None,
            "seed_projection_cosine": None,
        },
        {
            "source": "source", "method": "method", "checkpoint": 25,
            "status": "complete", "intent_preserved": True,
            "semantic_similarity_after": 0.1,
            "prefix_projection_cosine": 0.1,
            "seed_projection_cosine": 0.1,
        },
    ]

    summaries = summarize_materializations(rows)

    assert len(summaries) == 1
    summary = summaries[0]
    assert (summary.total_count, summary.complete_count, summary.intent_preserved_count) == (2, 1, 1)
    assert summary.failed_count == 1
    assert summary.semantic_similarity_mean == pytest.approx(0.9)
    assert summary.prefix_projection_cosine_mean == pytest.approx(0.8)


def test_v2_materialization_summary_uses_step_and_fidelity_fields() -> None:
    rows = [
        {
            "schema_version": "reviewer_eval.v2",
            "source": "source",
            "method": "method",
            "step": 100,
            "status": "complete",
            "frozen_positions_unchanged": True,
            "full_prompt_similarity": 1.0,
            "editable_span_similarity": 0.9,
        },
        {
            "schema_version": "reviewer_eval.v2",
            "source": "source",
            "method": "method",
            "step": 25,
            "status": "complete",
            "frozen_positions_unchanged": True,
            "full_prompt_similarity": 0.7,
            "editable_span_similarity": 0.6,
        },
    ]

    summaries = summarize_materializations(rows)

    assert len(summaries) == 1
    summary = summaries[0]
    assert (summary.total_count, summary.complete_count) == (1, 1)
    assert summary.frozen_positions_unchanged_count == 1
    assert summary.full_prompt_similarity_mean == pytest.approx(1.0)
    assert summary.editable_span_similarity_mean == pytest.approx(0.9)


def test_write_materialization_summaries_writes_final_checkpoint_table(tmp_path) -> None:
    ledger = tmp_path / "optimization" / "source" / "method" / "materialization.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"source":"source","method":"method","checkpoint":100,"status":"complete","intent_preserved":true,"semantic_similarity_after":0.9,"prefix_projection_cosine":0.8,"seed_projection_cosine":0.7}\n',
        encoding="utf-8",
    )

    destination = write_materialization_summaries(tmp_path)

    assert destination.name == "materialization_fidelity.csv"
    assert "0.9" in destination.read_text(encoding="utf-8")
