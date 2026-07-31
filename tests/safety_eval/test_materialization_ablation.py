from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.safety_eval.materialization_ablation import (
    Branch,
    MaterializationPair,
    index_pairs,
    summarize_pairs,
    summarize_roundtrips,
    write_pair_summaries,
)


def _pair(
    *,
    sample_id: str = "sample:1",
    source: str = "harmbench",
    branch: str = "High-Value",
    continuous: bool = True,
    materialized: bool = True,
    roundtrip_exact: bool = True,
    status: str = "complete",
    error: str | None = None,
) -> MaterializationPair:
    payload = {
        "source": source,
        "sample_id": sample_id,
        "branch": branch,
        "optimization_checkpoint": 100,
        "state_sha256": "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": "initial prompt",
        "reference_intent": "reference intent",
        "continuous_response": "continuous response",
        "materialized_text": "materialized prompt",
        "materialized_response": "materialized response",
        "editable_projected_token_ids": [4, 5],
        "projected_token_ids": [4, 8, 5],
        "retokenized_token_ids": [4, 8, 5] if roundtrip_exact else [4, 8, 6, 5],
        "projection_cosines": [0.9, 0.8],
        "roundtrip_exact_match": roundtrip_exact,
        "projected_length": 3,
        "retokenized_length": 3 if roundtrip_exact else 4,
        "max_new_tokens": 512,
        "status": status,
        "error": error,
        "judgments": {
            "octopus": {
                "continuous": continuous,
                "materialized": materialized,
            }
        },
    }
    if status == "failed":
        payload.update(
            state_sha256=None,
            continuous_response="",
            materialized_response="",
            materialized_text="",
            editable_projected_token_ids=[],
            projected_token_ids=[],
            retokenized_token_ids=[],
            projection_cosines=[],
            projected_length=0,
            retokenized_length=0,
            roundtrip_exact_match=False,
            judgments={},
        )
    return MaterializationPair.model_validate(payload)


def test_index_pairs_rejects_duplicate_canonical_keys() -> None:
    pair = _pair()
    with pytest.raises(ValueError, match="duplicate materialization pair"):
        index_pairs([pair, pair])


def test_pair_rejects_inconsistent_projection_evidence() -> None:
    payload = _pair().model_dump(mode="json")
    payload["projection_cosines"] = [0.9]
    with pytest.raises(ValidationError, match="projection cosine count"):
        MaterializationPair.model_validate(payload)


def test_pair_requires_error_only_for_failed_records() -> None:
    with pytest.raises(ValidationError, match="failed pair requires an error"):
        _pair(status="failed", error=None)
    with pytest.raises(ValidationError, match="complete pair cannot contain an error"):
        _pair(error="unexpected")


def test_complete_pair_requires_state_hash_but_failed_pair_may_lack_it() -> None:
    payload = _pair().model_dump(mode="json")
    payload["state_sha256"] = None
    with pytest.raises(ValidationError, match="complete pair requires a state hash"):
        MaterializationPair.model_validate(payload)

    assert _pair(status="failed", error="state is unavailable").state_sha256 is None


def test_pair_rejects_roundtrip_flag_that_disagrees_with_token_ids() -> None:
    payload = _pair().model_dump(mode="json")
    payload["retokenized_token_ids"] = [4, 8, 6]
    with pytest.raises(ValidationError, match="round-trip flag"):
        MaterializationPair.model_validate(payload)


def test_summarize_pairs_uses_exact_paired_formulas() -> None:
    pairs = [
        _pair(sample_id="a", continuous=True, materialized=True),
        _pair(sample_id="b", continuous=True, materialized=False),
        _pair(sample_id="c", continuous=False, materialized=True),
        _pair(sample_id="d", continuous=False, materialized=False),
    ]

    summaries = summarize_pairs(pairs, judge_key="octopus")
    assert [(row.source, row.branch) for row in summaries] == [
        ("harmbench", Branch.high_value),
        ("Overall", "All"),
    ]
    row = next(summary for summary in summaries if summary.source == "harmbench")

    assert row.branch is Branch.high_value
    assert row.valid_pairs == 4
    assert row.failed_pairs == 0
    assert row.continuous_unsafe == row.materialized_unsafe == 2
    assert row.continuous_asr == pytest.approx(0.5)
    assert row.materialized_asr == pytest.approx(0.5)
    assert row.difference_pp == pytest.approx(0.0)
    assert row.unsafe_retention == pytest.approx(0.5)
    assert row.label_agreement == pytest.approx(0.5)


def test_summarize_pairs_emits_only_cross_strata_and_explicit_overall() -> None:
    pairs = [
        _pair(sample_id="h-hv", source="harmbench", branch="High-Value"),
        _pair(sample_id="h-ss", source="harmbench", branch="Safety-Sensitivity"),
        _pair(sample_id="s-hv", source="s_eval", branch="High-Value"),
    ]

    summaries = summarize_pairs(pairs, judge_key="octopus")

    assert [(row.source, row.branch) for row in summaries] == [
        ("harmbench", Branch.high_value),
        ("harmbench", Branch.safety_sensitivity),
        ("s_eval", Branch.high_value),
        ("Overall", "All"),
    ]
    assert summaries[-1].total_pairs == 3


def test_summarize_pairs_preserves_failed_only_stratum_and_overall() -> None:
    failed = _pair(sample_id="failed", status="failed", error="generation failed")

    summaries = summarize_pairs([failed], judge_key="octopus")

    assert len(summaries) == 2
    for row in summaries:
        assert (row.total_pairs, row.valid_pairs, row.failed_pairs) == (1, 0, 1)
        assert (row.continuous_unsafe, row.materialized_unsafe, row.retained_unsafe, row.agreements) == (0, 0, 0, 0)
        assert row.continuous_asr is None
        assert row.materialized_asr is None
        assert row.difference_pp is None
        assert row.unsafe_retention is None
        assert row.label_agreement is None


def test_summarize_pairs_counts_failures_but_excludes_them_from_denominator() -> None:
    pairs = [
        _pair(sample_id="complete", continuous=False, materialized=False),
        _pair(sample_id="failed", status="failed", error="continuous generation failed"),
    ]

    row = next(summary for summary in summarize_pairs(pairs, judge_key="octopus") if summary.source == "harmbench")

    assert (row.total_pairs, row.valid_pairs, row.failed_pairs) == (2, 1, 1)
    assert row.unsafe_retention is None


def test_summarize_pairs_rejects_missing_judge_labels() -> None:
    pair = _pair()
    payload = pair.model_dump(mode="json")
    payload["judgments"] = {}
    unlabeled = MaterializationPair.model_validate(payload)

    with pytest.raises(ValueError, match="missing judge labels"):
        summarize_pairs([unlabeled], judge_key="octopus")


def test_roundtrip_summary_reports_counts_lengths_and_cosine() -> None:
    summaries = summarize_roundtrips(
        [_pair(sample_id="exact"), _pair(sample_id="changed", roundtrip_exact=False)]
    )
    row = next(summary for summary in summaries if summary.source == "harmbench")

    assert row.valid_pairs == 2
    assert row.exact_roundtrips == 1
    assert row.exact_roundtrip_rate == pytest.approx(0.5)
    assert row.mean_projected_length == pytest.approx(3.0)
    assert row.mean_retokenized_length == pytest.approx(3.5)
    assert row.mean_projection_cosine == pytest.approx(0.85)


def test_roundtrip_summary_emits_cross_strata_overall_and_failed_only_rows() -> None:
    complete = _pair(sample_id="complete", source="harmbench", branch="High-Value")
    failed = _pair(
        sample_id="failed",
        source="s_eval",
        branch="Safety-Sensitivity",
        status="failed",
        error="generation failed",
    )

    summaries = summarize_roundtrips([complete, failed])

    assert [(row.source, row.branch) for row in summaries] == [
        ("harmbench", Branch.high_value),
        ("s_eval", Branch.safety_sensitivity),
        ("Overall", "All"),
    ]
    failed_row = summaries[1]
    assert (failed_row.total_pairs, failed_row.valid_pairs, failed_row.failed_pairs) == (1, 0, 1)
    assert failed_row.exact_roundtrips == 0
    assert failed_row.exact_roundtrip_rate is None
    assert failed_row.mean_projected_length is None
    assert failed_row.mean_retokenized_length is None
    assert failed_row.mean_projection_cosine is None
    assert (summaries[-1].total_pairs, summaries[-1].valid_pairs, summaries[-1].failed_pairs) == (2, 1, 1)


def test_write_pair_summaries_writes_count_first_artifacts(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        _pair().model_dump_json() + "\n",
        encoding="utf-8",
    )

    pair_csv, roundtrip_csv, failures_json = write_pair_summaries(
        pairs_path,
        tmp_path / "analysis",
        judge_keys=("octopus",),
    )

    assert "Valid pairs" in pair_csv.read_text(encoding="utf-8")
    assert "Exact round trips" in roundtrip_csv.read_text(encoding="utf-8")
    assert failures_json.read_text(encoding="utf-8").strip() == "[]"
