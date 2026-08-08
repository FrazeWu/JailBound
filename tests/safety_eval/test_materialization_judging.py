from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from benchmark.safety_eval.materialization_ablation import MaterializationPair
from benchmark.safety_eval.materialization_judging import (
    PairedJudgmentRecord,
    judge_materialization_pair,
    merge_paired_judgments,
)


def _pair() -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": "harmbench",
        "sample_id": "harmbench:1",
        "branch": "High-Value",
        "optimization_checkpoint": 100,
        "state_sha256": "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": "initial",
        "reference_intent": "intent",
        "continuous_response": "continuous response",
        "materialized_text": "materialized prompt",
        "materialized_response": "materialized response",
        "editable_projected_token_ids": [1],
        "projected_token_ids": [1],
        "retokenized_token_ids": [1],
        "projection_cosines": [0.9],
        "roundtrip_exact_match": True,
        "projected_length": 1,
        "retokenized_length": 1,
        "max_new_tokens": 512,
        "status": "complete",
        "error": None,
        "judgments": {},
    })


class _FakeJudge:
    key = "octopus_seval_14b"
    revision = "fixture-revision"

    def __init__(self, *, fail_materialized: bool = False) -> None:
        self.fail_materialized = fail_materialized
        self.calls: list[dict[str, object]] = []

    def judge_response(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        materialized = kwargs["response_text"] == "materialized response"
        if materialized and self.fail_materialized:
            return SimpleNamespace(
                status="failed",
                unsafe_score=0.0,
                unsafe_label=False,
                raw_output="",
                failure_reason="fixture failure",
                judge_key=self.key,
                judge_revision=self.revision,
                threshold=kwargs["threshold"],
            )
        return SimpleNamespace(
            status="complete",
            unsafe_score=0.8 if materialized else 0.2,
            unsafe_label=materialized,
            raw_output="unsafe" if materialized else "safe",
            failure_reason=None,
            judge_key=self.key,
            judge_revision=self.revision,
            threshold=kwargs["threshold"],
        )


def test_judge_materialization_pair_uses_same_provenance_for_both_conditions() -> None:
    judge = _FakeJudge()

    result = judge_materialization_pair(
        _pair(),
        judge=judge,
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    )

    assert result.status == "complete"
    assert result.judge_key == "octopus_seval_14b"
    assert result.judge_revision == "fixture-revision"
    assert result.threshold == pytest.approx(0.5)
    assert result.continuous_score == pytest.approx(0.2)
    assert result.materialized_score == pytest.approx(0.8)
    assert result.continuous is False
    assert result.materialized is True
    assert [call["response_text"] for call in judge.calls] == [
        "continuous response",
        "materialized response",
    ]
    assert {call["method"] for call in judge.calls} == {
        "materialization_continuous:High-Value",
        "materialization_materialized:High-Value",
    }
    assert all(call["checkpoint"] == 100 for call in judge.calls)


def test_judge_materialization_pair_retains_terminal_failure() -> None:
    result = judge_materialization_pair(
        _pair(),
        judge=_FakeJudge(fail_materialized=True),
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    )

    assert result.status == "failed"
    assert result.continuous is None
    assert result.materialized is None
    assert result.error == "materialized: fixture failure"


def test_paired_judgment_rejects_inconsistent_complete_provenance() -> None:
    payload = judge_materialization_pair(
        _pair(),
        judge=_FakeJudge(),
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    ).model_dump(mode="json")
    payload["judge_revision"] = ""

    with pytest.raises(ValidationError, match="judge_revision"):
        PairedJudgmentRecord.model_validate(payload)


def test_paired_judgment_rejects_label_that_disagrees_with_score_threshold() -> None:
    payload = judge_materialization_pair(
        _pair(),
        judge=_FakeJudge(),
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    ).model_dump(mode="json")
    payload["continuous"] = True

    with pytest.raises(ValidationError, match="score and threshold"):
        PairedJudgmentRecord.model_validate(payload)


def test_merge_paired_judgments_replaces_failed_rows_but_not_complete_rows(tmp_path: Path) -> None:
    pair = _pair()
    failed = judge_materialization_pair(
        pair,
        judge=_FakeJudge(fail_materialized=True),
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    )
    complete = judge_materialization_pair(
        pair,
        judge=_FakeJudge(),
        run_id="materialization:test",
        config_hash="b" * 64,
        target_key="qwen2_5_7b",
        target_revision="local",
        threshold=0.5,
    )
    path = tmp_path / "paired_judgments.jsonl"

    assert merge_paired_judgments(path, [failed]) == (1, 0)
    assert merge_paired_judgments(path, [complete]) == (0, 1)
    assert merge_paired_judgments(path, [failed]) == (0, 0)

    rows = [PairedJudgmentRecord.model_validate_json(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0].status == "complete"
