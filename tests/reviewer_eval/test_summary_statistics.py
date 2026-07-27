from __future__ import annotations

import pytest

from benchmark.reviewer_eval.analysis import (
    BootstrapPrompt,
    JudgePair,
    bootstrap_prompt_ids,
    exact_mcnemar_pvalue,
    judge_agreement,
    threshold_rank_sensitivity,
)
from benchmark.reviewer_eval.metrics import Outcome, Rate, compute_asr, summarize_outcomes


def test_outcome_summary_keeps_failed_examples_in_itt_and_reports_execution_only_rate() -> None:
    outcomes = [Outcome.complete(True), Outcome.complete(False), Outcome.failed("generation")]

    summary = summarize_outcomes(outcomes)

    assert summary.itt_asr == Rate(1, 3)
    assert summary.execution_asr == Rate(1, 2)
    assert summary.failed_count == 1
    assert summary.itt_asr.display == "1 / 3 (33.33%)"
    assert summary.execution_asr.display == "1 / 2 (50.00%)"


def test_count_metrics_reject_frozen_pdf_rows() -> None:
    frozen = Outcome(unsafe=None, failure_kind="context", provenance="frozen_pdf")

    with pytest.raises(ValueError, match="frozen_pdf"):
        compute_asr([frozen])


def test_exact_mcnemar_pvalue_uses_only_discordant_pairs() -> None:
    assert exact_mcnemar_pvalue(method_only=3, baseline_only=0) == pytest.approx(0.25)
    assert exact_mcnemar_pvalue(method_only=0, baseline_only=0) == pytest.approx(1.0)


def test_judge_agreement_reports_raw_agreement_disagreement_and_kappa() -> None:
    result = judge_agreement([
        JudgePair("response:1", True, True),
        JudgePair("response:2", True, False),
        JudgePair("response:3", False, False),
        JudgePair("response:4", False, True),
    ])

    assert result.agreement == pytest.approx(0.5)
    assert result.disagreement == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.0)
    assert result.denominator == 4


def test_threshold_rank_sensitivity_preserves_exact_rate_ties_before_display_rounding() -> None:
    sensitivity = threshold_rank_sensitivity({
        0.4: {"method_a": Rate(1, 2), "method_b": Rate(2, 4), "method_c": Rate(1, 4)},
        0.6: {"method_a": Rate(1, 4), "method_b": Rate(3, 4), "method_c": Rate(1, 4)},
    })

    assert sensitivity[0].threshold == pytest.approx(0.4)
    assert sensitivity[0].ranks == (("method_a", 1), ("method_b", 1), ("method_c", 3))
    assert sensitivity[1].ranks == (("method_b", 1), ("method_a", 2), ("method_c", 2))


def test_stratified_bootstrap_resamples_prompt_ids_deterministically_within_strata() -> None:
    prompts = [
        BootstrapPrompt("prompt:1", "source_a|risk_1"),
        BootstrapPrompt("prompt:2", "source_a|risk_1"),
        BootstrapPrompt("prompt:3", "source_b|risk_2"),
        BootstrapPrompt("prompt:4", "source_b|risk_2"),
    ]

    first = bootstrap_prompt_ids(prompts, replicates=5, seed=7)
    second = bootstrap_prompt_ids(list(reversed(prompts)), replicates=5, seed=7)
    by_id = {prompt.prompt_id: prompt.stratum for prompt in prompts}

    assert first == second
    assert len(first) == 5
    assert all(len(draw) == 4 for draw in first)
    assert all(sum(by_id[prompt_id] == "source_a|risk_1" for prompt_id in draw) == 2 for draw in first)
    assert all(sum(by_id[prompt_id] == "source_b|risk_2" for prompt_id in draw) == 2 for draw in first)
    with pytest.raises(ValueError, match="frozen_pdf"):
        bootstrap_prompt_ids([BootstrapPrompt("pdf:1", "context", provenance="frozen_pdf")], replicates=1, seed=7)
