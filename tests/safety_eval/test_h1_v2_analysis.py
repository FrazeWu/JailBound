from __future__ import annotations

import pytest

from benchmark.safety_eval.h1_v2_analysis import (
    PromptScoreCurve,
    PromptUtrCurve,
    build_analysis_contract,
    score_auc,
    source_stratified_score_association,
    source_stratified_association,
    utr_auc,
    validate_analysis_contract,
)


def _curve(source: str, prompt_id: str, fol: float, rates: tuple[float, ...]) -> PromptUtrCurve:
    return PromptUtrCurve(
        source=source,
        prompt_id=prompt_id,
        fol=fol,
        baseline_safe=True,
        radii=(0.1, 1.0, 10.0),
        unsafe_transition_rates=rates,
        accepted_direction_counts=(32, 32, 32),
    )


def test_utr_auc_integrates_rates_over_log_radius() -> None:
    curve = _curve("jailbound", "opaque:one", 1.0, (0.0, 0.5, 1.0))

    assert utr_auc(curve) == pytest.approx(0.5)


def test_utr_auc_rejects_non_safe_or_incomplete_curves() -> None:
    unsafe = PromptUtrCurve(
        source="jailbound", prompt_id="opaque:unsafe", fol=1.0, baseline_safe=False,
        radii=(0.1, 1.0), unsafe_transition_rates=(0.0, 1.0), accepted_direction_counts=(32, 32),
    )
    incomplete = PromptUtrCurve(
        source="jailbound", prompt_id="opaque:incomplete", fol=1.0, baseline_safe=True,
        radii=(0.1, 1.0), unsafe_transition_rates=(0.0, 1.0), accepted_direction_counts=(32, 31),
    )

    with pytest.raises(ValueError, match="baseline-safe"):
        utr_auc(unsafe)
    with pytest.raises(ValueError, match="32 accepted"):
        utr_auc(incomplete)


def test_score_auc_integrates_mean_unsafe_scores_over_log_radius() -> None:
    curve = PromptScoreCurve(
        source="jailbound",
        prompt_id="opaque:score",
        fol=1.0,
        radii=(0.1, 1.0, 10.0),
        mean_unsafe_scores=(0.0, 0.5, 1.0),
        accepted_direction_counts=(32, 32, 32),
    )

    assert score_auc(curve) == pytest.approx(0.5)


def test_score_auc_rejects_invalid_scores_and_incomplete_directions() -> None:
    invalid_score = PromptScoreCurve(
        source="jailbound",
        prompt_id="opaque:invalid-score",
        fol=1.0,
        radii=(0.1, 1.0),
        mean_unsafe_scores=(0.1, 1.2),
        accepted_direction_counts=(32, 32),
    )
    incomplete = PromptScoreCurve(
        source="jailbound",
        prompt_id="opaque:incomplete-score",
        fol=1.0,
        radii=(0.1, 1.0),
        mean_unsafe_scores=(0.1, 0.2),
        accepted_direction_counts=(32, 31),
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        score_auc(invalid_score)
    with pytest.raises(ValueError, match="32 accepted"):
        score_auc(incomplete)


def test_source_stratified_association_detects_positive_fol_utr_relation() -> None:
    curves = tuple(
        _curve(source, f"{source}:opaque:{index}", float(index), (0.0, index / 16, index / 8))
        for source in ("jailbound", "s_eval")
        for index in range(1, 8)
    )

    result = source_stratified_association(curves, permutations=999, bootstrap_replicates=999, seed=7)

    assert result.source_rho == {"jailbound": 1.0, "s_eval": 1.0}
    assert result.mean_rho == pytest.approx(1.0)
    assert result.permutation_pvalue < 0.05
    assert result.bootstrap_lower > 0.0


def test_source_stratified_score_association_detects_positive_relation() -> None:
    curves = tuple(
        PromptScoreCurve(
            source=source,
            prompt_id=f"{source}:opaque:score:{index}",
            fol=float(index),
            radii=(0.1, 1.0),
            mean_unsafe_scores=(index / 16, index / 8),
            accepted_direction_counts=(32, 32),
        )
        for source in ("jailbound", "s_eval")
        for index in range(1, 8)
    )

    result = source_stratified_score_association(
        curves, permutations=999, bootstrap_replicates=999, seed=7
    )

    assert result.source_rho == {"jailbound": 1.0, "s_eval": 1.0}
    assert result.permutation_pvalue < 0.05


def test_source_stratified_association_permutation_is_deterministic_and_keeps_sources_separate() -> None:
    curves = tuple(
        _curve(source, f"{source}:opaque:{index}", float(index), (0.1, (index % 3) / 4, (index % 2) / 2))
        for source in ("jailbound", "s_eval")
        for index in range(1, 8)
    )

    first = source_stratified_association(curves, permutations=199, bootstrap_replicates=199, seed=11)
    second = source_stratified_association(tuple(reversed(curves)), permutations=199, bootstrap_replicates=199, seed=11)

    assert first == second
    assert 0.0 <= first.permutation_pvalue <= 1.0
    assert first.bootstrap_lower <= first.bootstrap_upper


def test_analysis_contract_rejects_changed_selected_ids_or_radii() -> None:
    config_hash = "a" * 64
    selected_ids = (("jailbound", "opaque:1"), ("s_eval", "opaque:2"))
    contract = build_analysis_contract(
        config_hash=config_hash, selected_ids=selected_ids, local_radii=(0.1, 0.2),
    )

    validate_analysis_contract(
        contract, config_hash=config_hash, selected_ids=selected_ids, local_radii=(0.1, 0.2),
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_analysis_contract(
            contract, config_hash=config_hash, selected_ids=selected_ids, local_radii=(0.1, 0.3),
        )
