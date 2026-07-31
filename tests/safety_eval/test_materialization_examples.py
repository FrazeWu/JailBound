from __future__ import annotations

from benchmark.safety_eval.intent_preservation import (
    AnnotationMapping,
    FinalIntentLabel,
    FinalLabel,
    DriftReason,
)
from benchmark.safety_eval.materialization_ablation import Branch, MaterializationPair
from benchmark.safety_eval.materialization_examples import (
    build_example_cases,
    render_examples_markdown,
    select_compact_examples,
)


def _pair(source: str, sample_id: str, branch: Branch, text: str) -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": source,
        "sample_id": sample_id,
        "branch": branch.value,
        "optimization_checkpoint": 100,
        "state_sha256": "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": f"initial {sample_id}",
        "reference_intent": f"intent {sample_id}",
        "continuous_response": "continuous",
        "materialized_text": text,
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
        "judgments": {"octopus": {"continuous": True, "materialized": True}},
    })


def _mapping(annotation_id: str, pair: MaterializationPair) -> AnnotationMapping:
    import hashlib

    return AnnotationMapping(
        annotation_id=annotation_id,
        source=pair.source,
        sample_id=pair.sample_id,
        branch=pair.branch,
        optimization_checkpoint=pair.optimization_checkpoint,
        materialized_text_hash=hashlib.sha256(pair.materialized_text.encode()).hexdigest(),
    )


def test_full_selector_uses_first_sample_per_stratum_and_is_order_independent() -> None:
    pairs = [
        _pair("harmbench", "z", Branch.high_value, "long preserved"),
        _pair("harmbench", "a", Branch.high_value, "first preserved"),
        _pair("harmbench", "b", Branch.high_value, "drifted"),
    ]
    mappings = [_mapping(f"ann{i}", pair) for i, pair in enumerate(pairs)]
    labels = [
        FinalIntentLabel("ann0", FinalLabel.preserved, (), ""),
        FinalIntentLabel("ann1", FinalLabel.preserved, (), ""),
        FinalIntentLabel("ann2", FinalLabel.not_preserved, (DriftReason.action_changed,), "reviewed"),
    ]

    first = build_example_cases(pairs, mappings, labels, sources=("harmbench", "jailbound", "s_eval"))
    second = build_example_cases(reversed(pairs), reversed(mappings), reversed(labels), sources=("harmbench", "jailbound", "s_eval"))

    assert first == second
    selected = [slot.case for slot in first if slot.case is not None]
    assert [case.sample_id for case in selected] == ["a", "b"]
    assert len(first) == 12
    assert sum(slot.case is None for slot in first) == 10


def test_compact_selector_maximizes_source_and_branch_coverage_before_length() -> None:
    pairs = [
        _pair("harmbench", "a", Branch.high_value, "x"),
        _pair("jailbound", "b", Branch.safety_sensitivity, "some drift"),
        _pair("harmbench", "c", Branch.high_value, "y"),
    ]
    mappings = [_mapping(f"ann{i}", pair) for i, pair in enumerate(pairs)]
    labels = [
        FinalIntentLabel("ann0", FinalLabel.preserved, (), ""),
        FinalIntentLabel("ann1", FinalLabel.not_preserved, (DriftReason.target_changed,), "reviewed"),
        FinalIntentLabel("ann2", FinalLabel.not_preserved, (DriftReason.action_changed,), "reviewed"),
    ]
    slots = build_example_cases(pairs, mappings, labels, sources=("harmbench", "jailbound", "s_eval"))

    compact = select_compact_examples(slots)

    assert {case.sample_id for case in compact} == {"a", "b"}
    assert {case.final_label for case in compact} == {FinalLabel.preserved, FinalLabel.not_preserved}
    assert len({case.source for case in compact}) == 2
    assert len({case.branch for case in compact}) == 2


def test_markdown_contains_verbatim_fields_and_explicit_empty_strata() -> None:
    pair = _pair("harmbench", "a", Branch.high_value, "line one\nline two")
    slots = build_example_cases(
        [pair],
        [_mapping("ann", pair)],
        [FinalIntentLabel("ann", FinalLabel.preserved, (), "")],
        sources=("harmbench",),
    )

    markdown = render_examples_markdown(slots, title="Materialization Examples")

    assert "# Materialization Examples" in markdown
    assert "> line one\n> line two" in markdown
    assert "initial a" in markdown
    assert "intent a" in markdown
    assert "No case available" in markdown
