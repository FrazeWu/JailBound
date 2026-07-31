from __future__ import annotations

from benchmark.safety_eval.fol_runtime import (
    build_perturbation_schedule,
    build_h1_v2_direction_schedule,
    causal_perplexity,
    select_id_shard,
    select_schedule_shard,
    select_accepted_perturbations,
    select_fol_experiment,
    select_h1_v2_candidates,
    select_h1_v2_validation,
    select_first_accepted_directions,
    select_h1_v2_oom_recovery_ids,
    select_h1_v2_pending_recovery_ids,
    select_h1_v2_eligible_ids,
    h1_v2_recovery_spec,
)
from benchmark.safety_eval.manifest import FolCandidate
from benchmark.safety_eval.fol_runtime import ConfirmatoryFolCandidate
from benchmark.safety_eval.schema import BenchmarkExample
from benchmark.safety_eval.fol_runtime import select_fol_candidates
import hashlib
import pytest
import torch


def test_fol_selection_reserves_disjoint_validation_bands_and_radius_calibration() -> None:
    rows = [
        FolCandidate(
            sample_id=f"source:{index:03d}",
            source="source",
            fol=float(index),
            risk_category="risk_a",
            initial_label=False,
            attack_loss=1.0 if index < 18 or index >= 27 else 2.0,
            token_length=20,
            perplexity=5.0,
        )
        for index in range(45)
    ]

    selection = select_fol_experiment(rows, seed=20260725)

    assert selection.status == "ready"
    assert [len(selection.low), len(selection.middle), len(selection.high)] == [7, 3, 7]
    assert len(selection.radius_calibration) == 5
    assert selection.matching_caliper == pytest.approx(0.5)
    assert len(selection.matching_distances) == 7
    selected = set(selection.low) | set(selection.middle) | set(selection.high)
    assert len(selected) == 17
    assert selected.isdisjoint(selection.radius_calibration)


def test_fol_candidate_selection_excludes_main_matrix_ids() -> None:
    def example(index: int) -> BenchmarkExample:
        attack_text = f"fixture {index}"
        return BenchmarkExample(
            example_id=f"source:{index:03d}", source="jailbound", source_file="fixture.jsonl",
            source_row=index, source_sha256="a" * 64, intent=f"intent {index}",
            attack_text=attack_text, target_text=None, source_risk_label=None,
            source_attack_label="direct_request", risk_category=f"risk_{index % 3}",
            threat_domain=f"domain_{index % 2}", attack_type=f"type_{index % 2}",
            language="en", selection_stratum=f"risk_{index % 3}|type_{index % 2}",
            selection_seed=20260725,
            prompt_sha256=hashlib.sha256(attack_text.encode()).hexdigest(), preprocessing=(),
        )

    rows = [example(index) for index in range(70)]
    selected = select_fol_candidates(rows, excluded_ids={row.example_id for row in rows[:17]}, seed=20260725)

    assert len(selected) == 45
    assert not {row.example_id for row in selected} & {row.example_id for row in rows[:17]}


def test_h1_v2_candidate_and_validation_selection_are_disjoint_and_deterministic() -> None:
    def example(index: int) -> BenchmarkExample:
        attack_text = f"fixture {index}"
        return BenchmarkExample(
            example_id=f"source:{index:03d}", source="jailbound", source_file="fixture.jsonl",
            source_row=index, source_sha256="a" * 64, intent=f"intent {index}",
            attack_text=attack_text, target_text=None, source_risk_label=None,
            source_attack_label="direct_request", risk_category="risk_a",
            threat_domain="domain_a", attack_type="type_a", language="en",
            selection_stratum="risk_a|type_a", selection_seed=20260725,
            prompt_sha256=hashlib.sha256(attack_text.encode()).hexdigest(), preprocessing=(),
        )

    rows = [example(index) for index in range(120)]
    exploratory = {row.example_id for row in rows[:17]}
    candidates = select_h1_v2_candidates(rows, excluded_ids=exploratory, seed=20260725)
    numeric = tuple(
        ConfirmatoryFolCandidate(
            sample_id=row.example_id, source=row.source, fol=float(index), risk_category="risk_a",
            attack_loss=1.0, token_length=20, perplexity=5.0, baseline_safe=True,
        )
        for index, row in enumerate(candidates)
    )

    first = select_h1_v2_validation(numeric, exploratory_ids=exploratory, seed=20260725)
    second = select_h1_v2_validation(tuple(reversed(numeric)), exploratory_ids=exploratory, seed=20260725)

    assert len(candidates) == 81
    assert not {row.example_id for row in candidates} & exploratory
    assert first == second
    assert first.status == "ready"
    assert [len(first.low), len(first.middle), len(first.high), len(first.reserves)] == [17, 3, 17, 4]
    assert len(set((*first.low, *first.middle, *first.high, *first.reserves))) == 41


def test_h1_v2_selection_is_inconclusive_without_41_baseline_safe_candidates() -> None:
    rows = tuple(
        ConfirmatoryFolCandidate(
            sample_id=f"source:{index:03d}", source="jailbound", fol=float(index), risk_category="risk_a",
            attack_loss=1.0, token_length=20, perplexity=5.0, baseline_safe=index < 40,
        )
        for index in range(81)
    )

    selection = select_h1_v2_validation(rows, exploratory_ids=set(), seed=20260725)

    assert selection.status == "inconclusive"


def test_h1_v2_selection_relaxes_risk_matching_only_after_strict_matching_is_infeasible() -> None:
    rows = tuple(
        ConfirmatoryFolCandidate(
            sample_id=f"source:{index:03d}", source="jailbound", fol=float(index),
            risk_category="low_risk" if index < 18 else "high_risk",
            attack_loss=1.0, token_length=20, perplexity=5.0, baseline_safe=True,
        )
        for index in range(41)
    )

    selection = select_h1_v2_validation(rows, exploratory_ids=set(), seed=20260725)

    assert selection.status == "ready"
    assert selection.risk_category_matching is False
    assert [len(selection.low), len(selection.middle), len(selection.high)] == [17, 3, 17]


def test_h1_v2_selection_uses_unmatched_endpoints_only_after_covariate_matching_fails() -> None:
    rows = tuple(
        ConfirmatoryFolCandidate(
            sample_id=f"source:{index:03d}", source="jailbound", fol=float(index),
            risk_category="low_risk" if index < 18 else "high_risk",
            attack_loss=1.0, token_length=20,
            perplexity=100000.0 if index in {39, 40} else 5.0,
            baseline_safe=True,
        )
        for index in range(41)
    )

    selection = select_h1_v2_validation(rows, exploratory_ids=set(), seed=20260725)

    assert selection.status == "ready"
    assert selection.matching_mode == "unmatched_endpoints"


def test_h1_v2_schedule_keeps_only_first_32_accepted_or_marks_group_insufficient() -> None:
    schedule = build_h1_v2_direction_schedule(
        sample_ids=("source:001", "source:002"), radii=(0.1,), max_direction_attempts=40, seed=20260725,
    )
    accepted = {
        row.perturbation_id: row.sample_id == "source:001" or row.direction_index < 31
        for row in schedule
    }

    result = select_first_accepted_directions(schedule, accepted_by_id=accepted, required_count=32)

    assert [row.direction_index for row in result.accepted] == list(range(32))
    assert result.insufficient == (("source:002", 0.1, 31),)


def test_h1_v2_schedule_identity_is_order_invariant() -> None:
    first = build_h1_v2_direction_schedule(
        sample_ids=("source:002", "source:001"), radii=(0.2, 0.1), max_direction_attempts=3, seed=20260725,
    )
    second = build_h1_v2_direction_schedule(
        sample_ids=("source:001", "source:002"), radii=(0.1, 0.2), max_direction_attempts=3, seed=20260725,
    )

    assert first == second


def test_h1_v2_oom_recovery_selects_only_complete_checkpoint_oom_failures() -> None:
    records = [
        {
            "sample_id": "source:oom", "checkpoint": checkpoint, "status": "failed",
            "failure_kind": "optimization", "failure_reason": "executor failed: OutOfMemoryError",
        }
        for checkpoint in (0, 25, 50, 100)
    ] + [
        {
            "sample_id": "source:mixed", "checkpoint": checkpoint,
            "status": "complete" if checkpoint == 0 else "failed",
            "failure_kind": None if checkpoint == 0 else "optimization",
            "failure_reason": None if checkpoint == 0 else "executor failed: OutOfMemoryError",
        }
        for checkpoint in (0, 25, 50, 100)
    ] + [
        {
            "sample_id": "source:other", "checkpoint": checkpoint, "status": "failed",
            "failure_kind": "optimization", "failure_reason": "executor failed: RuntimeError",
        }
        for checkpoint in (0, 25, 50, 100)
    ]

    assert select_h1_v2_oom_recovery_ids(records) == ("source:oom",)


def test_h1_v2_pending_recovery_excludes_candidates_with_prior_terminal_success() -> None:
    primary = [
        {
            "sample_id": sample_id, "checkpoint": checkpoint, "status": "failed",
            "failure_kind": "optimization", "failure_reason": "executor failed: OutOfMemoryError",
        }
        for sample_id in ("source:already_recovered", "source:pending")
        for checkpoint in (0, 25, 50, 100)
    ]
    recovery = [
        {"sample_id": "source:already_recovered", "checkpoint": 100, "status": "complete"},
    ]

    assert select_h1_v2_pending_recovery_ids(primary, recovery) == ("source:pending",)


def test_h1_v2_computational_exclusion_keeps_only_resolved_frozen_candidates() -> None:
    manifest_ids = tuple(f"source:{index:03d}" for index in range(81))
    excluded = "source:080"

    eligible = select_h1_v2_eligible_ids(
        manifest_ids=manifest_ids,
        terminal_ids=manifest_ids[:-1],
        computational_exclusions=(excluded,),
    )

    assert len(eligible) == 80
    assert excluded not in eligible
    with pytest.raises(ValueError, match="unresolved"):
        select_h1_v2_eligible_ids(
            manifest_ids=manifest_ids,
            terminal_ids=manifest_ids[:-2],
            computational_exclusions=(excluded,),
        )


def test_h1_v2_checkpointed_recovery_keeps_o_plus_identity_without_sdpa() -> None:
    assert h1_v2_recovery_spec("checkpointed") == (
        "jailbound_o_plus_recovery_checkpointed", True,
    )


def test_h1_v2_sdpa_recovery_is_limited_to_the_named_o_plus_recovery_method() -> None:
    assert h1_v2_recovery_spec("sdpa") == (
        "jailbound_o_plus_recovery_sdpa", False,
    )


def test_h1_v2_cpu_offload_recovery_keeps_o_plus_identity_without_checkpointing() -> None:
    assert h1_v2_recovery_spec("cpu_offload") == (
        "jailbound_o_plus_recovery_cpu_offload", False,
    )


def test_h1_v2_two_gpu_recovery_keeps_o_plus_identity_without_checkpointing() -> None:
    assert h1_v2_recovery_spec("two_gpu") == (
        "jailbound_o_plus_recovery_two_gpu", False,
    )


def test_h1_v2_two_gpu_checkpointed_recovery_keeps_o_plus_identity() -> None:
    assert h1_v2_recovery_spec("two_gpu_checkpointed") == (
        "jailbound_o_plus_recovery_two_gpu_checkpointed", True,
    )


def test_perturbation_schedule_has_stable_unique_ids_and_seeds() -> None:
    first = build_perturbation_schedule(
        sample_ids=("source:001", "source:002"),
        radii=(0.05, 0.1),
        directions_per_radius=2,
        seed=20260725,
    )
    second = build_perturbation_schedule(
        sample_ids=("source:002", "source:001"),
        radii=(0.1, 0.05),
        directions_per_radius=2,
        seed=20260725,
    )

    assert first == second
    assert len(first) == 8
    assert len({row.perturbation_id for row in first}) == 8
    assert len({row.direction_seed for row in first}) == 8


def test_causal_perplexity_uses_next_token_log_probabilities() -> None:
    logits = torch.tensor([[
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]])
    token_ids = torch.tensor([[0, 1, 0]])

    assert causal_perplexity(logits, token_ids) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least two"):
        causal_perplexity(logits[:, :1], token_ids[:, :1])


def test_accepted_perturbations_take_first_semantic_passes_per_radius() -> None:
    schedule = build_perturbation_schedule(
        sample_ids=("source:001",),
        radii=(0.1,),
        directions_per_radius=4,
        seed=20260725,
    )
    accepted = {
        schedule[0].perturbation_id: True,
        schedule[1].perturbation_id: False,
        schedule[2].perturbation_id: True,
        schedule[3].perturbation_id: True,
    }

    selected = select_accepted_perturbations(
        schedule,
        accepted_by_id=accepted,
        directions_per_radius=2,
    )

    assert [row.direction_index for row in selected] == [0, 2]


def test_schedule_shards_are_stable_disjoint_and_cover_the_frozen_rows() -> None:
    schedule = build_perturbation_schedule(
        sample_ids=("source:003", "source:001", "source:002"),
        radii=(0.1,),
        directions_per_radius=3,
        seed=20260725,
    )

    first = select_schedule_shard(schedule, shard_index=0, shard_count=2)
    second = select_schedule_shard(schedule, shard_index=1, shard_count=2)

    assert {row.perturbation_id for row in first}.isdisjoint(row.perturbation_id for row in second)
    assert {row.perturbation_id for row in first + second} == {row.perturbation_id for row in schedule}
    assert first == select_schedule_shard(tuple(reversed(schedule)), shard_index=0, shard_count=2)


def test_identifier_shards_are_stable_disjoint_and_cover_all_identifiers() -> None:
    identifiers = ("candidate:004", "candidate:001", "candidate:003", "candidate:002")

    first = select_id_shard(identifiers, shard_index=0, shard_count=2)
    second = select_id_shard(identifiers, shard_index=1, shard_count=2)

    assert set(first).isdisjoint(second)
    assert set(first + second) == set(identifiers)
    assert first == select_id_shard(tuple(reversed(identifiers)), shard_index=0, shard_count=2)
