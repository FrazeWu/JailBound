from __future__ import annotations

from benchmark.safety_eval.fol_runtime import (
    build_perturbation_schedule,
    causal_perplexity,
    select_id_shard,
    select_schedule_shard,
    select_accepted_perturbations,
    select_fol_experiment,
)
from benchmark.safety_eval.manifest import FolCandidate
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
