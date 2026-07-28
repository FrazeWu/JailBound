from __future__ import annotations

import pytest

from benchmark.safety_eval.optimizers.random_mutation import (
    CandidateCapExceeded,
    RandomMutationOptimizer,
)


def test_random_mutation_emits_fixed_checkpoints_and_selects_the_best_score() -> None:
    attempts = 0

    def rewriter(_: str) -> str:
        nonlocal attempts
        attempts += 1
        return f"state-{attempts:03d}"

    def semantic_acceptor(candidate: str) -> bool:
        return int(candidate.removeprefix("state-")) % 4 != 0

    optimizer = RandomMutationOptimizer()
    snapshots = optimizer.run(
        initial_candidate="state-000",
        objective_score=lambda candidate: float(int(candidate.removeprefix("state-"))),
        rewriter=rewriter,
        semantic_acceptor=semantic_acceptor,
        candidate_limit=133,
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 25, 50, 100]
    assert [snapshot.objective_score for snapshot in snapshots] == [0.0, 33.0, 66.0, 133.0]
    assert snapshots[-1].candidate == "state-133"
    assert [(snapshot.updates, snapshot.candidates_attempted, snapshot.candidates_accepted) for snapshot in snapshots] == [
        (0, 0, 0),
        (25, 33, 25),
        (50, 66, 50),
        (100, 133, 100),
    ]


def test_random_mutation_raises_typed_candidate_cap_failure_with_counters() -> None:
    attempts = 0

    def rewriter(_: str) -> str:
        nonlocal attempts
        attempts += 1
        return f"state-{attempts:03d}"

    with pytest.raises(CandidateCapExceeded) as error:
        RandomMutationOptimizer().run(
            initial_candidate="state-000",
            objective_score=lambda _: 0.0,
            rewriter=rewriter,
            semantic_acceptor=lambda _: False,
            candidate_limit=3,
        )

    assert attempts == 3
    assert error.value.candidates_attempted == 3
    assert error.value.candidates_accepted == 0
    assert error.value.updates == 0
