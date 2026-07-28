from __future__ import annotations

import torch
import pytest

from benchmark.safety_eval.objective import AttackObjective, EditableState
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.safety_eval.optimizers.gcg import GCGOptimizer


def _embedding() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.5, 0.0],
        ]
    )


def _objective() -> AttackObjective:
    return AttackObjective(
        answer_vector=torch.tensor([1.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 1.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.0,
        gamma_u=0.0,
    )


def _id_state() -> EditableState:
    ids = torch.tensor([[0]], dtype=torch.long)
    return EditableState(z=ids, u=ids.clone(), z0=ids.clone(), u0=ids.clone())


def test_gcg_uses_coordinate_gradient_to_select_best_allowed_token() -> None:
    ledger = BudgetLedger(update_limit=1, candidate_limit=1)

    snapshots = GCGOptimizer(_embedding(), search_width=1, top_k=2).run(
        _objective(), _id_state(), ledger, CheckpointEmitter([0, 1])
    )

    assert snapshots[-1].z_token_ids.tolist() == [[1]]
    assert snapshots[-1].u_token_ids.tolist() == [[0]]
    assert snapshots[-1].attack_loss > snapshots[0].attack_loss


def test_gcg_masks_forbidden_candidate_ids() -> None:
    ledger = BudgetLedger(update_limit=1, candidate_limit=2)

    snapshots = GCGOptimizer(_embedding(), forbidden_token_ids=(1,), search_width=2, top_k=3).run(
        _objective(), _id_state(), ledger, CheckpointEmitter([0, 1])
    )

    assert all((snapshot.z_token_ids != 1).all() and (snapshot.u_token_ids != 1).all() for snapshot in snapshots)
    assert snapshots[-1].z_token_ids.tolist() == [[3]]


def test_gcg_respects_global_candidate_cap_without_ending_fixed_updates() -> None:
    ledger = BudgetLedger(update_limit=3, candidate_limit=2)

    snapshots = GCGOptimizer(_embedding(), search_width=2, top_k=2).run(
        _objective(), _id_state(), ledger, CheckpointEmitter([0, 1, 3])
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 3]
    assert [snapshot.updates for snapshot in snapshots] == [0, 1, 3]
    assert ledger.updates == 3
    assert (ledger.candidates_attempted, ledger.candidates_accepted) == (2, 1)
    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (8, 3, 0)
    assert snapshots[-1].candidates_attempted == 2


def test_gcg_snapshots_are_detached_and_match_frozen_vocabulary_rows() -> None:
    ledger = BudgetLedger(update_limit=1, candidate_limit=1)

    snapshots = GCGOptimizer(_embedding(), search_width=1, top_k=1).run(
        _objective(), _id_state(), ledger, CheckpointEmitter([0, 1])
    )

    assert torch.equal(snapshots[-1].state.z, _embedding()[snapshots[-1].z_token_ids])
    assert snapshots[-1].state.z.requires_grad is False
    with torch.no_grad():
        snapshots[0].z_token_ids.add_(3)
    assert not torch.equal(snapshots[0].z_token_ids, snapshots[-1].z_token_ids)


def test_gcg_uses_partial_top_k_selection_instead_of_full_vocabulary_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = BudgetLedger(update_limit=1, candidate_limit=1)

    def full_sort_must_not_run(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("full vocabulary sort is not permitted")

    monkeypatch.setattr(torch, "argsort", full_sort_must_not_run)
    snapshots = GCGOptimizer(_embedding(), search_width=1, top_k=2).run(
        _objective(), _id_state(), ledger, CheckpointEmitter([0, 1])
    )

    assert snapshots[-1].z_token_ids.tolist() == [[1]]
