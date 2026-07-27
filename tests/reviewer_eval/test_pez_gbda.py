from __future__ import annotations

import pytest
import torch

from benchmark.reviewer_eval.objective import AttackObjective, EditableState
from benchmark.reviewer_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.reviewer_eval.optimizers.gbda import GBDAOptimizer, linear_temperature, masked_logits
from benchmark.reviewer_eval.optimizers.pez import PEZOptimizer, straight_through_project


def _objective() -> AttackObjective:
    return AttackObjective(
        answer_vector=torch.tensor([1.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 1.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.1,
        gamma_u=0.1,
    )


def _large_gradient_objective() -> AttackObjective:
    return AttackObjective(
        answer_vector=torch.tensor([100.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 100.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.1,
        gamma_u=0.1,
    )


def _embedding() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def _embedding_state() -> EditableState:
    return EditableState(
        z=torch.tensor([[[0.9, 0.2]]], requires_grad=True),
        u=torch.tensor([[[0.1, 0.8]]], requires_grad=True),
        z0=torch.tensor([[[0.9, 0.2]]]),
        u0=torch.tensor([[[0.1, 0.8]]]),
    )


def _logit_state() -> EditableState:
    return EditableState(
        z=torch.tensor([[[3.0, 1.0, 0.0]]], requires_grad=True),
        u=torch.tensor([[[1.0, 3.0, 0.0]]], requires_grad=True),
        z0=torch.zeros(1, 1, 3),
        u0=torch.zeros(1, 1, 3),
    )


def test_pez_projects_to_nearest_allowed_embedding_with_identity_ste_gradient() -> None:
    soft = torch.tensor([[[0.8, 0.2]]], requires_grad=True)

    hard, token_ids = straight_through_project(soft, _embedding(), forbidden_token_ids=(0,))
    hard.sum().backward()

    assert token_ids.tolist() == [[1]]
    assert torch.equal(hard.detach(), _embedding()[1].reshape(1, 1, 2))
    assert torch.equal(soft.grad, torch.ones_like(soft))


def test_pez_optimizer_runs_exact_budget_with_hard_snapshots_and_soft_state() -> None:
    ledger = BudgetLedger(update_limit=3, candidate_limit=9)

    snapshots = PEZOptimizer(_embedding(), forbidden_token_ids=(2,), learning_rate=0.05).run(
        _objective(), _embedding_state(), ledger, CheckpointEmitter([0, 1, 3])
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 3]
    assert [snapshot.updates for snapshot in snapshots] == [0, 1, 3]
    assert ledger.updates == 3
    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (6, 3, 0)
    assert all((snapshot.z_token_ids != 2).all() and (snapshot.u_token_ids != 2).all() for snapshot in snapshots)
    assert torch.equal(snapshots[0].state.z, _embedding()[snapshots[0].z_token_ids])
    assert snapshots[-1].soft_state.z.requires_grad is False
    with torch.no_grad():
        snapshots[0].state.z.add_(5)
    assert not torch.equal(snapshots[0].state.z, snapshots[-1].state.z)


def test_gbda_masks_forbidden_logits_and_uses_inclusive_temperature_endpoints() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])

    masked = masked_logits(logits, forbidden_token_ids=(2,))

    assert masked.argmax(dim=-1).tolist() == [[1]]
    assert torch.isneginf(masked[..., 2]).all()
    assert [linear_temperature(step, 4) for step in range(4)] == pytest.approx([1.0, 0.7, 0.4, 0.1])


def test_gbda_optimizer_runs_exact_budget_and_snapshots_allowed_argmax_ids() -> None:
    torch.manual_seed(7)
    ledger = BudgetLedger(update_limit=3, candidate_limit=9)

    snapshots = GBDAOptimizer(_embedding(), forbidden_token_ids=(2,), learning_rate=0.05).run(
        _objective(), _logit_state(), ledger, CheckpointEmitter([0, 1, 3])
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 3]
    assert [snapshot.updates for snapshot in snapshots] == [0, 1, 3]
    assert ledger.updates == 3
    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (6, 3, 0)
    assert [snapshot.temperature for snapshot in snapshots] == pytest.approx([1.0, 1.0, 0.1])
    assert all((snapshot.z_token_ids != 2).all() and (snapshot.u_token_ids != 2).all() for snapshot in snapshots)
    assert torch.equal(snapshots[0].state.z, _embedding()[snapshots[0].z_token_ids])
    assert snapshots[-1].soft_state.z.requires_grad is False
    assert snapshots[-1].logits_state.z.requires_grad is False


@pytest.mark.parametrize("method", ["pez", "gbda"])
def test_optimizers_jointly_clip_both_editable_tensors_before_adam_step(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    calls: list[tuple[int, float, float, float]] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def observe(parameters: object, max_norm: float, *args: object, **kwargs: object) -> torch.Tensor:
        tensors = list(parameters)  # type: ignore[arg-type]
        before = torch.linalg.vector_norm(torch.stack([tensor.grad.norm() for tensor in tensors])).item()
        result = original_clip(tensors, max_norm, *args, **kwargs)
        after = torch.linalg.vector_norm(torch.stack([tensor.grad.norm() for tensor in tensors])).item()
        calls.append((len(tensors), max_norm, before, after))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    if method == "pez":
        optimizer = PEZOptimizer(_embedding(), forbidden_token_ids=(2,), learning_rate=0.05)
        state = _embedding_state()
    else:
        optimizer = GBDAOptimizer(_embedding(), forbidden_token_ids=(2,), learning_rate=0.05)
        state = _logit_state()

    optimizer.run(_large_gradient_objective(), state, BudgetLedger(update_limit=1, candidate_limit=3), CheckpointEmitter([0, 1]))

    assert len(calls) == 1
    size, max_norm, before, after = calls[0]
    assert size == 2
    assert max_norm == 1.0
    assert before > 1.0
    assert after <= 1.0
