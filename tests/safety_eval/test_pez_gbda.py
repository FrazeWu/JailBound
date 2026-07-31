from __future__ import annotations

import pytest
import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.safety_eval.optimizers.gbda import GBDAOptimizer, linear_temperature, masked_logits
from benchmark.safety_eval.optimizers.gbda_official import (
    OfficialGBDAOptimizer,
    build_official_logit_state,
)
from benchmark.safety_eval.optimizers.pez import PEZOptimizer, straight_through_project


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


class _BatchCandidateObjective(AttackObjective):
    forward_passes_per_evaluation = 1

    def evaluate_candidates(self, state: EditableState) -> torch.Tensor:
        pooled = torch.cat((state.z, state.u), dim=1).mean(dim=1)
        answer = (pooled * self.answer_vector.to(pooled)).sum(dim=-1)
        refusal = (pooled * self.refusal_vector.to(pooled)).sum(dim=-1)
        z_penalty = (state.z - state.z0).square().sum(dim=(1, 2))
        u_penalty = (state.u - state.u0).square().sum(dim=(1, 2))
        return answer - refusal - self.gamma_z * z_penalty - self.gamma_u * u_penalty


class _RecordingBatchObjective(_BatchCandidateObjective):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.soft_batch_sizes: list[int] = []

    def evaluate(self, state: EditableState, **kwargs: object):  # type: ignore[no-untyped-def]
        self.soft_batch_sizes.append(state.z.shape[0])
        return super().evaluate(state, **kwargs)


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


def test_official_gbda_initializes_original_tokens_to_fifteen_over_zero_logits() -> None:
    state = build_official_logit_state(
        torch.tensor([[0, 2]]),
        torch.tensor([[1]]),
        vocabulary_size=3,
        device=torch.device("cpu"),
    )

    assert state.z.dtype == state.u.dtype == torch.float32
    assert state.z.tolist() == [[[15.0, 0.0, 0.0], [0.0, 0.0, 15.0]]]
    assert state.u.tolist() == [[[0.0, 15.0, 0.0]]]


def test_official_gbda_uses_fixed_soft_samples_and_selects_one_final_hard_candidate() -> None:
    torch.manual_seed(7)
    objective = _BatchCandidateObjective(
        answer_vector=torch.tensor([1.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 1.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.1,
        gamma_u=0.1,
    )
    state = build_official_logit_state(
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        vocabulary_size=3,
        device=torch.device("cpu"),
    )
    ledger = BudgetLedger(update_limit=2, candidate_limit=32)

    snapshots = OfficialGBDAOptimizer(
        _embedding(),
        forbidden_token_ids=(2,),
        soft_samples_per_update=3,
        hard_samples=5,
        hard_sample_batch_size=2,
    ).run(objective, state, ledger, CheckpointEmitter([0, 1, 2]))

    defaults = OfficialGBDAOptimizer(_embedding())
    assert defaults.learning_rate == pytest.approx(0.3)
    assert defaults.soft_sample_batch_size == 5
    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 2]
    assert [snapshot.temperature for snapshot in snapshots] == [1.0, 1.0, 1.0]
    assert (ledger.updates, ledger.candidates_attempted, ledger.candidates_accepted) == (2, 11, 1)
    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (7, 2, 0)
    assert snapshots[-1].hard_samples == 5
    assert all((snapshot.z_token_ids != 2).all() and (snapshot.u_token_ids != 2).all() for snapshot in snapshots)


def test_official_gbda_microbatches_ten_soft_samples_without_changing_the_update_budget() -> None:
    objective = _RecordingBatchObjective(
        answer_vector=torch.tensor([1.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 1.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.1,
        gamma_u=0.1,
    )
    state = build_official_logit_state(
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        vocabulary_size=3,
        device=torch.device("cpu"),
    )
    ledger = BudgetLedger(update_limit=1, candidate_limit=16)

    OfficialGBDAOptimizer(
        _embedding(),
        soft_samples_per_update=10,
        soft_sample_batch_size=2,
        hard_samples=1,
        hard_sample_batch_size=1,
    ).run(objective, state, ledger, CheckpointEmitter([0, 1]))

    assert objective.soft_batch_sizes == [2, 2, 2, 2, 2]
    assert (ledger.updates, ledger.candidates_attempted, ledger.candidates_accepted) == (1, 11, 1)
    assert (ledger.forward_passes, ledger.backward_passes) == (7, 5)


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
