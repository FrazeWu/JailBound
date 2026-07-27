from __future__ import annotations

import torch

from benchmark.reviewer_eval.objective import AttackObjective, EditableState


def test_attack_objective_has_gradients_for_both_editable_blocks() -> None:
    state = EditableState(
        z=torch.randn(1, 1, 3, requires_grad=True), u=torch.randn(1, 1, 3, requires_grad=True),
        z0=torch.zeros(1, 1, 3), u0=torch.zeros(1, 1, 3),
    )
    objective = AttackObjective(answer_vector=torch.tensor([1.0, 0.0, 0.0]), refusal_vector=torch.tensor([0.0, 1.0, 0.0]), epsilon=.1, lambda_fol=.1, gamma_z=.01, gamma_u=.01)
    value = objective.evaluate(state, include_fol=True)
    gradients = torch.autograd.grad(value.maximize, [state.z, state.u])
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0
    assert value.fol is not None and torch.isfinite(value.fol)


def test_hvp_is_finite_for_joint_editable_state() -> None:
    state = EditableState(torch.ones(1, 1, 2, requires_grad=True), torch.ones(1, 1, 2, requires_grad=True), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
    objective = AttackObjective(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), epsilon=.1, lambda_fol=.1, gamma_z=.1, gamma_u=.1)
    hz, hu = objective.hvp(state, (torch.ones_like(state.z), torch.ones_like(state.u)))
    assert torch.isfinite(hz).all() and torch.isfinite(hu).all()


def test_objective_freezes_requires_grad_anchor_vectors() -> None:
    answer_anchor = torch.tensor([1.0, 0.0], requires_grad=True)
    refusal_anchor = torch.tensor([0.0, 1.0], requires_grad=True)
    objective = AttackObjective(
        answer_anchor,
        refusal_anchor,
        epsilon=.1,
        lambda_fol=.1,
        gamma_z=.1,
        gamma_u=.1,
    )
    state = EditableState(
        torch.ones(1, 1, 2, requires_grad=True),
        torch.ones(1, 1, 2, requires_grad=True),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )

    objective.evaluate(state).maximize.backward()

    assert answer_anchor.grad is None
    assert refusal_anchor.grad is None
