from __future__ import annotations

import pytest
import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.safety_eval.optimizers import jailbound
from benchmark.safety_eval.optimizers.jailbound import DualBranchOptimizer, InitOptimizer, build_jailbound_optimizer


def _state() -> EditableState:
    return EditableState(
        z=torch.tensor([[[0.30, -0.10]]], requires_grad=True),
        u=torch.tensor([[[0.05, 0.20]]], requires_grad=True),
        z0=torch.zeros(1, 1, 2),
        u0=torch.zeros(1, 1, 2),
    )


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
        answer_vector=torch.tensor([10.0, 0.0]),
        refusal_vector=torch.tensor([0.0, 10.0]),
        epsilon=0.1,
        lambda_fol=0.3,
        gamma_z=0.1,
        gamma_u=0.1,
    )


def test_init_emits_only_untouched_checkpoint_zero_without_budget_cost() -> None:
    initial = _state()
    original_z = initial.z.detach().clone()
    original_u = initial.u.detach().clone()
    ledger = BudgetLedger(update_limit=3, candidate_limit=7)

    snapshots = InitOptimizer().run(_objective(), initial, ledger, CheckpointEmitter([0, 1, 3]))

    assert [snapshot.checkpoint for snapshot in snapshots] == [0]
    snapshot = snapshots[0]
    assert snapshot.updates == 0
    assert dict(snapshot.branch_updates) == {}
    assert snapshot.selection_branch == "init"
    assert torch.equal(snapshot.state.z, original_z)
    assert torch.equal(snapshot.state.u, original_u)
    assert ledger.updates == ledger.candidates_attempted == ledger.candidates_accepted == 0
    assert initial.z.requires_grad and initial.u.requires_grad
    assert torch.equal(initial.z.detach(), original_z)
    assert torch.equal(initial.u.detach(), original_u)


def test_init_rejects_emitter_without_checkpoint_zero() -> None:
    with pytest.raises(ValueError, match="checkpoint 0"):
        InitOptimizer().run(_objective(), _state(), BudgetLedger(update_limit=3, candidate_limit=7), CheckpointEmitter([1, 3]))


@pytest.mark.parametrize(
    ("method", "sign", "has_fol"),
    [("zol", 0, False), ("jailbound_o_minus", -1, True), ("jailbound_o_plus", 1, True)],
)
def test_single_branch_honors_exact_budget_checkpoint_timing_and_objective_sign(
    method: str, sign: int, has_fol: bool
) -> None:
    initial = _state()
    baseline_z0 = initial.z0.clone()
    baseline_u0 = initial.u0.clone()
    ledger = BudgetLedger(update_limit=3, candidate_limit=9)

    snapshots = build_jailbound_optimizer(method, learning_rate=0.05).run(
        _objective(), initial, ledger, CheckpointEmitter([0, 1, 3])
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 3]
    assert [snapshot.updates for snapshot in snapshots] == [0, 1, 3]
    assert ledger.updates == 3
    assert ledger.candidates_attempted == ledger.candidates_accepted == 0
    assert all(snapshot.selection_branch == method for snapshot in snapshots)
    assert all(snapshot.fol is not None for snapshot in snapshots) is has_fol
    assert torch.equal(initial.z0, baseline_z0)
    assert torch.equal(initial.u0, baseline_u0)

    first, second, last = snapshots
    assert torch.equal(first.state.z, initial.z.detach())
    assert torch.equal(first.state.u, initial.u.detach())
    assert not torch.equal(first.state.z, last.state.z)
    assert not torch.equal(first.state.u, last.state.u)
    if has_fol:
        assert first.fol is not None and first.fol > 0
        delta = first.maximize - first.attack_loss
        assert delta * sign > 0
    else:
        assert first.fol is None
        assert first.maximize == pytest.approx(first.attack_loss)

    with torch.no_grad():
        first.state.z.add_(99)
    assert not torch.equal(first.state.z, second.state.z)
    assert not torch.equal(first.state.z, last.state.z)


def test_dual_branch_alternates_exactly_and_breaks_initial_tie_toward_o_minus() -> None:
    initial = _state()
    ledger = BudgetLedger(
        update_limit=4,
        candidate_limit=9,
        branch_limits={"o_minus": 2, "o_plus": 2},
    )

    snapshots = DualBranchOptimizer(learning_rate=0.05).run(
        _objective(), initial, ledger, CheckpointEmitter([0, 1, 2, 4])
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 2, 4]
    assert [snapshot.updates for snapshot in snapshots] == [0, 1, 2, 4]
    assert snapshots[0].selection_branch == "o_minus"
    assert all(snapshot.selection_branch in {"o_minus", "o_plus"} for snapshot in snapshots)
    assert ledger.updates == 4
    assert ledger.branch_updates == {"o_minus": 2, "o_plus": 2}
    assert dict(snapshots[-1].branch_updates) == {"o_minus": 2, "o_plus": 2}
    assert torch.equal(initial.z.detach(), _state().z.detach())
    assert torch.equal(initial.u.detach(), _state().u.detach())


def test_dual_branch_rejects_terminal_budget_mismatch() -> None:
    ledger = BudgetLedger(
        update_limit=3,
        candidate_limit=9,
        branch_limits={"o_minus": 2, "o_plus": 2},
    )

    with pytest.raises(ValueError, match="branch limits"):
        DualBranchOptimizer().run(_objective(), _state(), ledger, CheckpointEmitter([0, 3]))


def test_builder_rejects_unknown_optimizer_identity() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_jailbound_optimizer("unknown")


@pytest.mark.parametrize("dual", [False, True])
def test_trajectories_jointly_clip_editable_gradients_before_adam_step(monkeypatch: pytest.MonkeyPatch, dual: bool) -> None:
    calls: list[tuple[int, float, float, float]] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def observe(parameters: object, max_norm: float, *args: object, **kwargs: object) -> torch.Tensor:
        parameter_list = list(parameters)  # type: ignore[arg-type]
        before = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in parameter_list])).item()
        result = original_clip(parameter_list, max_norm, *args, **kwargs)
        after = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in parameter_list])).item()
        calls.append((len(parameter_list), max_norm, before, after))
        return result

    monkeypatch.setattr(jailbound.torch.nn.utils, "clip_grad_norm_", observe)
    if dual:
        optimizer = DualBranchOptimizer(learning_rate=0.05)
        ledger = BudgetLedger(update_limit=4, candidate_limit=9, branch_limits={"o_minus": 2, "o_plus": 2})
        emitter = CheckpointEmitter([0, 4])
    else:
        optimizer = build_jailbound_optimizer("zol", learning_rate=0.05)
        ledger = BudgetLedger(update_limit=3, candidate_limit=9)
        emitter = CheckpointEmitter([0, 3])

    optimizer.run(_large_gradient_objective(), _state(), ledger, emitter)

    assert len(calls) == ledger.update_limit
    assert all(size == 2 and max_norm == 1.0 and before > 1.0 and after <= 1.0 for size, max_norm, before, after in calls)


def test_single_branch_records_deterministic_evaluation_and_backward_counts() -> None:
    ledger = BudgetLedger(update_limit=3, candidate_limit=9)

    snapshots = build_jailbound_optimizer("zol", learning_rate=0.05).run(
        _objective(), _state(), ledger, CheckpointEmitter([0, 1, 3])
    )

    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (6, 3, 0)
    assert [(snapshot.forward_passes, snapshot.backward_passes, snapshot.hvp_calls) for snapshot in snapshots] == [
        (1, 0, 0),
        (3, 1, 0),
        (6, 3, 0),
    ]


@pytest.mark.parametrize("method", ["jailbound_o_minus", "jailbound_o_plus"])
def test_fol_trajectory_counts_internal_and_outer_backward_calls(method: str) -> None:
    ledger = BudgetLedger(update_limit=3, candidate_limit=9)

    snapshots = build_jailbound_optimizer(method, learning_rate=0.05).run(
        _objective(), _state(), ledger, CheckpointEmitter([0, 1, 3])
    )

    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (6, 9, 0)
    assert [(snapshot.forward_passes, snapshot.backward_passes, snapshot.hvp_calls) for snapshot in snapshots] == [
        (1, 1, 0),
        (3, 4, 0),
        (6, 9, 0),
    ]


def test_o_plus_finite_difference_hvp_matches_the_exact_update_direction() -> None:
    initial = _state()
    exact_ledger = BudgetLedger(update_limit=1, candidate_limit=9)
    finite_difference_ledger = BudgetLedger(update_limit=1, candidate_limit=9)

    exact = build_jailbound_optimizer("jailbound_o_plus", learning_rate=0.05).run(
        _objective(), initial, exact_ledger, CheckpointEmitter([0, 1])
    )
    finite_difference = build_jailbound_optimizer(
        "jailbound_o_plus",
        learning_rate=0.05,
        finite_difference_fol=True,
        finite_difference_radius=1e-3,
    ).run(_objective(), initial, finite_difference_ledger, CheckpointEmitter([0, 1]))

    assert finite_difference_ledger.updates == exact_ledger.updates == 1
    assert finite_difference_ledger.hvp_calls == 1
    assert finite_difference[-1].fol is not None
    assert torch.allclose(finite_difference[-1].state.z, exact[-1].state.z, atol=1e-4, rtol=1e-4)
    assert torch.allclose(finite_difference[-1].state.u, exact[-1].state.u, atol=1e-4, rtol=1e-4)
