"""Tensor-state optimizer adapters with exact update accounting."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Literal, Mapping, Sequence

import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState, ObjectiveValue
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter


FolSign = Literal[-1, 0, 1]


@dataclass(frozen=True)
class OptimizerSnapshot:
    """A detached tensor-state checkpoint with accounting at that instant."""

    checkpoint: int
    state: EditableState
    attack_loss: float
    maximize: float
    internal_margin: float
    fol: float | None
    selection_branch: str
    updates: int
    branch_updates: Mapping[str, int]
    forward_passes: int
    backward_passes: int
    hvp_calls: int


@dataclass(frozen=True)
class BranchStateRecord:
    """One detached post-update candidate from a dual-branch search."""

    branch: str
    step: int
    attack_loss: float
    fol: float | None
    branch_objective: float
    state: EditableState


@dataclass(frozen=True)
class BranchSearchResult(Sequence[OptimizerSnapshot]):
    """Dual-branch checkpoints together with all candidates and selections."""

    pool: tuple[BranchStateRecord, ...]
    selected: tuple[BranchStateRecord, ...]
    snapshots: tuple[OptimizerSnapshot, ...]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int | slice) -> OptimizerSnapshot | tuple[OptimizerSnapshot, ...]:
        return self.snapshots[index]

    def __iter__(self) -> Iterator[OptimizerSnapshot]:
        return iter(self.snapshots)


def select_branch_states(
    pool: Sequence[BranchStateRecord], branch: str, k: int
) -> tuple[BranchStateRecord, ...]:
    candidates = [item for item in pool if item.branch == branch]
    if len(candidates) < k:
        raise ValueError(f"branch {branch} has fewer than {k} valid states")
    return tuple(sorted(candidates, key=lambda item: (-item.branch_objective, item.step))[:k])


def _clone_live_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone().requires_grad_(True),
        u=state.u.detach().clone().requires_grad_(True),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


def _clone_snapshot_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone(),
        u=state.u.detach().clone(),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


def _snapshot(
    *,
    checkpoint: int,
    state: EditableState,
    value: ObjectiveValue,
    selection_branch: str,
    ledger: BudgetLedger,
) -> OptimizerSnapshot:
    return OptimizerSnapshot(
        checkpoint=checkpoint,
        state=_clone_snapshot_state(state),
        attack_loss=float(value.attack_loss.detach().cpu()),
        maximize=float(value.maximize.detach().cpu()),
        internal_margin=float(value.margin.detach().cpu()),
        fol=None if value.fol is None else float(value.fol.detach().cpu()),
        selection_branch=selection_branch,
        updates=ledger.updates,
        branch_updates=MappingProxyType(dict(ledger.branch_updates)),
        forward_passes=ledger.forward_passes,
        backward_passes=ledger.backward_passes,
        hvp_calls=ledger.hvp_calls,
    )


def _evaluate(
    objective: AttackObjective,
    state: EditableState,
    *,
    fol_sign: FolSign,
    include_fol: bool,
    ledger: BudgetLedger,
) -> ObjectiveValue:
    ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
    value = objective.evaluate(state, fol_sign=fol_sign, include_fol=include_fol)
    if include_fol:
        ledger.record_backward()
    return value


def _finite_difference_value(
    objective: AttackObjective,
    state: EditableState,
    *,
    fol_sign: FolSign,
    ledger: BudgetLedger,
) -> tuple[ObjectiveValue, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Evaluate FOL from a first-order gradient without retaining its graph."""
    ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
    raw = objective.evaluate(state, fol_sign=0, include_fol=False)
    gradients = torch.autograd.grad(raw.attack_loss, (state.z, state.u))
    ledger.record_backward()
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
    fol = objective.epsilon * norm
    value = ObjectiveValue(
        maximize=raw.attack_loss.detach() + fol_sign * objective.lambda_fol * fol.detach(),
        attack_loss=raw.attack_loss.detach(),
        proxy_risk=raw.proxy_risk.detach(),
        fol=fol.detach(),
        answer_logp=raw.answer_logp.detach(),
        refusal_logp=raw.refusal_logp.detach(),
        margin=raw.margin.detach(),
    )
    return value, gradients, norm


def _finite_difference_gradient(
    objective: AttackObjective,
    state: EditableState,
    *,
    fol_sign: FolSign,
    radius: float,
    ledger: BudgetLedger,
) -> tuple[ObjectiveValue, tuple[torch.Tensor, torch.Tensor]]:
    """Approximate the FOL gradient with a central finite-difference HVP."""
    value, gradients, norm = _finite_difference_value(objective, state, fol_sign=fol_sign, ledger=ledger)
    normalizer = norm.clamp_min(torch.finfo(norm.dtype).eps)
    direction = tuple(gradient / normalizer for gradient in gradients)

    def perturbed(sign: float) -> EditableState:
        return EditableState(
            z=(state.z.detach() + sign * radius * direction[0]).detach().requires_grad_(True),
            u=(state.u.detach() + sign * radius * direction[1]).detach().requires_grad_(True),
            z0=state.z0.detach().clone(),
            u0=state.u0.detach().clone(),
        )

    _, positive, _ = _finite_difference_value(objective, perturbed(1.0), fol_sign=0, ledger=ledger)
    _, negative, _ = _finite_difference_value(objective, perturbed(-1.0), fol_sign=0, ledger=ledger)
    ledger.record_hvp()
    hessian_gradient = tuple((plus - minus) / (2.0 * radius) for plus, minus in zip(positive, negative))
    correction = objective.lambda_fol * objective.epsilon / normalizer
    return value, tuple(gradient + fol_sign * correction * hessian for gradient, hessian in zip(gradients, hessian_gradient))


@dataclass(frozen=True)
class InitOptimizer:
    """Emit the unmodified starting tensor state without consuming budget."""

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[OptimizerSnapshot]:
        state = _clone_live_state(initial_state)
        if not emitter.due(0):
            return []
        value = _evaluate(objective, state, fol_sign=0, include_fol=False, ledger=ledger)
        return [_snapshot(
            checkpoint=0,
            state=state,
            value=value,
            selection_branch="init",
            ledger=ledger,
        )]


@dataclass(frozen=True)
class JailboundOptimizer:
    """One exact-budget Adam trajectory over the editable tensor state."""

    method: str
    fol_sign: FolSign
    include_fol: bool
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    finite_difference_fol: bool = False
    finite_difference_radius: float = 1e-3

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.include_fol != (self.fol_sign != 0):
            raise ValueError("FOL inclusion must match the FOL sign")
        if self.finite_difference_fol and not self.include_fol:
            raise ValueError("finite-difference FOL requires a FOL objective")
        if self.finite_difference_radius <= 0:
            raise ValueError("finite-difference radius must be positive")

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[OptimizerSnapshot]:
        state = _clone_live_state(initial_state)
        optimizer = torch.optim.Adam((state.z, state.u), lr=self.learning_rate)
        snapshots: list[OptimizerSnapshot] = []

        if emitter.due(0):
            snapshots.append(self._snapshot(0, objective, state, ledger))
        for step in range(1, ledger.update_limit + 1):
            ledger.consume_update()
            optimizer.zero_grad(set_to_none=True)
            if self.finite_difference_fol:
                value, gradient = _finite_difference_gradient(
                    objective,
                    state,
                    fol_sign=self.fol_sign,
                    radius=self.finite_difference_radius,
                    ledger=ledger,
                )
                state.z.grad, state.u.grad = (-gradient[0], -gradient[1])
            else:
                value = _evaluate(objective, state, fol_sign=self.fol_sign, include_fol=self.include_fol, ledger=ledger)
                (-value.maximize).backward()
                ledger.record_backward()
            torch.nn.utils.clip_grad_norm_((state.z, state.u), self.max_grad_norm)
            optimizer.step()
            if emitter.due(step):
                snapshots.append(self._snapshot(step, objective, state, ledger))
        return snapshots

    def _snapshot(
        self,
        checkpoint: int,
        objective: AttackObjective,
        state: EditableState,
        ledger: BudgetLedger,
    ) -> OptimizerSnapshot:
        if self.finite_difference_fol:
            value, _, _ = _finite_difference_value(objective, state, fol_sign=self.fol_sign, ledger=ledger)
        else:
            value = _evaluate(objective, state, fol_sign=self.fol_sign, include_fol=self.include_fol, ledger=ledger)
        return _snapshot(
            checkpoint=checkpoint,
            state=state,
            value=value,
            selection_branch=self.method,
            ledger=ledger,
        )


def build_jailbound_optimizer(
    method: str,
    *,
    learning_rate: float = 0.01,
    max_grad_norm: float = 1.0,
    finite_difference_fol: bool = False,
    finite_difference_radius: float = 1e-3,
) -> JailboundOptimizer:
    """Create one of the three fixed FOL objective variants."""

    options: dict[str, tuple[FolSign, bool]] = {
        "zol": (0, False),
        "jailbound_o_minus": (-1, True),
        "jailbound_o_plus": (1, True),
    }
    try:
        fol_sign, include_fol = options[method]
    except KeyError as exc:
        raise ValueError(f"unsupported JailBound optimizer: {method}") from exc
    return JailboundOptimizer(
        method=method,
        fol_sign=fol_sign,
        include_fol=include_fol,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        finite_difference_fol=finite_difference_fol,
        finite_difference_radius=finite_difference_radius,
    )


@dataclass(frozen=True)
class DualBranchOptimizer:
    """Alternate exact Adam updates between independent FOL minus/plus states."""

    learning_rate: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
        final_states_per_branch: int = 1,
    ) -> BranchSearchResult:
        if final_states_per_branch < 1:
            raise ValueError("final_states_per_branch must be positive")
        self._validate_ledger(ledger)
        states = {"o_minus": _clone_live_state(initial_state), "o_plus": _clone_live_state(initial_state)}
        optimizers = {
            name: torch.optim.Adam((state.z, state.u), lr=self.learning_rate)
            for name, state in states.items()
        }
        snapshots: list[OptimizerSnapshot] = []
        pool: list[BranchStateRecord] = []

        if emitter.due(0):
            snapshots.append(self._selected_snapshot(0, objective, states, ledger))
        for step in range(1, ledger.update_limit + 1):
            branch = "o_minus" if step % 2 else "o_plus"
            ledger.consume_update(branch)
            optimizer = optimizers[branch]
            optimizer.zero_grad(set_to_none=True)
            value = _evaluate(
                objective,
                states[branch],
                fol_sign=-1 if branch == "o_minus" else 1,
                include_fol=True,
                ledger=ledger,
            )
            (-value.maximize).backward()
            ledger.record_backward()
            torch.nn.utils.clip_grad_norm_((states[branch].z, states[branch].u), self.max_grad_norm)
            optimizer.step()
            pooled_value = _evaluate(
                objective,
                states[branch],
                fol_sign=-1 if branch == "o_minus" else 1,
                include_fol=True,
                ledger=ledger,
            )
            pool.append(
                BranchStateRecord(
                    branch=branch,
                    step=step,
                    attack_loss=float(pooled_value.attack_loss.detach().cpu()),
                    fol=None if pooled_value.fol is None else float(pooled_value.fol.detach().cpu()),
                    branch_objective=float(pooled_value.maximize.detach().cpu()),
                    state=_clone_snapshot_state(states[branch]),
                )
            )
            if emitter.due(step):
                snapshots.append(self._selected_snapshot(step, objective, states, ledger))

        if ledger.branch_updates != dict(ledger.branch_limits):
            raise AssertionError("terminal branch accounting does not match configured limits")
        selected = select_branch_states(pool, "o_minus", final_states_per_branch) + select_branch_states(
            pool, "o_plus", final_states_per_branch
        )
        return BranchSearchResult(tuple(pool), selected, tuple(snapshots))

    @staticmethod
    def _validate_ledger(ledger: BudgetLedger) -> None:
        expected = {"o_minus", "o_plus"}
        if set(ledger.branch_limits) != expected:
            raise ValueError("dual branch limits must configure o_minus and o_plus")
        if ledger.branch_limits["o_minus"] != ledger.branch_limits["o_plus"]:
            raise ValueError("branch limits must be equal for strict alternation")
        if ledger.update_limit != sum(ledger.branch_limits.values()):
            raise ValueError("update limit must equal the total branch limits")

    @staticmethod
    def _selected_snapshot(
        checkpoint: int,
        objective: AttackObjective,
        states: Mapping[str, EditableState],
        ledger: BudgetLedger,
    ) -> OptimizerSnapshot:
        minus_value = _evaluate(objective, states["o_minus"], fol_sign=-1, include_fol=True, ledger=ledger)
        plus_value = _evaluate(objective, states["o_plus"], fol_sign=1, include_fol=True, ledger=ledger)
        if minus_value.attack_loss.detach() >= plus_value.attack_loss.detach():
            branch, value = "o_minus", minus_value
        else:
            branch, value = "o_plus", plus_value
        return _snapshot(
            checkpoint=checkpoint,
            state=states[branch],
            value=value,
            selection_branch=branch,
            ledger=ledger,
        )
