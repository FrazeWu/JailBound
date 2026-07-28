"""Tensor-only GBDA mechanics for controlled safety experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState, ObjectiveValue
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter


def _allowed_token_mask(vocabulary_size: int, forbidden_token_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    if vocabulary_size <= 0:
        raise ValueError("embedding vocabulary must be non-empty")
    mask = torch.ones(vocabulary_size, dtype=torch.bool, device=device)
    for token_id in forbidden_token_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < vocabulary_size:
            raise ValueError("forbidden token id is outside the vocabulary")
        mask[token_id] = False
    if not mask.any():
        raise ValueError("at least one token id must remain allowed")
    return mask


def masked_logits(logits: torch.Tensor, forbidden_token_ids: Sequence[int] = ()) -> torch.Tensor:
    """Return a copy of floating logits with forbidden vocabulary columns at ``-inf``."""

    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating [batch, positions, vocabulary] tensor")
    allowed = _allowed_token_mask(logits.shape[-1], forbidden_token_ids, logits.device)
    return logits.masked_fill(~allowed.view(1, 1, -1), -torch.inf)


def linear_temperature(
    update_index: int,
    update_limit: int,
    *,
    start: float = 1.0,
    end: float = 0.1,
) -> float:
    """Inclusive linear annealing over zero-indexed optimizer updates."""

    if update_limit <= 0:
        raise ValueError("update_limit must be positive")
    if not 0 <= update_index < update_limit:
        raise ValueError("update_index must be within the configured update limit")
    if start <= 0 or end <= 0:
        raise ValueError("temperatures must be positive")
    if update_limit == 1:
        return start
    return start + (end - start) * update_index / (update_limit - 1)


@dataclass(frozen=True)
class GBDASnapshot:
    """A detached GBDA checkpoint with hard argmax tokens, soft embeddings, and logits."""

    checkpoint: int
    state: EditableState
    soft_state: EditableState
    logits_state: EditableState
    z_token_ids: torch.Tensor
    u_token_ids: torch.Tensor
    temperature: float
    attack_loss: float
    maximize: float
    updates: int
    branch_updates: Mapping[str, int]
    forward_passes: int
    backward_passes: int
    hvp_calls: int


def _clone_live_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone().requires_grad_(True),
        u=state.u.detach().clone().requires_grad_(True),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


def _clone_detached_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone(),
        u=state.u.detach().clone(),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


@dataclass(frozen=True)
class GBDAOptimizer:
    """Exact-budget Adam optimizer over logits with Gumbel-softmax embeddings."""

    embedding: torch.Tensor
    forbidden_token_ids: tuple[int, ...] = ()
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.1

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2 or not self.embedding.is_floating_point():
            raise ValueError("embedding must be a floating point [vocabulary, dim] tensor")
        _allowed_token_mask(self.embedding.shape[0], self.forbidden_token_ids, self.embedding.device)
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("temperatures must be positive")

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[GBDASnapshot]:
        logits_state = _clone_live_state(initial_state)
        self._validate_logits(logits_state)
        baseline = self._initial_baseline(logits_state)
        optimizer = torch.optim.Adam((logits_state.z, logits_state.u), lr=self.learning_rate)
        snapshots: list[GBDASnapshot] = []

        if emitter.due(0):
            snapshots.append(self._snapshot(0, objective, logits_state, baseline, ledger))
        for step in range(1, ledger.update_limit + 1):
            ledger.consume_update()
            optimizer.zero_grad(set_to_none=True)
            temperature = linear_temperature(
                step - 1,
                ledger.update_limit,
                start=self.temperature_start,
                end=self.temperature_end,
            )
            soft_state = self._gumbel_state(logits_state, baseline, temperature)
            value = self._evaluate(objective, soft_state, ledger)
            (-value.maximize).backward()
            ledger.record_backward()
            torch.nn.utils.clip_grad_norm_((logits_state.z, logits_state.u), self.max_grad_norm)
            optimizer.step()
            if emitter.due(step):
                snapshots.append(self._snapshot(step, objective, logits_state, baseline, ledger))
        return snapshots

    def _validate_logits(self, state: EditableState) -> None:
        vocabulary_size = self.embedding.shape[0]
        if state.z.ndim != 3 or state.u.ndim != 3 or state.z.shape[-1] != vocabulary_size or state.u.shape[-1] != vocabulary_size:
            raise ValueError("GBDA editable z/u must be [batch, positions, vocabulary] logits")
        if state.z.shape[0] != state.u.shape[0]:
            raise ValueError("GBDA z/u logits must share a batch dimension")

    def _initial_baseline(self, logits_state: EditableState) -> EditableState:
        vocabulary = self.embedding.detach().to(device=logits_state.z.device, dtype=logits_state.z.dtype)
        z_probs = torch.softmax(masked_logits(logits_state.z, self.forbidden_token_ids), dim=-1)
        u_probs = torch.softmax(masked_logits(logits_state.u, self.forbidden_token_ids), dim=-1)
        z_baseline = (z_probs @ vocabulary).detach().clone()
        u_baseline = (u_probs @ vocabulary).detach().clone()
        return EditableState(
            z=z_baseline,
            u=u_baseline,
            z0=z_baseline.clone(),
            u0=u_baseline.clone(),
        )

    def _gumbel_state(self, logits_state: EditableState, baseline: EditableState, temperature: float) -> EditableState:
        vocabulary = self.embedding.detach().to(device=logits_state.z.device, dtype=logits_state.z.dtype)
        z_weights = torch.nn.functional.gumbel_softmax(
            masked_logits(logits_state.z, self.forbidden_token_ids), tau=temperature, hard=False, dim=-1
        )
        u_weights = torch.nn.functional.gumbel_softmax(
            masked_logits(logits_state.u, self.forbidden_token_ids), tau=temperature, hard=False, dim=-1
        )
        return EditableState(z_weights @ vocabulary, u_weights @ vocabulary, baseline.z0, baseline.u0)

    def _hard_state(self, logits_state: EditableState, baseline: EditableState) -> tuple[EditableState, torch.Tensor, torch.Tensor]:
        vocabulary = self.embedding.detach().to(device=logits_state.z.device, dtype=logits_state.z.dtype)
        z_token_ids = masked_logits(logits_state.z, self.forbidden_token_ids).argmax(dim=-1)
        u_token_ids = masked_logits(logits_state.u, self.forbidden_token_ids).argmax(dim=-1)
        return (
            EditableState(
                torch.nn.functional.embedding(z_token_ids, vocabulary),
                torch.nn.functional.embedding(u_token_ids, vocabulary),
                baseline.z0,
                baseline.u0,
            ),
            z_token_ids,
            u_token_ids,
        )

    @staticmethod
    def _evaluate(objective: AttackObjective, state: EditableState, ledger: BudgetLedger) -> ObjectiveValue:
        ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
        return objective.evaluate(state, include_fol=False)

    def _snapshot(
        self,
        checkpoint: int,
        objective: AttackObjective,
        logits_state: EditableState,
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> GBDASnapshot:
        temperature = linear_temperature(
            min(max(ledger.updates - 1, 0), ledger.update_limit - 1),
            ledger.update_limit,
            start=self.temperature_start,
            end=self.temperature_end,
        )
        soft_state = self._gumbel_state(logits_state, baseline, temperature)
        value = self._evaluate(objective, soft_state, ledger)
        hard_state, z_token_ids, u_token_ids = self._hard_state(logits_state, baseline)
        return GBDASnapshot(
            checkpoint=checkpoint,
            state=_clone_detached_state(hard_state),
            soft_state=_clone_detached_state(soft_state),
            logits_state=_clone_detached_state(logits_state),
            z_token_ids=z_token_ids.detach().clone(),
            u_token_ids=u_token_ids.detach().clone(),
            temperature=temperature,
            attack_loss=float(value.attack_loss.detach().cpu()),
            maximize=float(value.maximize.detach().cpu()),
            updates=ledger.updates,
            branch_updates=MappingProxyType(dict(ledger.branch_updates)),
            forward_passes=ledger.forward_passes,
            backward_passes=ledger.backward_passes,
            hvp_calls=ledger.hvp_calls,
        )
