"""Tensor-only PEZ mechanics for controlled safety experiments."""

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


def straight_through_project(
    soft: torch.Tensor,
    embedding: torch.Tensor,
    forbidden_token_ids: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project `[batch, positions, dim]` embeddings to allowed vocab rows via cosine similarity.

    The returned embedding has exact hard-token values on the forward pass and an
    identity straight-through gradient with respect to ``soft``.
    """

    if soft.ndim != 3:
        raise ValueError("soft embeddings must have shape [batch, positions, dim]")
    if embedding.ndim != 2 or embedding.shape[1] != soft.shape[-1]:
        raise ValueError("embedding must have shape [vocabulary, dim] matching soft")
    if not soft.is_floating_point() or not embedding.is_floating_point():
        raise ValueError("soft embeddings and vocabulary embedding must be floating point")

    vocabulary = embedding.detach().to(device=soft.device, dtype=soft.dtype)
    allowed = _allowed_token_mask(vocabulary.shape[0], forbidden_token_ids, soft.device)
    normalized_soft = torch.nn.functional.normalize(soft, dim=-1, eps=1e-12)
    normalized_vocabulary = torch.nn.functional.normalize(vocabulary, dim=-1, eps=1e-12)
    similarity = torch.einsum("bpd,vd->bpv", normalized_soft, normalized_vocabulary)
    similarity = similarity.masked_fill(~allowed.view(1, 1, -1), -torch.inf)
    token_ids = similarity.argmax(dim=-1)
    hard = torch.nn.functional.embedding(token_ids, vocabulary)
    return hard + soft - soft.detach(), token_ids


@dataclass(frozen=True)
class PEZSnapshot:
    """A detached PEZ checkpoint containing hard tokens and editable soft state."""

    checkpoint: int
    state: EditableState
    soft_state: EditableState
    z_token_ids: torch.Tensor
    u_token_ids: torch.Tensor
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
class PEZOptimizer:
    """Exact-budget Adam optimizer over soft prompt embeddings with PEZ projection."""

    embedding: torch.Tensor
    forbidden_token_ids: tuple[int, ...] = ()
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2 or not self.embedding.is_floating_point():
            raise ValueError("embedding must be a floating point [vocabulary, dim] tensor")
        _allowed_token_mask(self.embedding.shape[0], self.forbidden_token_ids, self.embedding.device)
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[PEZSnapshot]:
        soft_state = _clone_live_state(initial_state)
        optimizer = torch.optim.Adam((soft_state.z, soft_state.u), lr=self.learning_rate)
        snapshots: list[PEZSnapshot] = []

        if emitter.due(0):
            snapshots.append(self._snapshot(0, objective, soft_state, ledger))
        for step in range(1, ledger.update_limit + 1):
            ledger.consume_update()
            optimizer.zero_grad(set_to_none=True)
            projected_state, _, _ = self._project_state(soft_state)
            value = self._evaluate(objective, projected_state, ledger)
            (-value.maximize).backward()
            ledger.record_backward()
            torch.nn.utils.clip_grad_norm_((soft_state.z, soft_state.u), self.max_grad_norm)
            optimizer.step()
            if emitter.due(step):
                snapshots.append(self._snapshot(step, objective, soft_state, ledger))
        return snapshots

    def _project_state(self, soft_state: EditableState) -> tuple[EditableState, torch.Tensor, torch.Tensor]:
        hard_z, z_token_ids = straight_through_project(soft_state.z, self.embedding, self.forbidden_token_ids)
        hard_u, u_token_ids = straight_through_project(soft_state.u, self.embedding, self.forbidden_token_ids)
        return EditableState(hard_z, hard_u, soft_state.z0, soft_state.u0), z_token_ids, u_token_ids

    @staticmethod
    def _evaluate(objective: AttackObjective, state: EditableState, ledger: BudgetLedger) -> ObjectiveValue:
        ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
        return objective.evaluate(state, include_fol=False)

    def _snapshot(
        self,
        checkpoint: int,
        objective: AttackObjective,
        soft_state: EditableState,
        ledger: BudgetLedger,
    ) -> PEZSnapshot:
        projected_state, z_token_ids, u_token_ids = self._project_state(soft_state)
        value = self._evaluate(objective, projected_state, ledger)
        return PEZSnapshot(
            checkpoint=checkpoint,
            state=_clone_detached_state(projected_state),
            soft_state=_clone_detached_state(soft_state),
            z_token_ids=z_token_ids.detach().clone(),
            u_token_ids=u_token_ids.detach().clone(),
            attack_loss=float(value.attack_loss.detach().cpu()),
            maximize=float(value.maximize.detach().cpu()),
            updates=ledger.updates,
            branch_updates=MappingProxyType(dict(ledger.branch_updates)),
            forward_passes=ledger.forward_passes,
            backward_passes=ledger.backward_passes,
            hvp_calls=ledger.hvp_calls,
        )
