"""Tensor-only coordinate-gradient GCG mechanics for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState, ObjectiveValue
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter


EditableName = Literal["z", "u"]


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


@dataclass(frozen=True)
class _Candidate:
    editable: EditableName
    batch_index: int
    position_index: int
    token_id: int
    score: float


@dataclass(frozen=True)
class GCGSnapshot:
    """A detached GCG checkpoint containing hard IDs and their frozen embeddings."""

    checkpoint: int
    state: EditableState
    z_token_ids: torch.Tensor
    u_token_ids: torch.Tensor
    attack_loss: float
    maximize: float
    updates: int
    candidates_attempted: int
    candidates_accepted: int
    branch_updates: Mapping[str, int]
    forward_passes: int
    backward_passes: int
    hvp_calls: int


def _clone_ids(ids: torch.Tensor) -> torch.Tensor:
    return ids.detach().clone()


def _clone_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone(),
        u=state.u.detach().clone(),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


@dataclass(frozen=True)
class GCGOptimizer:
    """Exact-budget greedy coordinate updates over token IDs.

    ``initial_state.z`` and ``initial_state.u`` are integer tensors of shape
    ``[batch, positions]``. The accompanying anchor fields are intentionally
    ignored: this optimizer derives frozen embedding anchors from the supplied
    initial IDs so that ``AttackObjective`` receives its normal embedding state.
    """

    embedding: torch.Tensor
    forbidden_token_ids: tuple[int, ...] = ()
    search_width: int = 32
    top_k: int = 256
    candidate_batch_size: int = 8
    initial_z_token_ids: torch.Tensor | None = None
    initial_u_token_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2 or not self.embedding.is_floating_point():
            raise ValueError("embedding must be a floating point [vocabulary, dim] tensor")
        _allowed_token_mask(self.embedding.shape[0], self.forbidden_token_ids, self.embedding.device)
        if not isinstance(self.search_width, int) or self.search_width <= 0:
            raise ValueError("search_width must be a positive integer")
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(self.candidate_batch_size, int) or self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be a positive integer")

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[GCGSnapshot]:
        z_ids, u_ids = self._initial_ids(initial_state)
        baseline = EditableState(
            initial_state.z.detach().clone(),
            initial_state.u.detach().clone(),
            initial_state.z0.detach().clone(),
            initial_state.u0.detach().clone(),
        )
        snapshots: list[GCGSnapshot] = []

        if emitter.due(0):
            snapshots.append(self._snapshot(0, objective, z_ids, u_ids, baseline, ledger))
        for step in range(1, ledger.update_limit + 1):
            ledger.consume_update()
            candidates, current_loss = self._coordinate_candidates(objective, z_ids, u_ids, baseline, ledger)
            remaining = ledger.candidate_limit - ledger.candidates_attempted
            best_z, best_u = z_ids, u_ids
            best_loss = current_loss
            selected = candidates[: min(self.search_width, remaining)]
            scored = self._score_candidates(objective, z_ids, u_ids, selected, baseline, ledger)
            for candidate, candidate_z, candidate_u, candidate_loss in scored:
                if candidate_loss > best_loss:
                    best_z, best_u, best_loss = candidate_z, candidate_u, candidate_loss
            if best_z is not z_ids or best_u is not u_ids:
                ledger.candidates_accepted += 1
                z_ids, u_ids = best_z, best_u
            if emitter.due(step):
                snapshots.append(self._snapshot(step, objective, z_ids, u_ids, baseline, ledger))
        return snapshots

    def _score_candidates(
        self,
        objective: AttackObjective,
        z_ids: torch.Tensor,
        u_ids: torch.Tensor,
        candidates: Sequence[_Candidate],
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> list[tuple[_Candidate, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Score GCG proposals in bounded batches when the objective supports it."""
        evaluate_candidates = getattr(objective, "evaluate_candidates", None)
        can_batch = callable(evaluate_candidates) and z_ids.shape[0] == 1
        scored: list[tuple[_Candidate, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for offset in range(0, len(candidates), self.candidate_batch_size if can_batch else 1):
            group = candidates[offset : offset + (self.candidate_batch_size if can_batch else 1)]
            proposals = [(*self._apply_candidate(z_ids, u_ids, candidate), candidate) for candidate in group]
            if can_batch:
                batch_z = torch.cat([proposal[0] for proposal in proposals], dim=0)
                batch_u = torch.cat([proposal[1] for proposal in proposals], dim=0)
                candidate_state = self._state_from_ids(
                    batch_z,
                    batch_u,
                    baseline.z0.expand(len(group), -1, -1),
                    baseline.u0.expand(len(group), -1, -1),
                )
                candidate_losses = evaluate_candidates(candidate_state).detach()
                if candidate_losses.shape != (len(group),):
                    raise ValueError("batched candidate objective must return one score per candidate")
                ledger.record_forward()
            else:
                candidate_state = self._state_from_ids(proposals[0][0], proposals[0][1], baseline.z0, baseline.u0)
                candidate_losses = self._evaluate(objective, candidate_state, ledger).attack_loss.detach().reshape(1)
            for (candidate_z, candidate_u, candidate), candidate_loss in zip(proposals, candidate_losses):
                ledger.consume_candidate()
                scored.append((candidate, candidate_z, candidate_u, candidate_loss))
        return scored

    def _initial_ids(self, initial_state: EditableState) -> tuple[torch.Tensor, torch.Tensor]:
        if (self.initial_z_token_ids is None) != (self.initial_u_token_ids is None):
            raise ValueError("GCG requires both initial z and u token ID tensors")
        if self.initial_z_token_ids is None:
            return self._validate_token_ids(initial_state.z, initial_state.u)
        return self._validate_token_ids(self.initial_z_token_ids, self.initial_u_token_ids)

    def _validate_token_ids(self, z_ids: torch.Tensor, u_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_ids, u_ids = _clone_ids(z_ids), _clone_ids(u_ids)
        vocabulary_size = self.embedding.shape[0]
        for ids, name in ((z_ids, "z"), (u_ids, "u")):
            if ids.ndim != 2 or ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
                raise ValueError(f"GCG editable {name} must be integer [batch, positions] token IDs")
            if ids.numel() == 0 or (ids < 0).any() or (ids >= vocabulary_size).any():
                raise ValueError(f"GCG editable {name} contains a token ID outside the vocabulary")
        if z_ids.shape[0] != u_ids.shape[0]:
            raise ValueError("GCG z/u IDs must share a batch dimension")
        allowed = _allowed_token_mask(vocabulary_size, self.forbidden_token_ids, z_ids.device)
        if not allowed[z_ids].all() or not allowed[u_ids].all():
            raise ValueError("GCG initial IDs must not contain forbidden tokens")
        return z_ids.to(device=self.embedding.device, dtype=torch.long), u_ids.to(device=self.embedding.device, dtype=torch.long)

    def _vocabulary(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.embedding.detach().to(device=device, dtype=dtype)

    def _state_from_ids(
        self,
        z_ids: torch.Tensor,
        u_ids: torch.Tensor,
        z0: torch.Tensor,
        u0: torch.Tensor,
    ) -> EditableState:
        return EditableState(
            z=self._embedding_from_ids(z_ids),
            u=self._embedding_from_ids(u_ids),
            z0=z0,
            u0=u0,
        )

    def _embedding_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        vocabulary = self._vocabulary(ids.device, self.embedding.dtype)
        return torch.nn.functional.embedding(ids, vocabulary)

    def _one_hot_state(
        self,
        z_ids: torch.Tensor,
        u_ids: torch.Tensor,
        baseline: EditableState,
    ) -> tuple[EditableState, torch.Tensor, torch.Tensor]:
        vocabulary = self._vocabulary(z_ids.device, self.embedding.dtype)
        z_one_hot = torch.nn.functional.one_hot(z_ids, vocabulary.shape[0]).to(vocabulary.dtype).requires_grad_(True)
        u_one_hot = torch.nn.functional.one_hot(u_ids, vocabulary.shape[0]).to(vocabulary.dtype).requires_grad_(True)
        return EditableState(z_one_hot @ vocabulary, u_one_hot @ vocabulary, baseline.z0, baseline.u0), z_one_hot, u_one_hot

    def _coordinate_candidates(
        self,
        objective: AttackObjective,
        z_ids: torch.Tensor,
        u_ids: torch.Tensor,
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> tuple[list[_Candidate], torch.Tensor]:
        state, z_one_hot, u_one_hot = self._one_hot_state(z_ids, u_ids, baseline)
        value = self._evaluate(objective, state, ledger)
        z_gradient, u_gradient = torch.autograd.grad(value.attack_loss, (z_one_hot, u_one_hot))
        ledger.record_backward()
        allowed = _allowed_token_mask(self.embedding.shape[0], self.forbidden_token_ids, z_ids.device)
        candidates = self._top_candidates("z", z_gradient, z_ids, allowed)
        candidates.extend(self._top_candidates("u", u_gradient, u_ids, allowed))
        candidates.sort(
            key=lambda item: (
                -item.score,
                0 if item.editable == "z" else 1,
                item.batch_index,
                item.position_index,
                item.token_id,
            )
        )
        return candidates, value.attack_loss.detach()

    def _top_candidates(
        self,
        editable: EditableName,
        gradient: torch.Tensor,
        ids: torch.Tensor,
        allowed: torch.Tensor,
    ) -> list[_Candidate]:
        masked = gradient.detach().masked_fill(~allowed.view(1, 1, -1), -torch.inf)
        masked.scatter_(-1, ids.unsqueeze(-1), -torch.inf)
        candidates: list[_Candidate] = []
        for batch_index in range(ids.shape[0]):
            for position_index in range(ids.shape[1]):
                scores = masked[batch_index, position_index]
                available = int(torch.isfinite(scores).sum().item())
                if not available:
                    continue
                limit = min(self.top_k, available)
                cutoff = torch.topk(scores, k=limit, largest=True, sorted=False).values.min()
                higher = torch.nonzero(scores > cutoff, as_tuple=False).flatten()
                ties = torch.nonzero(scores == cutoff, as_tuple=False).flatten()
                selected = torch.cat((higher, ties[: limit - higher.numel()]))
                # ``argsort(..., stable=True)`` previously ranked ties by their
                # vocabulary index. Preserve that deterministic policy while
                # sorting only the bounded selected set.
                ranked = sorted(
                    (int(token_id) for token_id in selected.cpu().tolist()),
                    key=lambda token_id: (-float(scores[token_id].cpu()), token_id),
                )
                for token_id in ranked:
                    candidates.append(_Candidate(editable, batch_index, position_index, token_id, float(scores[token_id].cpu())))
        return candidates

    @staticmethod
    def _apply_candidate(z_ids: torch.Tensor, u_ids: torch.Tensor, candidate: _Candidate) -> tuple[torch.Tensor, torch.Tensor]:
        next_z, next_u = z_ids.clone(), u_ids.clone()
        target = next_z if candidate.editable == "z" else next_u
        target[candidate.batch_index, candidate.position_index] = candidate.token_id
        return next_z, next_u

    @staticmethod
    def _evaluate(objective: AttackObjective, state: EditableState, ledger: BudgetLedger) -> ObjectiveValue:
        ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
        return objective.evaluate(state, include_fol=False)

    def _snapshot(
        self,
        checkpoint: int,
        objective: AttackObjective,
        z_ids: torch.Tensor,
        u_ids: torch.Tensor,
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> GCGSnapshot:
        state = self._state_from_ids(z_ids, u_ids, baseline.z0, baseline.u0)
        value = self._evaluate(objective, state, ledger)
        return GCGSnapshot(
            checkpoint=checkpoint,
            state=_clone_state(state),
            z_token_ids=_clone_ids(z_ids),
            u_token_ids=_clone_ids(u_ids),
            attack_loss=float(value.attack_loss.detach().cpu()),
            maximize=float(value.maximize.detach().cpu()),
            updates=ledger.updates,
            candidates_attempted=ledger.candidates_attempted,
            candidates_accepted=ledger.candidates_accepted,
            branch_updates=MappingProxyType(dict(ledger.branch_updates)),
            forward_passes=ledger.forward_passes,
            backward_passes=ledger.backward_passes,
            hvp_calls=ledger.hvp_calls,
        )
