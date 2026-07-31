"""GBDA mechanics adapted from facebookresearch/text-adversarial-attack."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from benchmark.safety_eval.objective import AttackObjective, EditableState
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.safety_eval.optimizers.gbda import masked_logits


def build_official_logit_state(
    z_token_ids: torch.Tensor,
    u_token_ids: torch.Tensor,
    *,
    vocabulary_size: int,
    device: torch.device,
    initial_coefficient: float = 15.0,
) -> EditableState:
    """Initialize original-token coefficients to 15 over zero logits."""
    if vocabulary_size < 1 or initial_coefficient <= 0:
        raise ValueError("vocabulary size and initial coefficient must be positive")
    for token_ids in (z_token_ids, u_token_ids):
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("official GBDA token IDs must have shape [1, positions]")
        if token_ids.dtype.is_floating_point or token_ids.dtype.is_complex:
            raise ValueError("official GBDA token IDs must be integral")
        if bool(((token_ids < 0) | (token_ids >= vocabulary_size)).any()):
            raise ValueError("official GBDA token ID is outside the vocabulary")

    def initialize(token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, vocabulary_size), dtype=torch.float32, device=device)
        logits.scatter_(-1, token_ids.to(device=device, dtype=torch.long).unsqueeze(-1), initial_coefficient)
        return logits

    z_logits = initialize(z_token_ids)
    u_logits = initialize(u_token_ids)
    return EditableState(
        z=z_logits,
        u=u_logits,
        z0=z_logits.detach().clone(),
        u0=u_logits.detach().clone(),
    )


@dataclass(frozen=True)
class OfficialGBDASnapshot:
    checkpoint: int
    state: EditableState
    logits_state: EditableState
    z_token_ids: torch.Tensor
    u_token_ids: torch.Tensor
    temperature: float
    hard_samples: int
    attack_loss: float
    maximize: float
    updates: int
    branch_updates: Mapping[str, int]
    candidates_attempted: int
    candidates_accepted: int
    forward_passes: int
    backward_passes: int
    hvp_calls: int


def _detached_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone(),
        u=state.u.detach().clone(),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


@dataclass(frozen=True)
class OfficialGBDAOptimizer:
    """Adam over token logits using the official GBDA sampling schedule.

    The upstream classifier loss is replaced by the benchmark's shared causal-LM
    objective. Optimization mechanics retain the upstream defaults: coefficient
    15 initialization, Adam at 0.3, ten soft samples per update, temperature 1,
    and one hundred hard samples after the final update.
    """

    embedding: torch.Tensor
    forbidden_token_ids: tuple[int, ...] = ()
    learning_rate: float = 0.3
    soft_samples_per_update: int = 10
    soft_sample_batch_size: int = 5
    hard_samples: int = 100
    hard_sample_batch_size: int = 10
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2 or not self.embedding.is_floating_point():
            raise ValueError("embedding must be a floating [vocabulary, hidden] tensor")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("learning rate and temperature must be positive")
        if (
            self.soft_samples_per_update < 1
            or self.soft_sample_batch_size < 1
            or self.hard_samples < 1
            or self.hard_sample_batch_size < 1
        ):
            raise ValueError("official GBDA sample counts must be positive")
        probe = torch.zeros((1, 1, self.embedding.shape[0]), dtype=torch.float32, device=self.embedding.device)
        masked_logits(probe, self.forbidden_token_ids)

    def run(
        self,
        objective: AttackObjective,
        initial_state: EditableState,
        ledger: BudgetLedger,
        emitter: CheckpointEmitter,
    ) -> list[OfficialGBDASnapshot]:
        logits_state = self._live_logits(initial_state)
        baseline = self._initial_baseline(logits_state)
        optimizer = torch.optim.Adam((logits_state.z, logits_state.u), lr=self.learning_rate)
        snapshots: list[OfficialGBDASnapshot] = []

        if emitter.due(0):
            snapshots.append(self._deterministic_snapshot(0, objective, logits_state, baseline, ledger))
        for step in range(1, ledger.update_limit + 1):
            ledger.consume_update()
            optimizer.zero_grad(set_to_none=True)
            remaining = self.soft_samples_per_update
            while remaining:
                batch_size = min(self.soft_sample_batch_size, remaining)
                soft_state = self._soft_state(logits_state, baseline, sample_count=batch_size)
                for _ in range(batch_size):
                    ledger.consume_candidate()
                maximize = self._evaluate_soft_batch(objective, soft_state, ledger)
                weight = batch_size / self.soft_samples_per_update
                (-maximize * weight).backward()
                ledger.record_backward()
                remaining -= batch_size
            optimizer.step()
            if emitter.due(step):
                if step == ledger.update_limit:
                    snapshots.append(self._sampled_snapshot(step, objective, logits_state, baseline, ledger))
                else:
                    snapshots.append(self._deterministic_snapshot(step, objective, logits_state, baseline, ledger))
        return snapshots

    def _live_logits(self, state: EditableState) -> EditableState:
        vocabulary_size = self.embedding.shape[0]
        if state.z.ndim != 3 or state.u.ndim != 3 or state.z.shape[-1] != vocabulary_size or state.u.shape[-1] != vocabulary_size:
            raise ValueError("official GBDA logits must have shape [batch, positions, vocabulary]")
        if state.z.shape[0] != 1 or state.u.shape[0] != 1:
            raise ValueError("official GBDA optimizes one prompt at a time")
        return EditableState(
            z=state.z.detach().to(dtype=torch.float32).clone().requires_grad_(True),
            u=state.u.detach().to(dtype=torch.float32).clone().requires_grad_(True),
            z0=state.z0.detach().to(dtype=torch.float32).clone(),
            u0=state.u0.detach().to(dtype=torch.float32).clone(),
        )

    def _initial_baseline(self, logits_state: EditableState) -> EditableState:
        vocabulary = self.embedding.detach()

        def expected_embedding(logits: torch.Tensor) -> torch.Tensor:
            probabilities = torch.softmax(masked_logits(logits, self.forbidden_token_ids), dim=-1)
            return (probabilities.to(dtype=vocabulary.dtype) @ vocabulary).detach()

        z0 = expected_embedding(logits_state.z)
        u0 = expected_embedding(logits_state.u)
        return EditableState(z=z0, u=u0, z0=z0.clone(), u0=u0.clone())

    def _combined_logits(self, logits_state: EditableState) -> torch.Tensor:
        return masked_logits(torch.cat((logits_state.z, logits_state.u), dim=1), self.forbidden_token_ids)

    def _soft_state(
        self,
        logits_state: EditableState,
        baseline: EditableState,
        *,
        sample_count: int,
    ) -> EditableState:
        combined = self._combined_logits(logits_state).expand(sample_count, -1, -1)
        weights = torch.nn.functional.gumbel_softmax(combined, tau=self.temperature, hard=False, dim=-1)
        embeddings = weights.to(dtype=self.embedding.dtype) @ self.embedding.detach()
        z_positions = logits_state.z.shape[1]
        return EditableState(
            z=embeddings[:, :z_positions],
            u=embeddings[:, z_positions:],
            z0=baseline.z0,
            u0=baseline.u0,
        )

    @staticmethod
    def _evaluate_soft_batch(objective: AttackObjective, state: EditableState, ledger: BudgetLedger) -> torch.Tensor:
        optimization_loss = getattr(objective, "optimization_loss", None)
        if callable(optimization_loss):
            ledger.record_forward()
            maximize = optimization_loss(state)
        else:
            ledger.record_forward(getattr(objective, "forward_passes_per_evaluation", 1))
            maximize = objective.evaluate(state, include_fol=False).maximize
        batch_size = state.z.shape[0]
        if batch_size == 1:
            return maximize
        gamma_z = float(getattr(objective, "gamma_z", 0.0))
        gamma_u = float(getattr(objective, "gamma_u", 0.0))
        penalty = gamma_z * (state.z - state.z0).square().sum() + gamma_u * (state.u - state.u0).square().sum()
        correction = penalty * (1.0 - 1.0 / batch_size)
        return maximize + correction

    def _ids_to_state(
        self,
        token_ids: torch.Tensor,
        *,
        z_positions: int,
        baseline: EditableState,
    ) -> EditableState:
        embeddings = torch.nn.functional.embedding(token_ids, self.embedding.detach())
        return EditableState(
            z=embeddings[:, :z_positions],
            u=embeddings[:, z_positions:],
            z0=baseline.z0,
            u0=baseline.u0,
        )

    @staticmethod
    def _candidate_scores(objective: Any, state: EditableState) -> torch.Tensor:
        scorer = getattr(objective, "evaluate_candidates", None)
        if not callable(scorer):
            if state.z.shape[0] != 1:
                raise TypeError("official GBDA hard-sample selection requires evaluate_candidates")
            return objective.evaluate(state, include_fol=False).attack_loss.reshape(1)
        return scorer(state)

    def _deterministic_snapshot(
        self,
        checkpoint: int,
        objective: Any,
        logits_state: EditableState,
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> OfficialGBDASnapshot:
        token_ids = self._combined_logits(logits_state).argmax(dim=-1)
        state = self._ids_to_state(token_ids, z_positions=logits_state.z.shape[1], baseline=baseline)
        with torch.no_grad():
            score = self._candidate_scores(objective, state)[0]
        ledger.record_forward()
        return self._snapshot(checkpoint, state, token_ids, logits_state, ledger, float(score.detach().cpu()), 0)

    def _sampled_snapshot(
        self,
        checkpoint: int,
        objective: Any,
        logits_state: EditableState,
        baseline: EditableState,
        ledger: BudgetLedger,
    ) -> OfficialGBDASnapshot:
        combined = self._combined_logits(logits_state)
        z_positions = logits_state.z.shape[1]
        best_score: torch.Tensor | None = None
        best_ids: torch.Tensor | None = None
        remaining = self.hard_samples
        while remaining:
            batch_size = min(self.hard_sample_batch_size, remaining)
            weights = torch.nn.functional.gumbel_softmax(
                combined.expand(batch_size, -1, -1), tau=self.temperature, hard=True, dim=-1
            )
            token_ids = weights.argmax(dim=-1)
            state = self._ids_to_state(token_ids, z_positions=z_positions, baseline=baseline)
            with torch.no_grad():
                scores = self._candidate_scores(objective, state)
            ledger.record_forward()
            for _ in range(batch_size):
                ledger.consume_candidate()
            index = int(scores.argmax().item())
            score = scores[index]
            if best_score is None or bool(score > best_score):
                best_score = score.detach()
                best_ids = token_ids[index : index + 1].detach().clone()
            remaining -= batch_size
        if best_score is None or best_ids is None:
            raise RuntimeError("official GBDA did not produce a hard candidate")
        ledger.candidates_accepted += 1
        selected = self._ids_to_state(best_ids, z_positions=z_positions, baseline=baseline)
        return self._snapshot(
            checkpoint,
            selected,
            best_ids,
            logits_state,
            ledger,
            float(best_score.cpu()),
            self.hard_samples,
        )

    def _snapshot(
        self,
        checkpoint: int,
        state: EditableState,
        token_ids: torch.Tensor,
        logits_state: EditableState,
        ledger: BudgetLedger,
        score: float,
        hard_samples: int,
    ) -> OfficialGBDASnapshot:
        z_positions = logits_state.z.shape[1]
        return OfficialGBDASnapshot(
            checkpoint=checkpoint,
            state=_detached_state(state),
            logits_state=_detached_state(logits_state),
            z_token_ids=token_ids[:, :z_positions].detach().clone(),
            u_token_ids=token_ids[:, z_positions:].detach().clone(),
            temperature=self.temperature,
            hard_samples=hard_samples,
            attack_loss=score,
            maximize=score,
            updates=ledger.updates,
            branch_updates=MappingProxyType(dict(ledger.branch_updates)),
            candidates_attempted=ledger.candidates_attempted,
            candidates_accepted=ledger.candidates_accepted,
            forward_passes=ledger.forward_passes,
            backward_passes=ledger.backward_passes,
            hvp_calls=ledger.hvp_calls,
        )
