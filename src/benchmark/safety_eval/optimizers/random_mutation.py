"""Deterministic, representation-agnostic random-mutation optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .base import BudgetLedger


CandidateT = TypeVar("CandidateT")


class CandidateRewriter(Protocol[CandidateT]):
    """Produce one deterministic candidate proposal from the current candidate."""

    def __call__(self, candidate: CandidateT, /) -> CandidateT: ...


class SemanticAcceptor(Protocol[CandidateT]):
    """Decide whether a proposed candidate preserves the required semantics."""

    def __call__(self, candidate: CandidateT, /) -> bool: ...


class ObjectiveScorer(Protocol[CandidateT]):
    """Return an objective score where larger values are preferred."""

    def __call__(self, candidate: CandidateT, /) -> float: ...


class CandidateCapExceeded(RuntimeError):
    """The candidate cap was reached before 100 semantically accepted updates."""

    def __init__(self, *, updates: int, candidates_attempted: int, candidates_accepted: int) -> None:
        self.updates = updates
        self.candidates_attempted = candidates_attempted
        self.candidates_accepted = candidates_accepted
        super().__init__("candidate cap reached before 100 accepted random-mutation updates")


@dataclass(frozen=True)
class RandomMutationSnapshot(Generic[CandidateT]):
    checkpoint: int
    candidate: CandidateT
    objective_score: float
    updates: int
    candidates_attempted: int
    candidates_accepted: int


@dataclass(frozen=True)
class RandomMutationOptimizer:
    """Run 100 accepted rewrites while retaining the highest-scoring candidate."""

    update_limit: int = 100
    checkpoints: tuple[int, ...] = (0, 25, 50, 100)

    def run(
        self,
        *,
        initial_candidate: CandidateT,
        objective_score: ObjectiveScorer[CandidateT],
        rewriter: CandidateRewriter[CandidateT],
        semantic_acceptor: SemanticAcceptor[CandidateT],
        candidate_limit: int,
    ) -> list[RandomMutationSnapshot[CandidateT]]:
        if self.update_limit != 100 or self.checkpoints != (0, 25, 50, 100):
            raise ValueError("random mutation requires exactly 100 updates and checkpoints 0/25/50/100")

        ledger = BudgetLedger(update_limit=self.update_limit, candidate_limit=candidate_limit)
        current = initial_candidate
        best_candidate = initial_candidate
        best_score = objective_score(initial_candidate)
        snapshots = [self._snapshot(0, best_candidate, best_score, ledger)]

        while ledger.updates < self.update_limit:
            if ledger.candidates_attempted >= ledger.candidate_limit:
                raise CandidateCapExceeded(
                    updates=ledger.updates,
                    candidates_attempted=ledger.candidates_attempted,
                    candidates_accepted=ledger.candidates_accepted,
                )
            proposal = rewriter(current)
            accepted = semantic_acceptor(proposal)
            ledger.consume_candidate(accepted=accepted)
            if not accepted:
                continue

            ledger.consume_update()
            current = proposal
            score = objective_score(proposal)
            if score > best_score:
                best_candidate = proposal
                best_score = score
            if ledger.updates in self.checkpoints:
                snapshots.append(self._snapshot(ledger.updates, best_candidate, best_score, ledger))

        return snapshots

    @staticmethod
    def _snapshot(
        checkpoint: int,
        candidate: CandidateT,
        objective_score: float,
        ledger: BudgetLedger,
    ) -> RandomMutationSnapshot[CandidateT]:
        return RandomMutationSnapshot(
            checkpoint=checkpoint,
            candidate=candidate,
            objective_score=objective_score,
            updates=ledger.updates,
            candidates_attempted=ledger.candidates_attempted,
            candidates_accepted=ledger.candidates_accepted,
        )
