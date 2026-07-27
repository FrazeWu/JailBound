"""Shared mechanical accounting for every optimizer identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    update_limit: int
    candidate_limit: int
    branch_limits: Mapping[str, int] = field(default_factory=dict)
    updates: int = 0
    candidates_attempted: int = 0
    candidates_accepted: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    hvp_calls: int = 0
    branch_updates: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.branch_updates = {name: 0 for name in self.branch_limits}

    def consume_update(self, branch: str | None = None) -> None:
        if self.updates >= self.update_limit:
            raise BudgetExceeded("update budget exhausted")
        if branch is not None:
            if branch not in self.branch_limits:
                raise BudgetExceeded(f"unknown optimization branch: {branch}")
            if self.branch_updates[branch] >= self.branch_limits[branch]:
                raise BudgetExceeded(f"branch budget exhausted: {branch}")
        self.updates += 1
        if branch is not None:
            self.branch_updates[branch] += 1

    def consume_candidate(self, *, accepted: bool = False) -> None:
        if self.candidates_attempted >= self.candidate_limit:
            raise BudgetExceeded("candidate budget exhausted")
        self.candidates_attempted += 1
        self.candidates_accepted += int(accepted)

    def record_forward(self, count: int = 1) -> None:
        self.forward_passes += self._positive_count(count)

    def record_backward(self, count: int = 1) -> None:
        self.backward_passes += self._positive_count(count)

    def record_hvp(self, count: int = 1) -> None:
        self.hvp_calls += self._positive_count(count)

    @staticmethod
    def _positive_count(count: int) -> int:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        return count


@dataclass
class CheckpointEmitter:
    checkpoints: list[int]
    _emitted: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.checkpoints != sorted(set(self.checkpoints)):
            raise ValueError("checkpoints must be unique and ordered")
        if 0 not in self.checkpoints:
            raise ValueError("checkpoint 0 is required")

    def due(self, step: int) -> bool:
        if step not in self.checkpoints or step in self._emitted:
            return False
        self._emitted.add(step)
        return True
