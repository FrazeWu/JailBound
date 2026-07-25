"""
Attack State Module

Defines the core state representation for the QuoTe v2 embedding-space attack
optimisation pipeline.

The continuous attack state is  u = [z; U]  where:
  z  = soft prefix tensor (1, P, d)          — learned from scratch
  U  = editable seed embedding block (1, S, d) — initialised from seed token embeddings
  U₀ = initial seed block (frozen reference)

``AttackStatePool`` manages the two search branches (high-value and boundary).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_states(states: list[AttackState], path: str | Path) -> None:
    """Save a list of AttackStates (with tensors) to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [s.to_saveable() for s in states]
    torch.save(payloads, path)
    logger.info("Saved %d states to %s (%.1f MB)",
                len(states), path, path.stat().st_size / 1e6)


def load_states(path: str | Path) -> list[AttackState]:
    """Load states from a .pt file produced by save_states()."""
    payloads = torch.load(path, map_location="cpu", weights_only=False)
    states = [AttackState.from_saved(d) for d in payloads]
    logger.info("Loaded %d states from %s", len(states), path)
    return states


@dataclass
class AttackState:
    """One candidate in the bi-end search.

    Attributes:
        meta_prompt:           Upstream meta-attack-prompt text (fixed per condition).
        original_seed:         Original seed prompt (immutable reference).
        mutated_seed:          Current seed text (may equal original_seed — no text mutation).
        soft_prefix:           Learnable continuous prefix z (1, P, d). None before init.
        editable_seed_block:   Learnable seed embedding block U (1, S, d). None before init.
        initial_seed_block:    Frozen reference U₀ (1, S, d). None before init.
        frozen_scaffold_ids:   Token IDs for the frozen scaffold E(x)_{Ω̄_s} (1, T_frozen).
        risk_score:            Real risk from external judge [0, 1]. -1 if unset.
        proxy_risk:            Differentiable surrogate risk r̃(u). -1 if unset.
        zol:                   Zero-order = r̃(u).
        fol:                   First-order = ε · ‖∇_u r̃‖₂.
        branch_type:           "high_value" or "boundary".
        step:                  Optimisation step at which this state was recorded.
        condition_id:          Identifier for the (threat × attack) condition cell.
        behavior_id:           Upstream behavior ID for traceability.
        prefix_norm:           ‖z‖_F.
        seed_block_drift:      ‖U − U₀‖_F².
    """

    meta_prompt: str = ""
    original_seed: str = ""
    mutated_seed: str = ""

    soft_prefix: torch.Tensor | None = None
    editable_seed_block: torch.Tensor | None = None
    initial_seed_block: torch.Tensor | None = None
    frozen_scaffold_ids: torch.Tensor | None = None

    risk_score: float = -1.0
    proxy_risk: float = -1.0
    zol: float = -1.0
    fol: float = -1.0

    branch_type: str = "high_value"
    step: int = 0
    condition_id: str = ""
    behavior_id: str = ""
    prefix_norm: float = 0.0
    seed_block_drift: float = 0.0

    # ------------------------------------------------------------------
    # Branch objective scores
    # ------------------------------------------------------------------

    def high_value_score(self, lambda_fol: float, gamma_z: float, gamma_u: float) -> float:
        """O⁻ = r̃ − λ·FOL − γ_z·‖z‖² − γ_u·‖U−U₀‖²."""
        r = self.proxy_risk if self.proxy_risk >= 0 else self.risk_score
        return r - lambda_fol * self.fol - gamma_z * self.prefix_norm ** 2 - gamma_u * self.seed_block_drift

    def boundary_score(self, lambda_fol: float, gamma_z: float, gamma_u: float) -> float:
        """O⁺ = r̃ + λ·FOL − γ_z·‖z‖² − γ_u·‖U−U₀‖²."""
        r = self.proxy_risk if self.proxy_risk >= 0 else self.risk_score
        return r + lambda_fol * self.fol - gamma_z * self.prefix_norm ** 2 - gamma_u * self.seed_block_drift

    def branch_score(self, lambda_fol: float, gamma_z: float, gamma_u: float) -> float:
        if self.branch_type == "high_value":
            return self.high_value_score(lambda_fol, gamma_z, gamma_u)
        return self.boundary_score(lambda_fol, gamma_z, gamma_u)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_prompt": self.meta_prompt,
            "original_seed": self.original_seed,
            "mutated_seed": self.mutated_seed,
            "risk_score": self.risk_score,
            "proxy_risk": self.proxy_risk,
            "zol": self.zol,
            "fol": self.fol,
            "branch_type": self.branch_type,
            "step": self.step,
            "condition_id": self.condition_id,
            "behavior_id": self.behavior_id,
            "prefix_norm": self.prefix_norm,
            "seed_block_drift": self.seed_block_drift,
        }

    def to_saveable(self) -> dict[str, Any]:
        """Serialise including tensor fields (for torch.save)."""
        d = self.to_dict()
        d["soft_prefix"] = self.soft_prefix.detach().cpu() if self.soft_prefix is not None else None
        d["editable_seed_block"] = self.editable_seed_block.detach().cpu() if self.editable_seed_block is not None else None
        d["initial_seed_block"] = self.initial_seed_block.detach().cpu() if self.initial_seed_block is not None else None
        d["frozen_scaffold_ids"] = self.frozen_scaffold_ids.detach().cpu() if self.frozen_scaffold_ids is not None else None
        return d

    @classmethod
    def from_saved(cls, d: dict[str, Any]) -> AttackState:
        """Reconstruct from a dict produced by to_saveable()."""
        return cls(
            meta_prompt=d.get("meta_prompt", ""),
            original_seed=d.get("original_seed", ""),
            mutated_seed=d.get("mutated_seed", ""),
            soft_prefix=d.get("soft_prefix"),
            editable_seed_block=d.get("editable_seed_block"),
            initial_seed_block=d.get("initial_seed_block"),
            frozen_scaffold_ids=d.get("frozen_scaffold_ids"),
            risk_score=d.get("risk_score", -1.0),
            proxy_risk=d.get("proxy_risk", -1.0),
            zol=d.get("zol", -1.0),
            fol=d.get("fol", -1.0),
            branch_type=d.get("branch_type", "high_value"),
            step=d.get("step", 0),
            condition_id=d.get("condition_id", ""),
            behavior_id=d.get("behavior_id", ""),
            prefix_norm=d.get("prefix_norm", 0.0),
            seed_block_drift=d.get("seed_block_drift", 0.0),
        )

    def clone(self) -> AttackState:
        """Deep-copy, cloning tensor fields."""
        new = copy.copy(self)
        if self.soft_prefix is not None:
            new.soft_prefix = self.soft_prefix.detach().clone()
        if self.editable_seed_block is not None:
            new.editable_seed_block = self.editable_seed_block.detach().clone()
        if self.initial_seed_block is not None:
            new.initial_seed_block = self.initial_seed_block.detach().clone()
        if self.frozen_scaffold_ids is not None:
            new.frozen_scaffold_ids = self.frozen_scaffold_ids.clone()
        return new


# =====================================================================
# Pool
# =====================================================================


class AttackStatePool:
    """Manages candidate pools for both search branches."""

    def __init__(self, max_size: int, lambda_fol: float, gamma_z: float, gamma_u: float) -> None:
        self.max_size = max_size
        self.lambda_fol = lambda_fol
        self.gamma_z = gamma_z
        self.gamma_u = gamma_u
        self.high_value: list[AttackState] = []
        self.boundary: list[AttackState] = []

    def add(self, state: AttackState) -> None:
        pool = self.high_value if state.branch_type == "high_value" else self.boundary
        pool.append(state.clone())
        pool.sort(
            key=lambda s: s.branch_score(self.lambda_fol, self.gamma_z, self.gamma_u),
            reverse=True,
        )
        if len(pool) > self.max_size:
            pool.pop()

    def best(self, branch: str = "high_value") -> AttackState | None:
        pool = self.high_value if branch == "high_value" else self.boundary
        return pool[0] if pool else None

    def all_states(self) -> list[AttackState]:
        return list(self.high_value) + list(self.boundary)

    def __len__(self) -> int:
        return len(self.high_value) + len(self.boundary)
