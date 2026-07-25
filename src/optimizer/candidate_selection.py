"""
Candidate Selection Module

Selects optimal attack prompts from the high-value and boundary pools
using a composite score that balances risk, diversity, and FOL.

Supports:
- strongest top-k (pure risk)
- diverse top-k (MMR-style diversity)
- per-condition top-k
- mixed selection from both pools
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.nn.functional as F

from embedding.attack_state import AttackState, AttackStatePool
from config.quote_config import QuoTeConfig

logger = logging.getLogger(__name__)


def _overall_score(
    state: AttackState,
    alpha_score: float = 1.0,
    diversity_bonus: float = 0.0,
) -> float:
    """Composite ranking score: Score = α·R(H) + diversity_bonus.

    Uses proxy_risk (falls back to risk_score if judge has scored).
    """
    r = state.proxy_risk if state.proxy_risk >= 0 else state.risk_score
    return alpha_score * r + diversity_bonus


def _pairwise_diversity(a: AttackState, b: AttackState) -> float:
    """Simple text-level diversity: 1 − Jaccard(tokens_a, tokens_b)."""
    sa = set(a.mutated_seed.lower().split())
    sb = set(b.mutated_seed.lower().split())
    if not sa or not sb:
        return 1.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def _effective_risk(s: AttackState) -> float:
    """Return the best available risk estimate (proxy during search, judge after eval)."""
    return s.proxy_risk if s.proxy_risk >= 0 else s.risk_score


def select_strongest(candidates: list[AttackState], top_k: int) -> list[AttackState]:
    """Select top-k by risk score (descending). Uses proxy_risk when judge score unavailable."""
    ranked = sorted(candidates, key=lambda s: _effective_risk(s), reverse=True)
    return ranked[:top_k]


def select_diverse(
    candidates: list[AttackState],
    top_k: int,
    diversity_weight: float = 0.1,
) -> list[AttackState]:
    """MMR-style diverse selection: greedily pick candidates that are high-risk
    yet different from already-selected ones."""
    if not candidates:
        return []
    selected: list[AttackState] = []
    remaining = list(candidates)

    # First pick: highest risk
    remaining.sort(key=lambda s: _effective_risk(s), reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_score = float("-inf")
        for i, cand in enumerate(remaining):
            # Max diversity to any already-selected
            min_div = min(_pairwise_diversity(cand, sel) for sel in selected)
            score = _effective_risk(cand) + diversity_weight * min_div
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def select_per_condition(
    candidates: list[AttackState],
    top_k_per_condition: int = 3,
) -> list[AttackState]:
    """Select top-k per condition_id."""
    by_cond: dict[str, list[AttackState]] = defaultdict(list)
    for s in candidates:
        by_cond[s.condition_id].append(s)
    result: list[AttackState] = []
    for cond, states in by_cond.items():
        states.sort(key=lambda s: _effective_risk(s), reverse=True)
        result.extend(states[:top_k_per_condition])
    return result


def select_optimal_attacks(
    pools: list[AttackStatePool],
    config: QuoTeConfig,
) -> list[AttackState]:
    """Select final attack candidates from all pools.

    Mixes high_value and boundary pools according to ``config.hv_boundary_ratio``,
    then runs diverse top-k selection.

    Args:
        pools: List of AttackStatePools (one per seed).
        config: Hyperparameters.

    Returns:
        Selected list of AttackState instances ready for distillation.
    """
    all_hv: list[AttackState] = []
    all_bd: list[AttackState] = []
    for pool in pools:
        all_hv.extend(pool.high_value)
        all_bd.extend(pool.boundary)

    # Mix by ratio
    n_total = config.select_top_k
    n_hv = max(1, int(n_total * config.hv_boundary_ratio))
    n_bd = n_total - n_hv

    selected_hv = select_diverse(all_hv, n_hv, config.diversity_weight) if all_hv else []
    selected_bd = select_diverse(all_bd, n_bd, config.diversity_weight) if all_bd else []

    combined = selected_hv + selected_bd

    # If we don't have enough, fill from whichever pool has surplus
    if len(combined) < n_total:
        used_ids = {id(s) for s in combined}
        for s in sorted(all_hv + all_bd, key=lambda x: _effective_risk(x), reverse=True):
            if id(s) not in used_ids:
                combined.append(s)
                used_ids.add(id(s))
            if len(combined) >= n_total:
                break

    logger.info(
        "Selected %d candidates: %d high-value, %d boundary.",
        len(combined),
        sum(1 for s in combined if s.branch_type == "high_value"),
        sum(1 for s in combined if s.branch_type == "boundary"),
    )
    return combined
