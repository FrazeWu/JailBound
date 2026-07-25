"""
FOL-Guided Bi-End Search

Implements the QuoTe v2 search algorithm with two parallel branches:

* **High-value branch** — seeks high-risk, low-FOL states (settled maxima).
    O⁻ = r̃(u) − λ·FOL(u) − γ_z·‖z‖² − γ_u·‖U−U₀‖²

* **Boundary branch** — seeks high-FOL states (near decision boundary).
    O⁺ = r̃(u) + λ·FOL(u) − γ_z·‖z‖² − γ_u·‖U−U₀‖²

Each branch runs a single continuous optimisation of u = [z; U] (no text mutation).
The two pools are maintained as bounded priority queues.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from embedding.attack_state import AttackState, AttackStatePool
from config.quote_config import QuoTeConfig
from materialization.model_loader import LoadedModel
from optimizer.optimization import optimise_state
from objectives.safety_risk import build_anchor_token_ids
from embedding.soft_prefix import (
    build_frozen_scaffold_ids,
    init_editable_seed_block,
    init_soft_prefix,
)

logger = logging.getLogger(__name__)


def _run_branch(
    state: AttackState,
    branch_type: str,
    loaded: LoadedModel,
    answer_token_ids: torch.Tensor,
    refusal_token_ids: torch.Tensor,
    config: QuoTeConfig,
    pool: AttackStatePool,
    rng: torch.Generator | None = None,
    wildguard_judge: object | None = None,
) -> dict[str, Any]:
    """Run one search branch (high_value or boundary) for a single seed.

    Returns a trace dict with optimisation diagnostics.
    """
    state = state.clone()
    state.branch_type = branch_type

    # Single continuous optimisation run (no mutation cycles)
    state, opt_trace = optimise_state(
        state, loaded, answer_token_ids, refusal_token_ids, config,
        wildguard_judge=wildguard_judge,
    )

    # Record to pool
    pool.add(state)

    trace = {
        "branch": branch_type,
        "proxy_risk": state.proxy_risk,
        "fol": state.fol,
        "prefix_norm": state.prefix_norm,
        "seed_block_drift": state.seed_block_drift,
        "opt_steps": len(opt_trace),
        "branch_score": state.branch_score(config.lambda_fol, config.gamma_z, config.gamma_u),
    }
    return trace


def biend_search_single(
    meta_prompt: str,
    original_seed: str,
    condition_id: str,
    behavior_id: str,
    loaded: LoadedModel,
    config: QuoTeConfig,
    rng: torch.Generator | None = None,
    wildguard_judge: object | None = None,
) -> tuple[AttackStatePool, dict[str, Any]]:
    """Run bi-end search for one (condition, seed) pair.

    Returns:
        pool: AttackStatePool with high_value and boundary candidates.
        diagnostics: Dict with per-branch traces and timing.
    """
    # Build anchor token IDs
    answer_ids, refusal_ids = build_anchor_token_ids(loaded.tokenizer, loaded.device)

    pool = AttackStatePool(
        max_size=config.beam_size,
        lambda_fol=config.lambda_fol,
        gamma_z=config.gamma_z,
        gamma_u=config.gamma_u,
    )

    # Build frozen scaffold (system + chat template, without seed content)
    scaffold_ids = build_frozen_scaffold_ids(meta_prompt, loaded)

    # Initialise editable seed block from seed text
    U, U0 = init_editable_seed_block(original_seed, loaded)

    # Initial state template
    base_state = AttackState(
        meta_prompt=meta_prompt,
        original_seed=original_seed,
        mutated_seed=original_seed,
        soft_prefix=init_soft_prefix(config.prefix_length, loaded, rng),
        editable_seed_block=U,
        initial_seed_block=U0,
        frozen_scaffold_ids=scaffold_ids,
        condition_id=condition_id,
        behavior_id=behavior_id,
    )

    diagnostics: dict[str, Any] = {
        "behavior_id": behavior_id,
        "condition_id": condition_id,
    }

    # ---- High-value branch ----
    if config.ablation_mode not in ("boundary_only",):
        t0 = time.monotonic()
        hv_trace = _run_branch(
            base_state, "high_value",
            loaded, answer_ids, refusal_ids, config, pool, rng,
            wildguard_judge=wildguard_judge,
        )
        diagnostics["high_value"] = {
            **hv_trace,
            "elapsed_s": time.monotonic() - t0,
        }

    # ---- Boundary branch ----
    if config.ablation_mode not in ("high_value_only",):
        # Re-init z and U for independent boundary search
        boundary_state = base_state.clone()
        boundary_state.soft_prefix = init_soft_prefix(config.prefix_length, loaded, rng)
        boundary_state.editable_seed_block, _ = init_editable_seed_block(original_seed, loaded)
        t0 = time.monotonic()
        bd_trace = _run_branch(
            boundary_state, "boundary",
            loaded, answer_ids, refusal_ids, config, pool, rng,
            wildguard_judge=wildguard_judge,
        )
        diagnostics["boundary"] = {
            **bd_trace,
            "elapsed_s": time.monotonic() - t0,
        }

    return pool, diagnostics


def run_biend_search(
    seeds: list[dict[str, str]],
    loaded: LoadedModel,
    config: QuoTeConfig,
    rng: torch.Generator | None = None,
    wildguard_judge: object | None = None,
) -> tuple[list[AttackStatePool], list[dict[str, Any]]]:
    """Run bi-end search for a batch of seeds.

    Args:
        seeds: List of dicts with keys:
            meta_prompt, original_seed, condition_id, behavior_id.
        loaded: Frozen surrogate model.
        config: Hyperparameters.
        rng: Optional torch Generator.
        wildguard_judge: Optional WildGuard proxy judge.

    Returns:
        (pools, all_diagnostics)
    """
    pools: list[AttackStatePool] = []
    all_diag: list[dict[str, Any]] = []

    for i, seed_info in enumerate(seeds):
        logger.info(
            "Bi-end search %d/%d: behavior=%s",
            i + 1, len(seeds), seed_info.get("behavior_id", "?"),
        )
        pool, diag = biend_search_single(
            meta_prompt=seed_info["meta_prompt"],
            original_seed=seed_info["original_seed"],
            condition_id=seed_info.get("condition_id", ""),
            behavior_id=seed_info.get("behavior_id", f"seed_{i}"),
            loaded=loaded,
            config=config,
            rng=rng,
            wildguard_judge=wildguard_judge,
        )
        pools.append(pool)
        all_diag.append(diag)

    return pools, all_diag
