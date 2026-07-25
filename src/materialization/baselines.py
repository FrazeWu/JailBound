"""
Baselines and Ablation Variants

Provides baseline attack strategies and ablation modes that share the same
evaluation pipeline as the full QuoTe v2 method.

Baselines:
- random_rewrite: random word-shuffle mutation, no soft-prefix optimisation
- risk_only: soft-prefix optimisation without FOL guidance or seed rewriting
- one_end_high_value: only high-value branch
- one_end_boundary: only boundary branch

Ablation modes are controlled via ``config.ablation_mode`` and are handled
inside ``biend_search.py`` and ``optimization.py``.  This module provides
convenience wrappers.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from embedding.attack_state import AttackState, AttackStatePool
from config.quote_config import QuoTeConfig
from materialization.model_loader import LoadedModel
from optimizer.seed_mutation import RandomSeedMutator, SeedMutator
from embedding.semantic_constraint import SemanticEncoder
from embedding.soft_prefix import init_soft_prefix

logger = logging.getLogger(__name__)


def random_rewrite_baseline(
    seeds: list[dict[str, str]],
    loaded: LoadedModel,
    semantic_encoder: SemanticEncoder,
    config: QuoTeConfig,
    rng: torch.Generator | None = None,
) -> list[AttackState]:
    """Generate attack candidates via random rewriting only (no optimisation).

    For each seed, generates ``mutation_candidates`` random rewrites and
    returns them as AttackStates with ``risk_score = -1`` (to be filled by
    the evaluator).
    """
    mutator = RandomSeedMutator()
    results: list[AttackState] = []

    for seed_info in seeds:
        candidates = mutator.mutate(
            original_seed=seed_info["original_seed"],
            current_seed=seed_info["original_seed"],
            n_candidates=config.mutation_candidates,
        )
        for text, strategy in candidates:
            state = AttackState(
                meta_prompt=seed_info.get("meta_prompt", ""),
                original_seed=seed_info["original_seed"],
                mutated_seed=text,
                branch_type="baseline_random",
                rewrite_source=strategy,
                condition_id=seed_info.get("condition_id", ""),
                behavior_id=seed_info.get("behavior_id", ""),
            )
            results.append(state)

    logger.info("Random rewrite baseline: %d candidates from %d seeds.", len(results), len(seeds))
    return results


def risk_only_baseline(
    seeds: list[dict[str, str]],
    loaded: LoadedModel,
    refusal_token_ids: torch.Tensor,
    semantic_encoder: SemanticEncoder,
    config: QuoTeConfig,
    rng: torch.Generator | None = None,
) -> list[AttackState]:
    """Optimise soft prefix for risk only — no FOL, no seed rewriting.

    Equivalent to ablation_mode="risk_only".
    """
    from dataclasses import fields as dc_fields
    from optimization import optimise_state

    base_kwargs = {f.name: getattr(config, f.name) for f in dc_fields(config)}
    base_kwargs.update({
        "ablation_mode": "risk_only",
        "lambda_fol": 0.0,
        "beta_sem": 0.0,
        "rewrite_interval": config.max_opt_steps + 1,
    })
    ablation_config = QuoTeConfig(**base_kwargs)

    results: list[AttackState] = []
    for seed_info in seeds:
        state = AttackState(
            meta_prompt=seed_info.get("meta_prompt", ""),
            original_seed=seed_info["original_seed"],
            mutated_seed=seed_info["original_seed"],
            soft_prefix=init_soft_prefix(config.prefix_length, loaded, rng),
            branch_type="baseline_risk_only",
            condition_id=seed_info.get("condition_id", ""),
            behavior_id=seed_info.get("behavior_id", ""),
        )
        state, _ = optimise_state(state, loaded, refusal_token_ids, semantic_encoder, ablation_config)
        results.append(state)

    logger.info("Risk-only baseline: %d candidates from %d seeds.", len(results), len(seeds))
    return results
