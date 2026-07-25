"""
Proxy Risk Module

Provides a differentiable surrogate risk function for soft-prefix optimisation.

    proxy_risk(p) = h(E_full)

where E_full = [embed(meta_prompt) ; embed(mutated_seed) ; soft_prefix] and
h(E) = −log P(refusal | E) is the existing safety risk function from
``safety_risk.py``.

The proxy risk is **not** the real judge score — it is used only for gradient-
based optimisation.  The real risk (``risk_score``) comes from an external
LLM-as-judge after discrete distillation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from materialization.model_loader import LoadedModel
from objectives.safety_risk import compute_h_E

logger = logging.getLogger(__name__)


def build_full_embeds(
    loaded: LoadedModel,
    meta_prompt_ids: torch.Tensor,
    seed_ids: torch.Tensor,
    soft_prefix: torch.Tensor,
    seed_embeds_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate meta-prompt embeddings, seed embeddings, and soft prefix.

    The returned embedding tensor has ``requires_grad`` propagated from
    *soft_prefix* (and optionally from *seed_embeds_override*).

    Layout:  [meta_prompt_embeds | seed_embeds | soft_prefix]

    Args:
        loaded: Frozen LoadedModel.
        meta_prompt_ids: (1, T_m) token IDs for the meta-prompt portion.
        seed_ids: (1, T_s) token IDs for the (mutated) seed portion.
        soft_prefix: (1, P, d) learnable prefix tensor (requires_grad=True).
        seed_embeds_override: If given, use this (1, T_s, d) instead of
            looking up *seed_ids* in the embedding table.  Must have
            ``requires_grad=True`` if gradient w.r.t. seed embeddings is
            desired.

    Returns:
        full_embeds: (1, T_m + T_s + P, d) concatenated embedding.
        attention_mask: (1, T_m + T_s + P) all-ones mask.
    """
    embed_layer = loaded.embed_layer

    with torch.no_grad():
        meta_embeds = embed_layer(meta_prompt_ids)  # (1, T_m, d)

    if seed_embeds_override is not None:
        seed_embeds = seed_embeds_override
    else:
        # detach + clone so the seed embedding itself is a leaf for grad
        seed_embeds = embed_layer(seed_ids).detach().clone().requires_grad_(True)

    # soft_prefix already has requires_grad=True
    full_embeds = torch.cat([meta_embeds, seed_embeds, soft_prefix], dim=1)

    total_len = full_embeds.shape[1]
    attention_mask = torch.ones(
        1, total_len, dtype=torch.long, device=loaded.device,
    )
    return full_embeds, attention_mask


def compute_proxy_risk(
    loaded: LoadedModel,
    meta_prompt_ids: torch.Tensor,
    seed_ids: torch.Tensor,
    soft_prefix: torch.Tensor,
    refusal_token_ids: torch.Tensor,
    seed_embeds_override: torch.Tensor | None = None,
    response_position: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the differentiable proxy risk for a single attack state.

    Returns:
        proxy_risk: Scalar tensor with grad graph (higher → riskier).
        full_embeds: The concatenated embedding used (for FOL computation).
    """
    full_embeds, attn_mask = build_full_embeds(
        loaded, meta_prompt_ids, seed_ids, soft_prefix, seed_embeds_override,
    )

    h_E = compute_h_E(
        model=loaded.model,
        perturbed_embeds=full_embeds,
        attention_mask=attn_mask,
        refusal_token_ids=refusal_token_ids,
        response_position=response_position,
    )

    return h_E, full_embeds


def compute_fol_from_embeds(
    h_E: torch.Tensor,
    target_tensor: torch.Tensor,
    epsilon: float,
) -> float:
    """Compute first-order Lipschitz signal: FOL = ε · ‖∇_u R̃(H)‖₂.

    The gradient is computed w.r.t. *target_tensor* (the optimisable part,
    typically the soft prefix ``z``), not the full embedding sequence.

    Args:
        h_E: Scalar proxy risk (from ``compute_proxy_risk`` or ``compute_h_E``).
        target_tensor: The tensor to differentiate against (e.g. soft_prefix).
            Must be in the computation graph of *h_E*.
        epsilon: L2 perturbation radius.

    Returns:
        FOL scalar (float, detached).
    """
    grad = torch.autograd.grad(h_E, target_tensor, retain_graph=True)[0]  # same shape as target_tensor
    fol = epsilon * grad.norm(p=2).item()
    return fol
