"""
Embedding Transfer Module

Transfers optimised attack embeddings u* = [z*; U*] from the surrogate model
M_a to a target model M_b.

Two strategies:
1. **Same-tokenizer direct injection** — for models sharing the same tokenizer
   (e.g., Qwen2.5 family). Directly inject the embedding tensors.
2. **Cross-architecture nearest-token projection** — project each embedding
   vector to the nearest vocabulary token in M_b's embedding space, then
   re-embed with M_b's embedding layer.

    T_{a→b}(u*): nearest-token projection via cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedding.attack_state import AttackState
from materialization.model_loader import LoadedModel

logger = logging.getLogger(__name__)


def _project_to_nearest_tokens(
    embeds: torch.Tensor,
    target_embed_weight: torch.Tensor,
) -> torch.Tensor:
    """Project each embedding vector to the nearest vocabulary token.

    Args:
        embeds: (1, T, d_source) embedding tensor.
        target_embed_weight: (V, d_target) target model's embedding matrix.

    Returns:
        (1, T) token IDs in the target vocabulary.
    """
    # If dimensions differ, we can only do nearest-token in shared space
    # For same-dim models, cosine similarity works directly
    e = embeds.squeeze(0).float()  # (T, d)
    w = target_embed_weight.float()  # (V, d)

    # Truncate/pad if dimensions differ
    d_e = e.shape[1]
    d_w = w.shape[1]
    if d_e != d_w:
        min_d = min(d_e, d_w)
        e = e[:, :min_d]
        w = w[:, :min_d]

    # Cosine similarity
    e_norm = F.normalize(e, dim=-1)  # (T, d)
    w_norm = F.normalize(w, dim=-1)  # (V, d)
    sim = e_norm @ w_norm.T  # (T, V)
    token_ids = sim.argmax(dim=-1)  # (T,)
    return token_ids.unsqueeze(0)  # (1, T)


def transfer_state_same_tokenizer(
    state: AttackState,
    target_loaded: LoadedModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer for same-tokenizer models (direct embedding injection).

    Rebuilds H(u) = [z*; E_b(x)_{Ω̄_s}; U*] using the target model's
    embedding layer for the frozen scaffold, but keeping z* and U* as-is.

    If embedding dimensions differ (e.g. 7B→0.5B), falls back to
    nearest-token projection for z* and U*.

    Returns:
        full_embeds: (1, T, d_target) on target device.
        attention_mask: (1, T) on target device.
    """
    assert state.soft_prefix is not None
    assert state.editable_seed_block is not None
    assert state.frozen_scaffold_ids is not None

    # Use embed_layer's device (reliable even with device_map="auto")
    embed_device = target_loaded.embed_layer.weight.device
    z = state.soft_prefix.to(embed_device, target_loaded.model.dtype)
    U = state.editable_seed_block.to(embed_device, target_loaded.model.dtype)
    scaffold_ids = state.frozen_scaffold_ids.to(embed_device)

    target_dim = target_loaded.embed_layer.weight.shape[1]
    source_dim = z.shape[-1]

    with torch.no_grad():
        scaffold_embeds = target_loaded.embed_layer(scaffold_ids).to(z.dtype)

    if source_dim != target_dim:
        # Dimension mismatch: project z* and U* to nearest tokens in target vocab
        logger.debug(
            "Dimension mismatch (source=%d, target=%d): using nearest-token projection",
            source_dim, target_dim,
        )
        target_weight = target_loaded.embed_layer.weight.data
        z_ids = _project_to_nearest_tokens(z, target_weight)
        u_ids = _project_to_nearest_tokens(U, target_weight)
        with torch.no_grad():
            z = target_loaded.embed_layer(z_ids).to(scaffold_embeds.dtype)
            U = target_loaded.embed_layer(u_ids).to(scaffold_embeds.dtype)

    full_embeds = torch.cat([z, scaffold_embeds, U], dim=1)
    attn_mask = torch.ones(1, full_embeds.shape[1], dtype=torch.long, device=embed_device)
    return full_embeds, attn_mask


def transfer_state_cross_arch(
    state: AttackState,
    source_loaded: LoadedModel,
    target_loaded: LoadedModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer for cross-architecture models via nearest-token projection.

    Projects z* and U* to nearest tokens in the target vocabulary, then
    re-embeds them using the target's embedding layer.

    Returns:
        full_embeds: (1, T, d_target) on target device.
        attention_mask: (1, T) on target device.
    """
    assert state.soft_prefix is not None
    assert state.editable_seed_block is not None

    target_embed_weight = target_loaded.embed_layer.weight.data

    # Project z* → nearest tokens → re-embed
    z_token_ids = _project_to_nearest_tokens(
        state.soft_prefix.to(source_loaded.device), target_embed_weight.to(source_loaded.device),
    ).to(target_loaded.device)

    # Project U* → nearest tokens → re-embed
    u_token_ids = _project_to_nearest_tokens(
        state.editable_seed_block.to(source_loaded.device), target_embed_weight.to(source_loaded.device),
    ).to(target_loaded.device)

    with torch.no_grad():
        z_embeds = target_loaded.embed_layer(z_token_ids)
        u_embeds = target_loaded.embed_layer(u_token_ids)

    # Re-tokenize the meta prompt scaffold for the target model
    from soft_prefix import build_frozen_scaffold_ids
    scaffold_ids = build_frozen_scaffold_ids(state.meta_prompt, target_loaded)

    with torch.no_grad():
        scaffold_embeds = target_loaded.embed_layer(scaffold_ids)

    full_embeds = torch.cat([z_embeds, scaffold_embeds, u_embeds], dim=1)
    attn_mask = torch.ones(1, full_embeds.shape[1], dtype=torch.long, device=target_loaded.device)
    return full_embeds, attn_mask


def generate_from_embeddings(
    full_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate text from embedding input (no discrete prompt needed).

    Args:
        full_embeds: (1, T, d) embedding tensor.
        attention_mask: (1, T) mask.
        model: Target model.
        tokenizer: Target tokenizer.
        max_new_tokens: Max generation length.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Generated response text.
    """
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    # When inputs_embeds is used, generate() returns ONLY the new token IDs
    # (no input tokens in output). Do NOT slice by input length.
    gen_ids = outputs[0]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
