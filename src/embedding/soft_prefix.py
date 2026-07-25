"""
Soft Prefix & Editable Seed Block Module

Handles initialisation and embedding-space assembly for the QuoTe v2
attack optimisation pipeline.

The continuous attack state is  u = [z; U]  where:
  z  = soft prefix (1, P, d)          — randomly initialised from vocab embeddings
  U  = editable seed block (1, S, d)  — initialised from seed token embeddings

The full model input is:
    H(u) = [z; E(x)_{Ω̄_s}; U]
         = [soft_prefix; frozen_scaffold_embeds; editable_seed_block]

  z           — learnable prefix (grad ✓)
  E(x)_{Ω̄_s} — frozen scaffold: chat template + meta_prompt tokens (grad ✗)
  U           — learnable seed embeddings (grad ✓)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from materialization.model_loader import LoadedModel

logger = logging.getLogger(__name__)


def init_soft_prefix(
    prefix_length: int,
    loaded: LoadedModel,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialise a soft prefix z from random vocab embeddings.

    Returns:
        (1, prefix_length, embed_dim) float tensor with requires_grad=True.
    """
    vocab_size = loaded.vocab_size
    if rng is not None:
        indices = torch.randint(
            0, vocab_size, (prefix_length,),
            generator=rng, device=rng.device,
        )
    else:
        indices = torch.randint(0, vocab_size, (prefix_length,))
    indices = indices.to(loaded.device)
    with torch.no_grad():
        prefix = loaded.embed_layer(indices).clone()  # (P, d)
    prefix = prefix.unsqueeze(0).to(loaded.model.dtype)  # (1, P, d)
    return prefix.detach().requires_grad_(True)


def init_editable_seed_block(
    seed_text: str,
    loaded: LoadedModel,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise the editable seed embedding block U from seed token embeddings.

    Args:
        seed_text: The seed attack prompt text.
        loaded: Frozen surrogate model.
        max_length: Max tokens for the seed portion.

    Returns:
        U:  (1, S, d) editable seed embeddings with requires_grad=True.
        U0: (1, S, d) frozen initial copy (for drift penalty).
    """
    tokens = loaded.tokenizer(
        seed_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )
    seed_ids = tokens["input_ids"].to(loaded.device)  # (1, S)
    with torch.no_grad():
        U = loaded.embed_layer(seed_ids).clone().to(loaded.model.dtype)  # (1, S, d)
    U0 = U.detach().clone()  # frozen reference
    return U.detach().requires_grad_(True), U0


def build_frozen_scaffold_ids(
    meta_prompt: str,
    loaded: LoadedModel,
    max_length: int = 256,
) -> torch.Tensor:
    """Tokenize the frozen scaffold: system meta-prompt + chat template markers.

    The scaffold includes everything EXCEPT the user seed content:
      <|im_start|>system\\n{meta_prompt}<|im_end|>\\n<|im_start|>user\\n

    Plus the trailing assistant generation prompt:
      <|im_end|>\\n<|im_start|>assistant\\n

    Returns:
        (1, T_scaffold) token IDs on device.
    """
    tokenizer = loaded.tokenizer

    # Build template with a placeholder seed so we can split
    placeholder = "<<SEED_PLACEHOLDER>>"
    messages = [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": placeholder},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        full_text = (
            f"<|im_start|>system\n{meta_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{placeholder}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # Split around placeholder to get prefix and suffix scaffold
    parts = full_text.split(placeholder)
    scaffold_prefix = parts[0]  # system + user preamble
    scaffold_suffix = parts[1] if len(parts) > 1 else ""  # user end + assistant start

    # Tokenize both parts
    prefix_ids = tokenizer(
        scaffold_prefix, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_length,
    )["input_ids"].to(loaded.device)

    suffix_ids = tokenizer(
        scaffold_suffix, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_length // 4,
    )["input_ids"].to(loaded.device)

    # Concatenate: [scaffold_prefix | scaffold_suffix]
    scaffold_ids = torch.cat([prefix_ids, suffix_ids], dim=1)  # (1, T_scaffold)
    return scaffold_ids


def build_full_input(
    soft_prefix: torch.Tensor,
    frozen_scaffold_ids: torch.Tensor,
    editable_seed_block: torch.Tensor,
    loaded: LoadedModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the complete embedding input H(u) = [z; E(x)_{Ω̄_s}; U].

    Layout:
        [soft_prefix (grad ✓) | frozen_scaffold_embeds (grad ✗) | editable_seed_block (grad ✓)]

    Args:
        soft_prefix: (1, P, d) learnable prefix tensor z.
        frozen_scaffold_ids: (1, T_scaffold) token IDs for frozen scaffold.
        editable_seed_block: (1, S, d) learnable seed embeddings U.
        loaded: Frozen surrogate model.

    Returns:
        full_embeds: (1, P + T_scaffold + S, d) concatenated embeddings.
        attention_mask: (1, P + T_scaffold + S) all-ones mask.
    """
    with torch.no_grad():
        scaffold_embeds = loaded.embed_layer(frozen_scaffold_ids).to(soft_prefix.dtype)

    full_embeds = torch.cat([soft_prefix, scaffold_embeds, editable_seed_block], dim=1)
    total_len = full_embeds.shape[1]
    attention_mask = torch.ones(1, total_len, dtype=torch.long, device=loaded.device)

    return full_embeds, attention_mask
