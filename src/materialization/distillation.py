"""
Attack Distillation Module

Converts the continuous soft-prefix attack state into a discrete text prompt
that can be sent to any target model (including black-box APIs).

Strategy: **Nearest-token projection** — for each position in the soft prefix,
find the vocabulary token whose embedding is closest (L2) and decode the
resulting token sequence.  The final attack prompt is:

    rendered = meta_prompt_text + " " + mutated_seed + " " + decoded_prefix

Each distilled prompt retains a reference to the source AttackState for
traceability.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from embedding.attack_state import AttackState
from materialization.model_loader import LoadedModel

logger = logging.getLogger(__name__)


def _nearest_token_ids(
    soft_prefix: torch.Tensor,
    embed_layer: torch.nn.Embedding,
) -> torch.Tensor:
    """Project each prefix position to the nearest vocabulary embedding.

    Args:
        soft_prefix: (1, P, d) continuous prefix tensor.
        embed_layer: Frozen embedding table.

    Returns:
        (P,) tensor of token IDs.
    """
    weight = embed_layer.weight.detach()  # (V, d)
    prefix = soft_prefix.squeeze(0).detach().float()  # (P, d)
    weight_f = weight.float()

    # Cosine similarity is more stable than L2 for high-dim spaces
    prefix_norm = F.normalize(prefix, dim=-1)
    weight_norm = F.normalize(weight_f, dim=-1)
    sims = prefix_norm @ weight_norm.T  # (P, V)
    token_ids = sims.argmax(dim=-1)  # (P,)
    return token_ids


def distill_to_text(
    state: AttackState,
    loaded: LoadedModel,
) -> str:
    """Convert an AttackState with soft_prefix into a pure-text attack prompt.

    The rendered prompt is stored in ``state.rendered_prompt`` and also returned.
    """
    if state.soft_prefix is None:
        # No prefix to decode — just concatenate meta + seed
        rendered = f"{state.meta_prompt}\n{state.mutated_seed}".strip()
        state.rendered_prompt = rendered
        return rendered

    token_ids = _nearest_token_ids(state.soft_prefix, loaded.embed_layer)
    decoded_suffix = loaded.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    # Assemble: seed text + decoded prefix as a suffix trigger
    if decoded_suffix:
        rendered = f"{state.mutated_seed} {decoded_suffix}"
    else:
        rendered = state.mutated_seed

    state.rendered_prompt = rendered
    return rendered


def distill_batch(
    states: list[AttackState],
    loaded: LoadedModel,
) -> list[str]:
    """Distill a batch of states into discrete text prompts."""
    results: list[str] = []
    for s in states:
        results.append(distill_to_text(s, loaded))
    return results
