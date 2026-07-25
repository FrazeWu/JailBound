"""
QuoTe Embedding Perturbation Module

Handles three core operations required by the QuoTe pipeline:

1. **Extraction**: tokenize an input, extract embeddings E ∈ R^{T×d}.
2. **Perturbation mask**: identify which token positions are eligible for
   perturbation (user content only — structural tokens are excluded).
3. **L2 projection**: project δ back into the ε-ball after each gradient step.

Design constraints:
- Perturbation mask excludes <|im_start|>, <|im_end|>, role tokens, and padding.
- All operations are continuous in embedding space; no discrete token recovery.
- Input embeddings returned with requires_grad=True for gradient computation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

from materialization.model_loader import LoadedModel

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class EmbedBatch:
    """
    Container for a tokenized + embedded input batch ready for perturbation.

    Attributes:
        input_ids: (B, T) original token IDs (for reference only; not modified).
        original_embeds: (B, T, d) clean embeddings E — detached, no grad.
        perturbed_embeds: (B, T, d) current E+δ — requires_grad=True.
        attention_mask: (B, T) attention mask (1 for real tokens, 0 for padding).
        perturbation_mask: (B, T) float mask: 1.0 where δ is allowed, 0.0 elsewhere.
        response_positions: (B,) int64 — position whose logits predict first response token.
    """

    input_ids: torch.Tensor
    original_embeds: torch.Tensor
    perturbed_embeds: torch.Tensor
    attention_mask: torch.Tensor
    perturbation_mask: torch.Tensor
    response_positions: torch.Tensor


# =============================================================================
# Tokenization helpers
# =============================================================================


def _apply_chat_template(
    behavior: str,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format a behavior string into the Qwen3 chat template.

    Args:
        behavior: The harmful behavior / instruction string.
        tokenizer: Qwen3 tokenizer (must support apply_chat_template).
        system_prompt: Optional system prompt; defaults to a neutral placeholder.

    Returns:
        Formatted string ready for tokenization (includes special tokens).
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": behavior},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        # add_generation_prompt=True appends <|im_start|>assistant\n
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback if apply_chat_template is unavailable
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{behavior}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        formatted = "".join(parts)

    return formatted


# =============================================================================
# Perturbation mask
# =============================================================================


def _build_perturbation_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: frozenset[int],
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """
    Build a binary mask marking positions eligible for embedding perturbation.

    Excluded positions (mask = 0):
    - Padding tokens (attention_mask == 0)
    - All IDs in special_token_ids (<|im_start|>, <|im_end|>, BOS, EOS, PAD …)
    - The single-token role strings immediately after <|im_start|>

    Included positions (mask = 1):
    - User content tokens (the actual behavior text)

    Args:
        input_ids: (B, T) int64 token IDs.
        attention_mask: (B, T) bool/int mask.
        special_token_ids: frozenset of IDs that must not be perturbed.
        tokenizer: Used only to determine role token IDs (fallback).

    Returns:
        (B, T) float32 mask where 1.0 = perturb, 0.0 = skip.
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=input_ids.device)

    # Zero out padding
    mask = mask * attention_mask.float()

    # Zero out special/structural token positions
    for sid in special_token_ids:
        mask = mask * (input_ids != sid).float()

    return mask


# =============================================================================
# Embedding extraction
# =============================================================================


def embed_behaviors(
    behaviors: list[str],
    loaded: LoadedModel,
    system_prompt: Optional[str] = None,
    max_length: int = 512,
) -> EmbedBatch:
    """
    Tokenize a list of behavior strings, extract embeddings, and build the
    perturbation mask.  Returns an EmbedBatch with perturbed_embeds initialized
    to the clean embeddings (δ = 0) and requires_grad=True.

    Args:
        behaviors: List of harmful behavior strings (one per batch item).
        loaded: LoadedModel with frozen model, tokenizer, and special_token_ids.
        system_prompt: Optional system prompt string.
        max_length: Maximum tokenized sequence length (padding/truncation).

    Returns:
        EmbedBatch ready for the QuoTe pipeline.
    """
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    # Build formatted strings with chat template
    formatted = [_apply_chat_template(b, tokenizer, system_prompt) for b in behaviors]

    # Tokenize with padding
    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,  # Already embedded by apply_chat_template
    )

    input_ids: torch.Tensor = encoding["input_ids"].to(device)
    attention_mask: torch.Tensor = encoding["attention_mask"].to(device)

    # Extract clean embeddings — detached (no grad)
    embed_layer = loaded.embed_layer
    with torch.no_grad():
        original_embeds: torch.Tensor = embed_layer(input_ids).detach()

    # Initialize perturbed_embeds as a leaf tensor (δ = 0 initially)
    perturbed_embeds = original_embeds.clone().requires_grad_(True)

    # Build perturbation mask
    perturbation_mask = _build_perturbation_mask(
        input_ids, attention_mask, loaded.special_token_ids, tokenizer
    )

    # Find response start positions
    response_positions = _find_response_positions_batch(input_ids, tokenizer)

    logger.debug(
        "Embedded %d behavior(s); seq_len=%d, perturb_mask mean=%.3f",
        len(behaviors),
        input_ids.shape[1],
        perturbation_mask.float().mean().item(),
    )

    return EmbedBatch(
        input_ids=input_ids,
        original_embeds=original_embeds,
        perturbed_embeds=perturbed_embeds,
        attention_mask=attention_mask,
        perturbation_mask=perturbation_mask,
        response_positions=response_positions,
    )


def _find_response_positions_batch(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """
    Find the token position whose logits predict the first assistant response
    token, for each item in the batch.

    Args:
        input_ids: (B, T) token ID tensor.
        tokenizer: Model tokenizer.

    Returns:
        (B,) int64 tensor of position indices.
    """
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    first_assistant_tok = assistant_ids[0] if assistant_ids else -1

    batch_size, seq_len = input_ids.shape
    positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    for b in range(batch_size):
        ids_list = input_ids[b].tolist()
        found = seq_len - 1  # default: last token
        if first_assistant_tok >= 0:
            for pos in range(seq_len - 1, -1, -1):
                if ids_list[pos] == first_assistant_tok:
                    found = pos
                    break
        positions[b] = found

    return positions


# =============================================================================
# L2 projection
# =============================================================================


def project_l2(
    original_embeds: torch.Tensor,
    perturbed_embeds: torch.Tensor,
    perturbation_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Project the perturbation δ = perturbed_embeds - original_embeds back into
    the L2 ε-ball, applying the perturbation mask to ensure only eligible
    positions are modified.

    The projection is:
        δ_masked = δ * perturbation_mask.unsqueeze(-1)
        if ‖δ_masked‖₂ > ε:
            δ_masked = δ_masked * (ε / ‖δ_masked‖₂)
        E' = E + δ_masked

    Args:
        original_embeds: (B, T, d) clean embeddings (no grad required).
        perturbed_embeds: (B, T, d) current embeddings (may or may not have grad).
        perturbation_mask: (B, T) float mask (0/1).
        epsilon: L2 ball radius.

    Returns:
        Projected (B, T, d) embedding tensor with requires_grad=True.
        The returned tensor is a *new leaf tensor* with grad enabled.
    """
    with torch.no_grad():
        # Compute raw delta
        delta = perturbed_embeds.detach() - original_embeds

        # Apply mask: cast to embedding dtype first to avoid float32 upcast,
        # then zero out non-eligible positions
        mask_expanded = perturbation_mask.to(delta.dtype).unsqueeze(-1)  # (B, T, 1)
        delta = delta * mask_expanded

        # Compute per-batch L2 norm of delta
        # Flatten (T, d) for norm computation, then reshape
        delta_flat = delta.view(delta.shape[0], -1)  # (B, T*d)
        norms = delta_flat.norm(p=2, dim=-1, keepdim=True)  # (B, 1)

        # Project: scale down if norm exceeds ε
        scale = torch.clamp(epsilon / (norms + 1e-8), max=1.0)  # (B, 1)
        delta_flat = delta_flat * scale
        delta = delta_flat.view_as(delta)

        # Reconstruct projected embeddings — cast back to original dtype.
        # delta may be float32 (mask upcast); match original_embeds.dtype.
        projected = (original_embeds + delta).to(original_embeds.dtype)

    # Return as a new leaf tensor with gradient enabled
    return projected.requires_grad_(True)


def apply_perturbation_mask_to_grad(
    grad: torch.Tensor,
    perturbation_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Zero out gradient components at masked-out positions.

    Used in custom gradient hooks to prevent gradient flow through
    structural token positions during the perturbation search.

    Args:
        grad: (B, T, d) gradient tensor.
        perturbation_mask: (B, T) float mask.

    Returns:
        (B, T, d) gradient with non-eligible positions zeroed.
    """
    return grad * perturbation_mask.to(grad.dtype).unsqueeze(-1)


def detach_and_clone_as_leaf(embeds: torch.Tensor) -> torch.Tensor:
    """
    Detach, clone, and re-enable gradients on a tensor.

    Used after each perturbation step to create a fresh leaf tensor
    suitable for the next iteration's gradient computation.

    Args:
        embeds: Any embedding tensor.

    Returns:
        New leaf tensor with same data and requires_grad=True.
    """
    return embeds.detach().clone().requires_grad_(True)
