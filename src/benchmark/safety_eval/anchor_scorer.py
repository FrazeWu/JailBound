"""Exact teacher-forced scoring for full answer and refusal continuations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContinuationScores:
    proxy_risk: torch.Tensor
    answer_logp: torch.Tensor
    refusal_logp: torch.Tensor


def tokenize_anchor_set(
    tokenizer: Any, anchors: Sequence[str]
) -> tuple[torch.Tensor, ...]:
    encoded = tuple(
        torch.tensor(
            tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
        )
        for text in anchors
    )
    if not encoded or any(ids.numel() == 0 for ids in encoded):
        raise ValueError("anchor sets require non-empty token sequences")
    return encoded


def _validate_anchor_set(
    anchors: Sequence[torch.Tensor], label: str
) -> tuple[torch.Tensor, ...]:
    values = tuple(anchors)
    if not values or any(
        ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (torch.int32, torch.int64)
        for ids in values
    ):
        raise ValueError(f"{label} anchors require non-empty rank-1 token sequences")
    return values


def _prompt_lengths(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype not in (torch.int32, torch.int64, torch.bool):
        raise ValueError("prompt_attention_mask must be an integral rank-2 tensor")
    lengths = mask.long().sum(dim=1)
    if (lengths < 1).any() or not torch.equal(
        mask.bool(),
        torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1),
    ):
        raise ValueError("prompt_attention_mask must be non-empty right-padded prefixes")
    return lengths


def score_continuation_sets(
    *,
    model: Any,
    embedding_layer: Any,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    answer_anchors: Sequence[torch.Tensor],
    refusal_anchors: Sequence[torch.Tensor],
) -> ContinuationScores:
    """Score every continuation token in one padded teacher-forced forward pass."""
    if prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must have shape [batch, tokens, hidden]")
    if prompt_attention_mask.shape != prompt_embeds.shape[:2]:
        raise ValueError("prompt_attention_mask must match prompt_embeds batch and tokens")
    prompt_lengths = _prompt_lengths(prompt_attention_mask)
    answer = _validate_anchor_set(answer_anchors, "answer")
    refusal = _validate_anchor_set(refusal_anchors, "refusal")
    anchors = answer + refusal
    batch_size, _, hidden_size = prompt_embeds.shape
    anchor_count = len(anchors)
    max_anchor_length = max(ids.numel() for ids in anchors)
    sequence_length = int(prompt_lengths.max().item()) + max_anchor_length - 1
    expanded = prompt_embeds.new_zeros((batch_size * anchor_count, sequence_length, hidden_size))
    attention_mask = torch.zeros(
        (batch_size * anchor_count, sequence_length),
        dtype=torch.long,
        device=prompt_embeds.device,
    )
    target_ids = torch.zeros(
        (batch_size * anchor_count, max_anchor_length),
        dtype=torch.long,
        device=prompt_embeds.device,
    )
    target_positions = torch.zeros_like(target_ids)
    target_mask = torch.zeros_like(target_ids, dtype=torch.bool)

    for batch_index in range(batch_size):
        prompt_length = int(prompt_lengths[batch_index].item())
        for anchor_index, anchor in enumerate(anchors):
            row = batch_index * anchor_count + anchor_index
            ids = anchor.to(device=prompt_embeds.device, dtype=torch.long)
            continuation = embedding_layer(ids[:-1]).to(dtype=prompt_embeds.dtype)
            full_length = prompt_length + continuation.shape[0]
            expanded[row, :prompt_length] = prompt_embeds[batch_index, :prompt_length]
            expanded[row, prompt_length:full_length] = continuation
            attention_mask[row, :full_length] = 1
            target_length = ids.numel()
            target_ids[row, :target_length] = ids
            target_positions[row, :target_length] = torch.arange(
                prompt_length - 1,
                prompt_length - 1 + target_length,
                device=prompt_embeds.device,
            )
            target_mask[row, :target_length] = True

    outputs = model(
        inputs_embeds=expanded,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
    if logits.shape[:2] != expanded.shape[:2]:
        raise ValueError("model logits must align with padded continuation inputs")
    log_probs = F.log_softmax(logits, dim=-1)
    row_indices = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    selected = log_probs[row_indices, target_positions].gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    per_anchor = (selected * target_mask).sum(dim=1).reshape(batch_size, anchor_count)
    answer_logp = per_anchor[:, : len(answer)].mean(dim=1)
    refusal_logp = per_anchor[:, len(answer) :].mean(dim=1)
    return ContinuationScores(
        proxy_risk=answer_logp - refusal_logp,
        answer_logp=answer_logp,
        refusal_logp=refusal_logp,
    )
