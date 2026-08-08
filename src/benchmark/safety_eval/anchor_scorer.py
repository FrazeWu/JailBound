"""Differentiable teacher-forced scoring for multi-token continuation anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ContinuationScores:
    proxy_risk: torch.Tensor
    answer_logp: torch.Tensor
    refusal_logp: torch.Tensor


def tokenize_anchor_set(tokenizer: Any, anchors: Sequence[str]) -> tuple[torch.Tensor, ...]:
    encoded = tuple(
        torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        for text in anchors
    )
    if not encoded or any(token_ids.ndim != 1 or token_ids.numel() == 0 for token_ids in encoded):
        raise ValueError("anchor sets require non-empty token sequences")
    return encoded


def score_continuation_sets(
    *,
    model: Any,
    embedding_layer: Any,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    answer_anchors: Sequence[torch.Tensor],
    refusal_anchors: Sequence[torch.Tensor],
) -> ContinuationScores:
    """Score all anchors in one padded model forward without length normalization."""
    if prompt_embeds.ndim != 3 or prompt_attention_mask.shape != prompt_embeds.shape[:2]:
        raise ValueError("prompt embeddings and attention mask have incompatible shapes")
    if not bool(torch.all(prompt_attention_mask == 1)):
        raise ValueError("continuation scoring requires an unpadded prompt")
    anchors = tuple(answer_anchors) + tuple(refusal_anchors)
    answer_count = len(answer_anchors)
    if answer_count == 0 or not refusal_anchors:
        raise ValueError("answer and refusal anchor sets must both be non-empty")
    if any(anchor.ndim != 1 or anchor.numel() == 0 for anchor in anchors):
        raise ValueError("anchors must be non-empty rank-1 token tensors")

    batch_size, prompt_length, hidden_size = prompt_embeds.shape
    max_anchor_length = max(int(anchor.numel()) for anchor in anchors)
    sequence_length = prompt_length + max_anchor_length - 1
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int]] = []
    for batch_index in range(batch_size):
        prompt = prompt_embeds[batch_index : batch_index + 1]
        for anchor_index, anchor in enumerate(anchors):
            token_ids = anchor.to(device=prompt_embeds.device, dtype=torch.long)
            continuation = embedding_layer(token_ids[:-1].unsqueeze(0)).to(dtype=prompt_embeds.dtype)
            row = torch.cat((prompt, continuation), dim=1)
            padding = sequence_length - row.shape[1]
            if padding:
                row = torch.cat((row, torch.zeros((1, padding, hidden_size), device=row.device, dtype=row.dtype)), dim=1)
            mask = torch.zeros((1, sequence_length), device=prompt_attention_mask.device, dtype=prompt_attention_mask.dtype)
            mask[:, : prompt_length + token_ids.numel() - 1] = 1
            rows.append(row)
            masks.append(mask)
            metadata.append((batch_index, anchor_index, int(token_ids.numel())))

    inputs = torch.cat(rows, dim=0)
    attention = torch.cat(masks, dim=0)
    output = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != inputs.shape[:2]:
        raise ValueError("causal model returned invalid logits")
    log_probs = functional.log_softmax(logits.float(), dim=-1)

    per_batch: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    for row_index, (batch_index, anchor_index, anchor_length) in enumerate(metadata):
        anchor = anchors[anchor_index].to(device=log_probs.device, dtype=torch.long)
        positions = torch.arange(
            prompt_length - 1,
            prompt_length - 1 + anchor_length,
            device=log_probs.device,
        )
        score = log_probs[row_index, positions, anchor].sum()
        per_batch[batch_index].append(score)
    stacked = torch.stack(tuple(torch.stack(scores) for scores in per_batch))
    answer_logp = stacked[:, :answer_count].mean(dim=1)
    refusal_logp = stacked[:, answer_count:].mean(dim=1)
    return ContinuationScores(
        proxy_risk=answer_logp - refusal_logp,
        answer_logp=answer_logp,
        refusal_logp=refusal_logp,
    )
