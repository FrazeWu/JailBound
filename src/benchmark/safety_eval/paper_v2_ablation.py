"""Same-state continuous/materialized helpers for paper-aligned prompt states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .prompt_contract import TokenizedEditablePrompt, scatter_editable


@dataclass(frozen=True)
class PaperV2ContinuousChatInput:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    base_content_token_ids: tuple[int, ...]
    content_start: int


def _tensor_field(payload: object, field: str) -> torch.Tensor:
    value = payload.get(field) if isinstance(payload, dict) else getattr(payload, field, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"tokenizer did not return tensor {field}")
    return value


def build_paper_v2_continuous_chat_input(
    *,
    model: Any,
    tokenizer: Any,
    prompt: TokenizedEditablePrompt,
    z: torch.Tensor,
    u: torch.Tensor,
) -> PaperV2ContinuousChatInput:
    """Insert ``[z; Phi_tilde(p; U)]`` into one standard user-content span."""
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt.full_text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or rendered.count(prompt.full_text) != 1:
        raise ValueError("chat template does not contain one exact user-content span")
    content_start_char = rendered.index(prompt.full_text)
    content_end_char = content_start_char + len(prompt.full_text)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _tensor_field(encoded, "input_ids")
    offsets = _tensor_field(encoded, "offset_mapping")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or offsets.shape != (*input_ids.shape, 2):
        raise ValueError("chat tokenizer returned invalid offsets")
    selected = tuple(
        index
        for index, (start, end) in enumerate(offsets[0].detach().cpu().tolist())
        if end > content_start_char and start < content_end_char
    )
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise ValueError("user-content token span is empty or non-contiguous")
    content_ids = input_ids[:, selected[0] : selected[-1] + 1]
    expected_ids = prompt.base_token_ids.to(device=content_ids.device, dtype=content_ids.dtype)
    if not torch.equal(content_ids, expected_ids):
        raise ValueError("chat-template content tokenization differs from the prompt contract")

    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError("model has no input embedding matrix")
    ids = input_ids.to(device=weight.device, dtype=torch.long)
    with torch.no_grad():
        rendered_embeddings = embedding(ids).detach()
    z = z.to(device=weight.device, dtype=rendered_embeddings.dtype)
    u = u.to(device=weight.device, dtype=rendered_embeddings.dtype)
    base_embeddings = rendered_embeddings[:, selected[0] : selected[-1] + 1]
    replaced = scatter_editable(u, base_embeddings, prompt.editable_positions)
    full = torch.cat(
        (
            rendered_embeddings[:, : selected[0]],
            z,
            replaced,
            rendered_embeddings[:, selected[-1] + 1 :],
        ),
        dim=1,
    )
    attention = torch.ones(full.shape[:2], dtype=torch.long, device=full.device)
    return PaperV2ContinuousChatInput(
        inputs_embeds=full,
        attention_mask=attention,
        base_content_token_ids=tuple(int(token_id) for token_id in content_ids[0].detach().cpu().tolist()),
        content_start=selected[0],
    )
