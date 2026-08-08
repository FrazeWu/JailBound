"""Paper-aligned editable prompt token mapping and reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .schema import EditableSpan


def _tensor_field(payload: object, field: str) -> torch.Tensor:
    value = payload.get(field) if isinstance(payload, dict) else getattr(payload, field, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"tokenizer did not return tensor {field}")
    return value


@dataclass(frozen=True)
class TokenizedEditablePrompt:
    full_text: str
    base_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    editable_positions: tuple[int, ...]
    frozen_positions: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    boundary_expansions: tuple[tuple[int, int], ...]
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if self.base_token_ids.ndim != 2 or self.base_token_ids.shape[0] != 1:
            raise ValueError("base token IDs must have shape [1, tokens]")
        if self.attention_mask.shape != self.base_token_ids.shape:
            raise ValueError("prompt attention mask has incompatible shape")
        if len(self.token_offsets) != self.base_token_ids.shape[1]:
            raise ValueError("token offsets do not cover the base sequence")
        positions = self.editable_positions + self.frozen_positions
        if len(set(positions)) != len(positions) or set(positions) != set(range(self.base_token_ids.shape[1])):
            raise ValueError("editable and frozen positions must partition the base sequence")
        if not self.editable_positions:
            raise ValueError("prompt requires at least one editable token")
        if not self.tokenizer_revision.strip():
            raise ValueError("tokenizer revision is required")

    def gather_editable_ids(self) -> torch.Tensor:
        index = torch.tensor(self.editable_positions, dtype=torch.long, device=self.base_token_ids.device)
        return self.base_token_ids.index_select(1, index)


def scatter_editable(values: torch.Tensor, base: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    """Replace ``Omega_s`` along the token axis while preserving all other positions."""
    if base.ndim < 2 or values.ndim != base.ndim or values.shape[0] != base.shape[0]:
        raise ValueError("editable values and base tensor have incompatible shapes")
    if values.shape[1] != len(positions):
        raise ValueError("editable values do not match Omega_s")
    if values.shape[2:] != base.shape[2:]:
        raise ValueError("editable values and base tensor have incompatible trailing dimensions")
    if len(set(positions)) != len(positions) or any(position < 0 or position >= base.shape[1] for position in positions):
        raise ValueError("editable positions are invalid")
    result = base.clone()
    index = torch.tensor(positions, dtype=torch.long, device=base.device)
    result.index_copy_(1, index, values.to(device=base.device, dtype=base.dtype))
    return result


def _validate_spans(full_text: str, spans: Sequence[EditableSpan]) -> None:
    if not full_text or not spans:
        raise ValueError("editable prompt requires text and spans")
    previous_end = -1
    for span in spans:
        if span.start < previous_end:
            raise ValueError("editable spans must be ordered and non-overlapping")
        if span.end > len(full_text) or full_text[span.start : span.end] != span.quote:
            raise ValueError("editable span quote does not match full text")
        previous_end = span.end


def tokenize_editable_prompt(
    full_text: str,
    *,
    spans: Sequence[EditableSpan],
    tokenizer: Any,
    tokenizer_revision: str,
) -> TokenizedEditablePrompt:
    """Tokenize the complete prompt once and map annotated spans by offset overlap."""
    _validate_spans(full_text, spans)
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = _tensor_field(encoded, "input_ids")
    attention_mask = _tensor_field(encoded, "attention_mask")
    offset_tensor = _tensor_field(encoded, "offset_mapping")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or offset_tensor.shape != (*input_ids.shape, 2):
        raise ValueError("tokenizer returned invalid prompt offsets")
    offsets = tuple((int(start), int(end)) for start, end in offset_tensor[0].detach().cpu().tolist())

    editable: set[int] = set()
    expansions: list[tuple[int, int]] = []
    for span in spans:
        positions = tuple(
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > span.start and start < span.end
        )
        if not positions:
            raise ValueError("editable span maps to no ordinary tokens")
        editable.update(positions)
        expansions.append((offsets[positions[0]][0], offsets[positions[-1]][1]))
    editable_positions = tuple(sorted(editable))
    frozen_positions = tuple(index for index in range(input_ids.shape[1]) if index not in editable)
    return TokenizedEditablePrompt(
        full_text=full_text,
        base_token_ids=input_ids,
        attention_mask=attention_mask,
        editable_positions=editable_positions,
        frozen_positions=frozen_positions,
        token_offsets=offsets,
        boundary_expansions=tuple(expansions),
        tokenizer_revision=tokenizer_revision,
    )
