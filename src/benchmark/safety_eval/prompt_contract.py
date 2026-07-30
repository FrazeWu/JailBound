"""Map annotated prompt character spans to replaceable token positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


TokenOffset = tuple[int, int]


@dataclass(frozen=True)
class TokenizedEditablePrompt:
    """A single tokenized prompt partitioned into editable and frozen tokens."""

    full_text: str
    base_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    editable_positions: tuple[int, ...]
    frozen_positions: tuple[int, ...]
    token_offsets: tuple[TokenOffset, ...]
    boundary_expansions: tuple[TokenOffset, ...]
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.full_text, str):
            raise ValueError("full_text must be a string")
        if not isinstance(self.tokenizer_revision, str) or not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision must be non-empty")
        if not isinstance(self.base_token_ids, torch.Tensor) or not isinstance(
            self.attention_mask, torch.Tensor
        ):
            raise ValueError("base_token_ids and attention_mask must be tensors")
        if self.base_token_ids.ndim != 2 or self.base_token_ids.shape[0] != 1:
            raise ValueError("base_token_ids must have shape [1, tokens]")
        if self.attention_mask.shape != self.base_token_ids.shape:
            raise ValueError(
                "attention_mask must have the same [1, tokens] shape as base_token_ids"
            )

        token_count = self.base_token_ids.shape[1]
        _require_tuple("editable_positions", self.editable_positions)
        _require_tuple("frozen_positions", self.frozen_positions)
        _require_tuple("token_offsets", self.token_offsets)
        _require_tuple("boundary_expansions", self.boundary_expansions)

        if not self.editable_positions:
            raise ValueError("editable_positions must be non-empty")
        if self.editable_positions != tuple(sorted(set(self.editable_positions))):
            raise ValueError("editable_positions must be sorted and unique")
        expected_frozen = tuple(
            index for index in range(token_count) if index not in self.editable_positions
        )
        if self.frozen_positions != expected_frozen:
            raise ValueError(
                "frozen_positions must be the exact ordered complement of editable_positions"
            )
        if any(
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position >= token_count
            for position in self.editable_positions
        ):
            raise ValueError("editable_positions contain an invalid token index")
        if len(self.token_offsets) != token_count:
            raise ValueError("token_offsets length must match the token sequence length")
        for offset in self.token_offsets:
            _validate_interval(offset, len(self.full_text), "token_offsets", allow_zero_width=True)
        if not self.boundary_expansions:
            raise ValueError("boundary_expansions must be non-empty")
        for expansion in self.boundary_expansions:
            _validate_interval(
                expansion,
                len(self.full_text),
                "boundary_expansions",
                allow_zero_width=False,
            )

    def gather_editable_ids(self) -> torch.Tensor:
        """Select editable token IDs along the prompt sequence dimension."""
        index = torch.tensor(self.editable_positions, device=self.base_token_ids.device)
        return self.base_token_ids.index_select(1, index)


def scatter_editable(
    values: torch.Tensor,
    base: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    """Return a clone of ``base`` with sequence positions replaced by ``values``."""
    if not isinstance(base, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError("base and values must be tensors with compatible shapes")
    if base.ndim < 2 or values.ndim != base.ndim:
        raise ValueError("base and values must have compatible shapes with a sequence dimension")

    try:
        normalized_positions = tuple(positions)
    except TypeError as error:
        raise ValueError("positions must be a sequence of token indices") from error
    if not normalized_positions:
        raise ValueError("positions must be non-empty")
    if any(
        not isinstance(position, int) or isinstance(position, bool)
        for position in normalized_positions
    ):
        raise ValueError("positions must contain only integer token indices")
    if len(set(normalized_positions)) != len(normalized_positions):
        raise ValueError("positions must not contain duplicates")
    if any(position < 0 or position >= base.shape[1] for position in normalized_positions):
        raise ValueError("positions must be within the base sequence dimension")

    expected_shape = (base.shape[0], len(normalized_positions), *base.shape[2:])
    if tuple(values.shape) != expected_shape:
        raise ValueError(
            f"values shape must be {expected_shape} for base shape {tuple(base.shape)} "
            "and positions"
        )

    result = base.clone()
    result[:, list(normalized_positions), ...] = values
    return result


def tokenize_editable_prompt(
    full_text: str,
    spans: Sequence[Any],
    tokenizer: Any,
    tokenizer_revision: str,
) -> TokenizedEditablePrompt:
    """Tokenize a complete prompt once and map annotated spans to whole tokens."""
    if not isinstance(full_text, str):
        raise ValueError("full_text must be a string")
    if not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
        raise ValueError("tokenizer_revision must be non-empty")

    try:
        normalized_spans = tuple(spans)
    except TypeError as error:
        raise ValueError("at least one editable span is required") from error
    if not normalized_spans:
        raise ValueError("at least one editable span is required")
    span_intervals = _validate_spans(full_text, normalized_spans)

    encoded = tokenizer(full_text, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer output must be a mapping")
    for field in ("input_ids", "attention_mask", "offset_mapping"):
        if field not in encoded or encoded[field] is None:
            raise ValueError(f"tokenizer output must include usable {field}")

    input_ids = _normalize_token_field(encoded["input_ids"], "input_ids")
    attention_mask = _normalize_token_field(encoded["attention_mask"], "attention_mask")
    token_offsets = _normalize_offsets(encoded["offset_mapping"])
    lengths = (input_ids.shape[1], attention_mask.shape[1], len(token_offsets))
    if len(set(lengths)) != 1:
        raise ValueError(
            "input_ids, attention_mask, and offset_mapping must have matching sequence lengths"
        )

    editable: set[int] = set()
    boundary_expansions: list[TokenOffset] = []
    for span_start, span_end in span_intervals:
        mapped = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_start < token_end
            and token_start < span_end
            and token_end > span_start
        ]
        if not mapped:
            raise ValueError("every editable span must map to at least one ordinary token")
        editable.update(mapped)
        boundary_expansions.append(
            (
                min(token_offsets[index][0] for index in mapped),
                max(token_offsets[index][1] for index in mapped),
            )
        )

    editable_positions = tuple(sorted(editable))
    if not editable_positions:
        raise ValueError("editable token set must be non-empty")
    frozen_positions = tuple(
        index for index in range(input_ids.shape[1]) if index not in editable
    )
    return TokenizedEditablePrompt(
        full_text=full_text,
        base_token_ids=input_ids,
        attention_mask=attention_mask,
        editable_positions=editable_positions,
        frozen_positions=frozen_positions,
        token_offsets=token_offsets,
        boundary_expansions=tuple(boundary_expansions),
        tokenizer_revision=tokenizer_revision,
    )


def _validate_spans(full_text: str, spans: tuple[Any, ...]) -> tuple[TokenOffset, ...]:
    intervals: list[TokenOffset] = []
    previous_end: int | None = None
    for span in spans:
        try:
            start = span.start
            end = span.end
            quote = span.quote
        except AttributeError as error:
            raise ValueError("each editable span must provide start, end, and quote") from error
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(full_text)
        ):
            raise ValueError("editable span bounds must satisfy 0 <= start < end <= len(full_text)")
        if not isinstance(quote, str) or full_text[start:end] != quote:
            raise ValueError("editable span quote must exactly match full_text")
        if previous_end is not None and start < previous_end:
            raise ValueError("editable spans must be ordered and non-overlapping")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _normalize_token_field(value: Any, field: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"tokenizer output {field} must contain one usable sequence") from error
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(f"tokenizer output {field} must contain exactly one sequence")
    return tensor.to(dtype=torch.long).clone()


def _normalize_offsets(value: Any) -> tuple[TokenOffset, ...]:
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu()
    else:
        try:
            raw = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "tokenizer output offset_mapping must contain character interval pairs"
            ) from error

    if raw.numel() == 0:
        return ()
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw.squeeze(0)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(
            "tokenizer output offset_mapping must contain exactly one sequence of pairs"
        )
    if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
        raise ValueError("tokenizer output offset_mapping must contain integer pairs")

    return tuple((int(start), int(end)) for start, end in raw.tolist())


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")


def _validate_interval(
    interval: object,
    text_length: int,
    field: str,
    *,
    allow_zero_width: bool,
) -> None:
    if (
        not isinstance(interval, tuple)
        or len(interval) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in interval)
    ):
        raise ValueError(f"{field} must contain integer interval tuples")
    start, end = interval
    if not 0 <= start <= end <= text_length or (not allow_zero_width and start == end):
        raise ValueError(f"{field} contains an invalid character interval")
