from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from benchmark.safety_eval.prompt_contract import (
    TokenizedEditablePrompt,
    scatter_editable,
    tokenize_editable_prompt,
)
from benchmark.safety_eval.schema import EditableSpan, EditableSpanRole


class _FixtureTokenizer:
    def __init__(self, output: Mapping[str, Any]) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append((text, kwargs))
        return self.output


@dataclass(frozen=True)
class _RawSpan:
    start: int
    end: int
    quote: str


def _span(text: str, start: int, end: int) -> EditableSpan:
    return EditableSpan(
        start=start,
        end=end,
        quote=text[start:end],
        role=EditableSpanRole.harmful_payload,
        confidence=1.0,
        rationale="test fixture",
    )


def _tokenizer(
    offsets: list[tuple[int, int]],
    *,
    input_ids: list[int] | None = None,
    attention_mask: list[int] | None = None,
) -> _FixtureTokenizer:
    token_count = len(offsets)
    return _FixtureTokenizer(
        {
            "input_ids": input_ids if input_ids is not None else list(range(10, 10 + token_count)),
            "attention_mask": attention_mask if attention_mask is not None else [1] * token_count,
            "offset_mapping": offsets,
        }
    )


def test_repeated_identical_quotes_are_selected_by_their_explicit_offsets() -> None:
    text = "go go"
    tokenizer = _tokenizer([(0, 0), (0, 2), (3, 5), (0, 0)])

    prompt = tokenize_editable_prompt(
        text,
        (_span(text, 0, 2), _span(text, 3, 5)),
        tokenizer,
        "fixture-r1",
    )

    assert prompt.editable_positions == (1, 2)
    assert prompt.frozen_positions == (0, 3)
    assert prompt.boundary_expansions == ((0, 2), (3, 5))


def test_chinese_text_uses_character_offsets() -> None:
    text = "请勿攻击系统"
    tokenizer = _tokenizer([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    prompt = tokenize_editable_prompt(
        text,
        (_span(text, 2, 4),),
        tokenizer,
        "fixture-zh",
    )

    assert prompt.editable_positions == (2, 3)
    assert prompt.boundary_expansions == ((2, 4),)


def test_multiple_non_contiguous_spans_preserve_the_frozen_complement() -> None:
    text = "alpha beta gamma"
    tokenizer = _tokenizer([(0, 5), (5, 6), (6, 10), (10, 11), (11, 16)])

    prompt = tokenize_editable_prompt(
        text,
        (_span(text, 0, 5), _span(text, 11, 16)),
        tokenizer,
        "fixture-r1",
    )

    assert prompt.editable_positions == (0, 4)
    assert prompt.frozen_positions == (1, 2, 3)
    assert set(prompt.editable_positions).isdisjoint(prompt.frozen_positions)
    assert sorted(prompt.editable_positions + prompt.frozen_positions) == list(range(5))


def test_span_boundaries_expand_to_every_overlapping_whole_token() -> None:
    text = "abcdefghij"
    tokenizer = _tokenizer([(0, 2), (2, 5), (5, 8), (8, 10)])

    prompt = tokenize_editable_prompt(
        text,
        (_span(text, 1, 3), _span(text, 7, 9)),
        tokenizer,
        "fixture-r1",
    )

    assert prompt.editable_positions == (0, 1, 2, 3)
    assert prompt.boundary_expansions == ((0, 5), (5, 10))


@pytest.mark.parametrize("offsets", [[], [(0, 0), (0, 0)]])
def test_empty_or_all_zero_width_offsets_cannot_map_a_span(
    offsets: list[tuple[int, int]],
) -> None:
    text = "abc"
    tokenizer = _tokenizer(offsets)

    with pytest.raises(ValueError, match="map to at least one ordinary token"):
        tokenize_editable_prompt(text, (_span(text, 0, 1),), tokenizer, "fixture-r1")


def test_zero_width_special_tokens_are_ignored_and_remain_frozen() -> None:
    text = "abc"
    tokenizer = _tokenizer(
        [(0, 0), (0, 1), (1, 3), (0, 0)],
        input_ids=[101, 11, 12, 102],
        attention_mask=[1, 1, 1, 1],
    )

    prompt = tokenize_editable_prompt(
        text,
        (_span(text, 0, 3),),
        tokenizer,
        "fixture-r1",
    )

    assert prompt.editable_positions == (1, 2)
    assert prompt.frozen_positions == (0, 3)
    assert prompt.token_offsets == ((0, 0), (0, 1), (1, 3), (0, 0))


@pytest.mark.parametrize(
    "spans",
    [
        (_RawSpan(2, 4, "cd"), _RawSpan(0, 2, "ab")),
        (_RawSpan(0, 3, "abc"), _RawSpan(2, 4, "cd")),
    ],
)
def test_unordered_or_overlapping_spans_are_rejected(
    spans: tuple[_RawSpan, _RawSpan],
) -> None:
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        tokenize_editable_prompt("abcdef", spans, _tokenizer([(0, 6)]), "fixture-r1")


def test_at_least_one_span_is_required() -> None:
    with pytest.raises(ValueError, match="at least one editable span"):
        tokenize_editable_prompt("abc", (), _tokenizer([(0, 3)]), "fixture-r1")


@pytest.mark.parametrize(
    "span",
    [
        _RawSpan(-1, 1, "a"),
        _RawSpan(1, 1, ""),
        _RawSpan(2, 1, ""),
        _RawSpan(0, 4, "abc"),
    ],
)
def test_invalid_or_zero_width_span_bounds_are_rejected_at_the_function_boundary(
    span: _RawSpan,
) -> None:
    with pytest.raises(ValueError, match=r"0 <= start < end <= len\(full_text\)"):
        tokenize_editable_prompt("abc", (span,), _tokenizer([(0, 3)]), "fixture-r1")


def test_span_quote_must_exactly_match_the_supplied_full_text() -> None:
    with pytest.raises(ValueError, match="quote must exactly match full_text"):
        tokenize_editable_prompt(
            "abcdef",
            (_RawSpan(1, 3, "wrong"),),
            _tokenizer([(0, 6)]),
            "fixture-r1",
        )


def test_tokenizer_revision_must_be_non_empty() -> None:
    text = "abc"
    with pytest.raises(ValueError, match="tokenizer_revision must be non-empty"):
        tokenize_editable_prompt(text, (_span(text, 0, 1),), _tokenizer([(0, 3)]), "  ")


def test_tokenizer_must_return_offsets() -> None:
    tokenizer = _FixtureTokenizer({"input_ids": [1], "attention_mask": [1]})

    with pytest.raises(ValueError, match="offset_mapping"):
        tokenize_editable_prompt("a", (_span("a", 0, 1),), tokenizer, "fixture-r1")


def test_complete_text_is_tokenized_exactly_once_with_offset_mapping_requested() -> None:
    text = "one editable phrase"
    tokenizer = _tokenizer([(0, 3), (3, 4), (4, 12), (12, 13), (13, 19)])

    tokenize_editable_prompt(
        text,
        (_span(text, 4, 12),),
        tokenizer,
        "fixture-r1",
    )

    assert tokenizer.calls == [(text, {"return_offsets_mapping": True})]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ({"attention_mask": [1], "offset_mapping": [(0, 1)]}, "input_ids"),
        ({"input_ids": [1], "offset_mapping": [(0, 1)]}, "attention_mask"),
        ({"input_ids": [1], "attention_mask": [1]}, "offset_mapping"),
    ],
)
def test_required_tokenizer_fields_must_be_present(
    output: Mapping[str, Any],
    message: str,
) -> None:
    tokenizer = _FixtureTokenizer(output)

    with pytest.raises(ValueError, match=message):
        tokenize_editable_prompt("a", (_span("a", 0, 1),), tokenizer, "fixture-r1")


@pytest.mark.parametrize(
    "output",
    [
        {"input_ids": [1, 2], "attention_mask": [1], "offset_mapping": [(0, 1), (1, 2)]},
        {"input_ids": [1], "attention_mask": [1, 1], "offset_mapping": [(0, 1), (1, 2)]},
        {"input_ids": [1, 2], "attention_mask": [1, 1], "offset_mapping": [(0, 1)]},
    ],
)
def test_tokenizer_field_sequence_lengths_must_match(output: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError, match="matching sequence lengths"):
        tokenize_editable_prompt(
            "ab",
            (_span("ab", 0, 1),),
            _FixtureTokenizer(output),
            "fixture-r1",
        )


def test_huggingface_style_batched_tensors_normalize_to_one_sequence() -> None:
    tokenizer = _FixtureTokenizer(
        {
            "input_ids": torch.tensor([[101, 7, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 2], [0, 0]]]),
        }
    )

    prompt = tokenize_editable_prompt(
        "hi",
        (_span("hi", 0, 2),),
        tokenizer,
        "hf-fixture",
    )

    assert prompt.base_token_ids.shape == (1, 3)
    assert prompt.attention_mask.shape == (1, 3)
    assert prompt.token_offsets == ((0, 0), (0, 2), (0, 0))
    assert prompt.editable_positions == (1,)


@pytest.mark.parametrize(
    "input_ids",
    [
        pytest.param(torch.tensor([1.9]), id="float"),
        pytest.param(torch.tensor([float("nan")]), id="nan"),
        pytest.param(torch.tensor([1 + 2j]), id="complex"),
        pytest.param(torch.tensor([True]), id="bool"),
        pytest.param(torch.tensor([-1]), id="negative"),
    ],
)
def test_tokenizer_input_ids_must_be_non_negative_integers(
    input_ids: torch.Tensor,
) -> None:
    tokenizer = _FixtureTokenizer(
        {
            "input_ids": input_ids,
            "attention_mask": [1],
            "offset_mapping": [(0, 1)],
        }
    )

    with pytest.raises(ValueError, match="input_ids"):
        tokenize_editable_prompt("a", (_span("a", 0, 1),), tokenizer, "fixture-r1")


@pytest.mark.parametrize(
    "attention_mask",
    [
        pytest.param(torch.tensor([1.0]), id="float"),
        pytest.param(torch.tensor([1 + 0j]), id="complex"),
        pytest.param(torch.tensor([2]), id="non-binary-integer"),
    ],
)
def test_tokenizer_attention_mask_must_be_boolean_or_binary_integer(
    attention_mask: torch.Tensor,
) -> None:
    tokenizer = _FixtureTokenizer(
        {
            "input_ids": [1],
            "attention_mask": attention_mask,
            "offset_mapping": [(0, 1)],
        }
    )

    with pytest.raises(ValueError, match="attention_mask"):
        tokenize_editable_prompt("a", (_span("a", 0, 1),), tokenizer, "fixture-r1")


def test_tokenizer_token_fields_normalize_to_independent_long_tensors() -> None:
    input_ids = torch.tensor([7], dtype=torch.int32)
    attention_mask = torch.tensor([True])
    tokenizer = _FixtureTokenizer(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": [(0, 1)],
        }
    )

    prompt = tokenize_editable_prompt(
        "a", (_span("a", 0, 1),), tokenizer, "fixture-r1"
    )

    assert prompt.base_token_ids.dtype == torch.long
    assert prompt.attention_mask.dtype == torch.long
    assert prompt.base_token_ids.data_ptr() != input_ids.data_ptr()
    assert prompt.attention_mask.data_ptr() != attention_mask.data_ptr()


def test_gather_editable_ids_returns_sequence_dimension_values() -> None:
    prompt = TokenizedEditablePrompt(
        full_text="abc",
        base_token_ids=torch.tensor([[101, 11, 12, 102]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        editable_positions=(1, 2),
        frozen_positions=(0, 3),
        token_offsets=((0, 0), (0, 1), (1, 3), (0, 0)),
        boundary_expansions=((0, 3),),
        tokenizer_revision="fixture-r1",
    )

    gathered = prompt.gather_editable_ids()

    assert gathered.shape == (1, 2)
    assert torch.equal(gathered, torch.tensor([[11, 12]]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_token_ids": torch.tensor([1, 2])},
        {"attention_mask": torch.ones((1, 2), dtype=torch.long)},
        {"editable_positions": (2, 1), "frozen_positions": (0, 3)},
        {"editable_positions": (1,), "frozen_positions": (0, 3)},
        {"token_offsets": ((0, 0),)},
    ],
)
def test_tokenized_prompt_rejects_internally_inconsistent_shapes_and_partitions(
    overrides: dict[str, Any],
) -> None:
    fields: dict[str, Any] = {
        "full_text": "abc",
        "base_token_ids": torch.tensor([[101, 11, 12, 102]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "editable_positions": (1, 2),
        "frozen_positions": (0, 3),
        "token_offsets": ((0, 0), (0, 1), (1, 3), (0, 0)),
        "boundary_expansions": ((0, 3),),
        "tokenizer_revision": "fixture-r1",
    }
    fields.update(overrides)

    with pytest.raises(ValueError):
        TokenizedEditablePrompt(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("base_token_ids", torch.tensor([[1.9]]), id="ids-float"),
        pytest.param(
            "base_token_ids",
            torch.tensor([[float("nan")]]),
            id="ids-nan",
        ),
        pytest.param("base_token_ids", torch.tensor([[1 + 2j]]), id="ids-complex"),
        pytest.param("base_token_ids", torch.tensor([[True]]), id="ids-bool"),
        pytest.param("base_token_ids", torch.tensor([[-1]]), id="ids-negative"),
        pytest.param("attention_mask", torch.tensor([[1.0]]), id="mask-float"),
        pytest.param("attention_mask", torch.tensor([[1 + 0j]]), id="mask-complex"),
        pytest.param("attention_mask", torch.tensor([[2]]), id="mask-non-binary"),
    ],
)
def test_tokenized_prompt_rejects_invalid_token_field_values(
    field: str,
    value: torch.Tensor,
) -> None:
    fields: dict[str, Any] = {
        "full_text": "a",
        "base_token_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
        "editable_positions": (0,),
        "frozen_positions": (),
        "token_offsets": ((0, 1),),
        "boundary_expansions": ((0, 1),),
        "tokenizer_revision": "fixture-r1",
    }
    fields[field] = value

    with pytest.raises(ValueError, match=field):
        TokenizedEditablePrompt(**fields)


def test_tokenized_prompt_accepts_boolean_attention_mask() -> None:
    prompt = TokenizedEditablePrompt(
        full_text="a",
        base_token_ids=torch.tensor([[1]]),
        attention_mask=torch.tensor([[True]]),
        editable_positions=(0,),
        frozen_positions=(),
        token_offsets=((0, 1),),
        boundary_expansions=((0, 1),),
        tokenizer_revision="fixture-r1",
    )

    assert prompt.attention_mask.dtype == torch.bool


def test_scatter_replaces_token_ids_without_mutating_base_or_frozen_positions() -> None:
    base = torch.tensor([[101, 11, 12, 13, 102]], dtype=torch.long)
    before = base.clone()

    result = scatter_editable(torch.tensor([[71, 73]]), base, (1, 3))

    assert torch.equal(result, torch.tensor([[101, 71, 12, 73, 102]]))
    assert torch.equal(base, before)
    assert result.data_ptr() != base.data_ptr()
    assert base[:, (0, 2, 4)].numpy().tobytes() == result[:, (0, 2, 4)].numpy().tobytes()


def test_scatter_replaces_embedding_vectors_along_sequence_dimension() -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    values = torch.tensor(
        [
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]],
        ]
    )

    result = scatter_editable(values, base, (0, 3))

    assert torch.equal(result[:, (0, 3), :], values)
    assert torch.equal(result[:, (1, 2), :], base[:, (1, 2), :])


@pytest.mark.parametrize(
    ("values", "base", "positions"),
    [
        (torch.ones((1, 1)), torch.ones(3), (0,)),
        (torch.ones(1), torch.ones((1, 3)), (0,)),
        (torch.ones((2, 1)), torch.ones((1, 3)), (0,)),
        (torch.ones((1, 2)), torch.ones((1, 3)), (0,)),
        (torch.ones((1, 1, 3)), torch.ones((1, 3, 2)), (0,)),
    ],
)
def test_scatter_rejects_shape_mismatches(
    values: torch.Tensor,
    base: torch.Tensor,
    positions: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        scatter_editable(values, base, positions)


@pytest.mark.parametrize(
    "positions",
    [(), (1, 1), (-1,), (3,), (1.5,), (True,)],
)
def test_scatter_rejects_invalid_duplicate_or_out_of_range_positions(
    positions: tuple[Any, ...],
) -> None:
    values = torch.empty((1, len(positions)))

    with pytest.raises(ValueError, match="positions"):
        scatter_editable(values, torch.ones((1, 3)), positions)
