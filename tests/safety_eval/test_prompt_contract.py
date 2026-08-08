from __future__ import annotations

import torch

from benchmark.safety_eval.prompt_contract import scatter_editable, tokenize_editable_prompt
from benchmark.safety_eval.schema import EditableSpan


class _OffsetTokenizer:
    def __init__(self, token_ids: list[int], offsets: list[tuple[int, int]]) -> None:
        self.token_ids = token_ids
        self.offsets = offsets

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {
            "return_tensors": "pt",
            "add_special_tokens": True,
            "return_offsets_mapping": True,
        }
        return {
            "input_ids": torch.tensor([self.token_ids]),
            "attention_mask": torch.ones((1, len(self.token_ids)), dtype=torch.long),
            "offset_mapping": torch.tensor([self.offsets]),
        }


def _span(start: int, end: int, quote: str) -> EditableSpan:
    return EditableSpan(
        start=start,
        end=end,
        quote=quote,
        role="harmful_payload",
        confidence=1.0,
        rationale="fixture",
    )


def test_tokenize_editable_prompt_maps_overlapping_spans_without_retokenizing_fragments() -> None:
    prompt = tokenize_editable_prompt(
        "abcdefghij",
        spans=(_span(1, 3, "bc"), _span(7, 9, "hi")),
        tokenizer=_OffsetTokenizer(
            token_ids=[10, 11, 12, 13],
            offsets=[(0, 2), (2, 5), (5, 8), (8, 10)],
        ),
        tokenizer_revision="fixture-r1",
    )

    assert prompt.editable_positions == (0, 1, 2, 3)
    assert prompt.frozen_positions == ()
    assert prompt.boundary_expansions == ((0, 5), (5, 10))
    assert prompt.gather_editable_ids().tolist() == [[10, 11, 12, 13]]


def test_scatter_editable_replaces_only_omega_s() -> None:
    base = torch.tensor([[10, 11, 12, 13]])
    result = scatter_editable(torch.tensor([[30, 31]]), base, (1, 3))

    assert result.tolist() == [[10, 30, 12, 31]]
    assert result[:, (0, 2)].tolist() == base[:, (0, 2)].tolist()
