"""Tests for the semantically blind embedding roundtrip runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_blind_alpaca_embedding_roundtrip.py"
SPEC = importlib.util.spec_from_file_location("blind_roundtrip", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_random_output_is_seeded_and_does_not_require_text_inspection() -> None:
    rows = [{"output": "first"}, {"output": "second"}]

    assert MODULE.select_random_output(rows, seed=9) == MODULE.select_random_output(rows, seed=9)


def test_checkpoint_steps_include_each_interval_and_final_step() -> None:
    assert MODULE.checkpoint_steps(200, 50) == [50, 100, 150, 200]


def test_source_embeddings_can_be_used_as_a_frozen_optimization_target() -> None:
    embedding_layer = torch.nn.Embedding(4, 3)
    source_ids = torch.tensor([[0, 1]])

    target = MODULE.source_embeddings_from_ids(embedding_layer, source_ids)
    state = target.clone().requires_grad_(True)
    torch.nn.functional.mse_loss(state, target).backward()

    assert not target.requires_grad
    assert state.grad is not None


def test_target_token_ids_are_deterministically_different_from_source_ids() -> None:
    source_ids = torch.tensor([[0, 7, 9]])

    target_ids = MODULE.different_target_ids(source_ids, vocab_size=10, offset=3)

    assert torch.equal(target_ids, torch.tensor([[3, 0, 2]]))
    assert not torch.any(target_ids == source_ids)


def test_materialized_layout_preserves_both_frozen_segments_verbatim() -> None:
    materialized = MODULE.combine_segment_text("<z>", "[frozen-1]", "<u>", "[frozen-2]")

    assert materialized == "<z>[frozen-1]<u>[frozen-2]"
    assert "[frozen-1]" in materialized
    assert "[frozen-2]" in materialized


def test_readable_target_ids_use_the_allowed_vocabulary_and_avoid_source_ids() -> None:
    source_ids = torch.tensor([[1, 3]])
    readable_ids = torch.tensor([1, 2, 3])

    target_ids = MODULE.readable_distinct_target_ids(source_ids, readable_ids, offset=0)

    assert torch.equal(target_ids, torch.tensor([[2, 1]]))
    assert not torch.any(target_ids == source_ids)


def test_progressive_target_mask_activates_new_tokens_at_each_checkpoint() -> None:
    active_counts = [
        int(MODULE.progressive_target_mask(total_tokens=23, stage=stage, stages=4).sum())
        for stage in range(5)
    ]

    assert active_counts == [0, 6, 12, 18, 23]


def test_initial_prompt_uses_original_segment_ids_without_projection() -> None:
    class FakeTokenizer:
        @staticmethod
        def decode(ids: list[int], **_: object) -> str:
            return "/".join(str(token_id) for token_id in ids)

    prompt = MODULE.initial_prompt_from_ids(
        FakeTokenizer(), torch.tensor([[7, 8]]), "[frozen-1]", torch.tensor([[1, 2]]), "[frozen-2]"
    )

    assert prompt == "7/8[frozen-1]1/2[frozen-2]"


def test_checkpoint_json_preserves_unicode_text_without_ascii_escaping() -> None:
    serialized = MODULE.serialize_checkpoint_record(50, "prefix \u4e2d suffix")

    assert '"step": 50' in serialized
    assert "\u4e2d" in serialized
    assert r"\u4e2d" not in serialized


def test_ascii_english_vocabulary_excludes_non_ascii_tokens_and_keeps_punctuation() -> None:
    class FakeTokenizer:
        @staticmethod
        def decode(ids: list[int], **_: object) -> str:
            return {0: " word", 1: "中", 2: "!", 3: "<|special|>"}[ids[0]]

    token_ids = MODULE.ascii_english_vocabulary_ids(FakeTokenizer(), vocab_size=4)

    assert torch.equal(token_ids, torch.tensor([0, 2]))
