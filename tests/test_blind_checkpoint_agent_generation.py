"""Tests for blind checkpoint agent generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_blind_checkpoint_agent_generation.py"
SPEC = importlib.util.spec_from_file_location("blind_agent_generation", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generate_checkpoint_outputs_uses_text_and_embedding_inputs_separately() -> None:
    class FakeTokenizer:
        def __call__(self, _: str, **__: object) -> SimpleNamespace:
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

        def decode(self, ids: list[int] | torch.Tensor, **__: object) -> str:
            values = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            return "text-response" if values == [3] else "embedding-response"

    class FakeModel:
        def __init__(self) -> None:
            self.embedding = torch.nn.Embedding(8, 4)
            self.calls: list[str] = []

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embedding

        def generate(self, **kwargs: object) -> torch.Tensor:
            if "input_ids" in kwargs:
                self.calls.append("text")
                return torch.tensor([[1, 2, 3]])
            self.calls.append("embedding")
            return torch.tensor([[4]])

    model = FakeModel()
    text_records, embedding_records = MODULE.generate_checkpoint_outputs(
        model, FakeTokenizer(), [{"step": 50, "prompt": "opaque"}], max_new_tokens=8
    )

    assert model.calls == ["text", "embedding"]
    assert text_records == [{"step": 50, "output": "text-response"}]
    assert embedding_records == [{"step": 50, "output": "embedding-response"}]


def test_checkpoint_source_paths_can_select_ascii_only() -> None:
    ascii_path = Path("ascii.jsonl")
    utf8_path = Path("utf8.jsonl")

    selected = MODULE.checkpoint_source_paths(["ascii_english"], utf8_path, ascii_path)

    assert selected == {"ascii_english": ascii_path}
