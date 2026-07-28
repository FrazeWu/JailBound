from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "freeze_semantic_calibration.py"
_SPEC = importlib.util.spec_from_file_location("freeze_semantic_calibration", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_paraphrase = _MODULE._paraphrase


def test_paraphrase_passes_an_explicit_attention_mask_to_generation() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_: object, **__: object) -> torch.Tensor:
            return torch.tensor([[11, 12]])

        def decode(self, value: torch.Tensor, **_: object) -> str:
            assert value.tolist() == [13]
            return "rewritten fixture"

    class Model:
        def get_input_embeddings(self) -> SimpleNamespace:
            return SimpleNamespace(weight=torch.empty(1))

        def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            assert torch.equal(kwargs["attention_mask"], torch.ones_like(input_ids))
            return torch.tensor([[11, 12, 13]])

    assert _paraphrase(Model(), Tokenizer(), "fixture") == "rewritten fixture"
