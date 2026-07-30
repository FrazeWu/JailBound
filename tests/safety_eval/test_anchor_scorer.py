from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from benchmark.safety_eval.anchor_scorer import (
    score_continuation_sets,
    tokenize_anchor_set,
)


class TinyCausalModel(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.embeddings.weight.copy_(
                torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(
                    vocab_size, hidden_size
                )
                / 10
            )
            self.projection.weight.copy_(
                torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(
                    vocab_size, hidden_size
                )
                / 20
            )
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        self.forward_calls += 1
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def manual_logp(
    model: TinyCausalModel, prompt: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    token_embeds = model.get_input_embeddings()(anchor[:-1]).unsqueeze(0)
    inputs = torch.cat((prompt, token_embeds), dim=1)
    logits = model.projection(inputs.cumsum(dim=1))[0, : anchor.numel()]
    return F.log_softmax(logits, dim=-1).gather(1, anchor.unsqueeze(1)).sum()


def test_continuation_scorer_sums_teacher_forced_token_logps() -> None:
    model = TinyCausalModel(vocab_size=7, hidden_size=4)
    prompt = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    answer = (torch.tensor([1, 2]), torch.tensor([3]))
    refusal = (torch.tensor([4, 5, 6]),)

    result = score_continuation_sets(
        model=model,
        embedding_layer=model.get_input_embeddings(),
        prompt_embeds=prompt,
        prompt_attention_mask=torch.ones((1, 1), dtype=torch.long),
        answer_anchors=answer,
        refusal_anchors=refusal,
    )
    expected_answer = (
        manual_logp(model, prompt, answer[0])
        + manual_logp(model, prompt, answer[1])
    ) / 2
    expected_refusal = manual_logp(model, prompt, refusal[0])

    assert torch.allclose(result.answer_logp, expected_answer)
    assert torch.allclose(result.refusal_logp, expected_refusal)
    assert torch.allclose(result.proxy_risk, expected_answer - expected_refusal)
    assert model.forward_calls == 1
    result.proxy_risk.backward()
    assert prompt.grad is not None and prompt.grad.abs().sum() > 0


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


@pytest.mark.parametrize("anchors", ((), ("",), ("ok", "")))
def test_tokenize_anchor_set_rejects_empty_sequences(anchors: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="non-empty token sequences"):
        tokenize_anchor_set(_Tokenizer(), anchors)

