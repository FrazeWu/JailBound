from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

from benchmark.safety_eval.anchor_scorer import score_continuation_sets


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.embedding = nn.Embedding(7, 4)
        self.projection = nn.Linear(4, 7, bias=False)
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool):
        assert use_cache is False
        self.forward_calls += 1
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def _manual_logp(model: _TinyCausalModel, prompt: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    context = prompt
    total = torch.zeros((), dtype=prompt.dtype)
    for token_id in anchor:
        logits = model.projection(context.cumsum(dim=1))[:, -1, :]
        total = total + functional.log_softmax(logits, dim=-1)[0, token_id]
        token_embed = model.embedding(token_id.reshape(1, 1))
        context = torch.cat((context, token_embed), dim=1)
    return total


def test_continuation_scorer_sums_full_teacher_forced_anchor_logps_in_one_forward() -> None:
    model = _TinyCausalModel()
    prompt = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    answer = (torch.tensor([1, 2]), torch.tensor([3]))
    refusal = (torch.tensor([4, 5, 6]),)
    expected_answer = (_manual_logp(model, prompt, answer[0]) + _manual_logp(model, prompt, answer[1])) / 2
    expected_refusal = _manual_logp(model, prompt, refusal[0])
    model.forward_calls = 0

    result = score_continuation_sets(
        model=model,
        embedding_layer=model.get_input_embeddings(),
        prompt_embeds=prompt,
        prompt_attention_mask=torch.ones((1, 1), dtype=torch.long),
        answer_anchors=answer,
        refusal_anchors=refusal,
    )

    assert torch.allclose(result.answer_logp, expected_answer.unsqueeze(0))
    assert torch.allclose(result.refusal_logp, expected_refusal.unsqueeze(0))
    assert torch.allclose(result.proxy_risk, (expected_answer - expected_refusal).unsqueeze(0))
    assert model.forward_calls == 1
    assert torch.autograd.grad(result.proxy_risk.sum(), prompt)[0].abs().sum() > 0
