from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.safety_eval.transformer_objective import TransformerObjectiveAdapter
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.prompt_contract import TokenizedEditablePrompt


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embeddings = nn.Embedding(8, 3)
        self.projection = nn.Linear(3, 8, bias=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool):
        self.calls.append((inputs_embeds.detach().clone(), attention_mask.detach().clone(), use_cache))
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def _adapter(model: _TinyCausalModel) -> TransformerObjectiveAdapter:
    return TransformerObjectiveAdapter(
        model,
        prompt=_editable_prompt([1, 2], (1,)),
        answer_anchor_ids=(torch.tensor([1, 2]),),
        refusal_anchor_ids=(torch.tensor([3, 4]),),
        epsilon=0.25,
        lambda_fol=0.5,
        gamma_z=0.1,
        gamma_u=0.2,
    )


def _editable_prompt(
    base_ids: list[int] = [1, 2, 3, 4],
    editable_positions: tuple[int, ...] = (1, 3),
) -> TokenizedEditablePrompt:
    return TokenizedEditablePrompt(
        full_text="abcd",
        base_token_ids=torch.tensor([base_ids]),
        attention_mask=torch.ones((1, len(base_ids)), dtype=torch.long),
        editable_positions=editable_positions,
        frozen_positions=tuple(
            index for index in range(len(base_ids)) if index not in editable_positions
        ),
        token_offsets=tuple((index, index + 1) for index in range(len(base_ids))),
        boundary_expansions=((0, 1),),
        tokenizer_revision="fixture-r1",
    )


def test_objective_replaces_omega_s_and_prepends_z() -> None:
    model = _TinyCausalModel()
    prompt = _editable_prompt()
    adapter = TransformerObjectiveAdapter(
        model,
        prompt=prompt,
        answer_anchor_ids=(torch.tensor([1, 2]),),
        refusal_anchor_ids=(torch.tensor([3, 4]),),
        epsilon=0.25,
        lambda_fol=0.5,
        gamma_z=0.1,
        gamma_u=0.2,
    )
    state = adapter.build_editable_state(torch.tensor([[5]]))

    full, mask = adapter.full_inputs(state)
    expected = torch.cat(
        (
            adapter.embedding(torch.tensor([[5]])),
            adapter.embedding(torch.tensor([[1]])),
            state.u[:, 0:1],
            adapter.embedding(torch.tensor([[3]])),
            state.u[:, 1:2],
        ),
        dim=1,
    )
    assert torch.allclose(full, expected)
    assert mask.tolist() == [[1, 1, 1, 1, 1]]


def test_adapter_builds_editable_embedding_state_and_freezes_model_parameters() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)

    state = adapter.build_editable_state(torch.tensor([[5]]))

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert state.z.requires_grad and state.u.requires_grad
    assert not state.z0.requires_grad and not state.u0.requires_grad
    assert torch.allclose(state.z, model.embeddings(torch.tensor([[5]])))
    assert torch.allclose(state.u, model.embeddings(torch.tensor([[2]])))
    assert state.z.data_ptr() != state.z0.data_ptr()
    assert state.u.data_ptr() != state.u0.data_ptr()


def test_adapter_evaluates_dual_anchor_risk_with_true_log_probs_and_fol() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    state = adapter.build_editable_state(torch.tensor([[5]]))

    value = adapter.evaluate(state, include_fol=True, fol_sign=1)

    gradients = torch.autograd.grad(value.maximize, (state.z, state.u))

    assert len(model.calls) == 1
    assert all(use_cache is False for _, _, use_cache in model.calls)
    assert all(torch.equal(mask, torch.ones((2, 4), dtype=torch.long)) for _, mask, _ in model.calls)
    assert torch.allclose(value.proxy_risk, value.answer_logp - value.refusal_logp)
    assert value.fol is not None and torch.isfinite(value.fol)
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0


def test_adapter_scores_batched_candidates_with_one_model_forward() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    state = adapter.build_editable_state(torch.tensor([[5], [6]]))

    scores = adapter.evaluate_candidates(state)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[:2] == (4, 4)
