from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from benchmark.safety_eval.paper_v2_objective import PaperV2TransformerObjective
from benchmark.safety_eval.prompt_contract import TokenizedEditablePrompt


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(13)
        self.embedding = nn.Embedding(9, 4)
        self.projection = nn.Linear(4, 9, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool):
        assert use_cache is False
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def _prompt() -> TokenizedEditablePrompt:
    return TokenizedEditablePrompt(
        full_text="abcd",
        base_token_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        editable_positions=(1, 3),
        frozen_positions=(0, 2),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
        boundary_expansions=((1, 2), (3, 4)),
        tokenizer_revision="fixture-r1",
    )


def test_paper_v2_objective_replaces_u_at_omega_s_and_prepends_z() -> None:
    model = _TinyCausalModel()
    objective = PaperV2TransformerObjective(
        model,
        prompt=_prompt(),
        answer_anchor_ids=(torch.tensor([5, 6]),),
        refusal_anchor_ids=(torch.tensor([7, 8]),),
        epsilon=0.1,
        lambda_fol=0.1,
        gamma_z=0.01,
        gamma_u=0.01,
    )
    state = objective.build_editable_state(torch.tensor([[5]]))

    full, mask = objective.full_inputs(state)
    expected_base = torch.stack(
        (
            model.embedding.weight[1],
            state.u[0, 0],
            model.embedding.weight[3],
            state.u[0, 1],
        )
    ).unsqueeze(0)
    expected = torch.cat((state.z, expected_base), dim=1)

    assert torch.allclose(full, expected)
    assert mask.tolist() == [[1, 1, 1, 1, 1]]
    assert torch.allclose(state.u0, model.embedding(torch.tensor([[2, 4]])))


def test_paper_v2_objective_fol_is_joint_over_z_and_u() -> None:
    model = _TinyCausalModel()
    objective = PaperV2TransformerObjective(
        model,
        prompt=_prompt(),
        answer_anchor_ids=(torch.tensor([5, 6]),),
        refusal_anchor_ids=(torch.tensor([7, 8]),),
        epsilon=0.1,
        lambda_fol=0.1,
        gamma_z=0.01,
        gamma_u=0.01,
    )
    state = objective.build_editable_state(torch.tensor([[5]]))

    value = objective.evaluate(state, fol_sign=1, include_fol=True)
    gradients = torch.autograd.grad(value.maximize, (state.z, state.u))

    assert value.fol is not None and torch.isfinite(value.fol)
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0


def test_paper_v2_objective_preserves_training_mode_for_gradient_checkpointing() -> None:
    model = _TinyCausalModel()
    model.is_gradient_checkpointing = True
    model.train()

    PaperV2TransformerObjective(
        model,
        prompt=_prompt(),
        answer_anchor_ids=(torch.tensor([5, 6]),),
        refusal_anchor_ids=(torch.tensor([7, 8]),),
        epsilon=0.1,
        lambda_fol=0.1,
        gamma_z=0.01,
        gamma_u=0.01,
    )

    assert model.training is True


def test_paper_v2_objective_uses_eval_mode_without_gradient_checkpointing() -> None:
    model = _TinyCausalModel()
    model.train()

    PaperV2TransformerObjective(
        model,
        prompt=_prompt(),
        answer_anchor_ids=(torch.tensor([5, 6]),),
        refusal_anchor_ids=(torch.tensor([7, 8]),),
        epsilon=0.1,
        lambda_fol=0.1,
        gamma_z=0.01,
        gamma_u=0.01,
    )

    assert model.training is False
