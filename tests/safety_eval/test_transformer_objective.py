from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.safety_eval.transformer_objective import TransformerObjectiveAdapter
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.optimizers.base import BudgetLedger, CheckpointEmitter
from benchmark.safety_eval.optimizers.gcg import GCGOptimizer


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
        frozen_prompt_token_ids=torch.tensor([[1, 2]]),
        answer_token_ids=torch.tensor([1, 2]),
        refusal_token_ids=torch.tensor([3, 4]),
        epsilon=0.25,
        lambda_fol=0.5,
        gamma_z=0.1,
        gamma_u=0.2,
    )


def test_adapter_builds_editable_embedding_state_and_freezes_model_parameters() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)

    state = adapter.build_editable_state(torch.tensor([[5]]), torch.tensor([[6]]))

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert state.z.requires_grad and state.u.requires_grad
    assert not state.z0.requires_grad and not state.u0.requires_grad
    assert torch.allclose(state.z, model.embeddings(torch.tensor([[5]])))
    assert torch.allclose(state.u, model.embeddings(torch.tensor([[6]])))
    assert state.z.data_ptr() != state.z0.data_ptr()
    assert state.u.data_ptr() != state.u0.data_ptr()


def test_adapter_evaluates_dual_anchor_risk_with_true_log_probs_and_fol() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    state = adapter.build_editable_state(torch.tensor([[5]]), torch.tensor([[6]]))

    value = adapter.evaluate(state, include_fol=True, fol_sign=1)

    full_embeds = torch.cat((state.z, adapter.frozen_prompt_embeddings, state.u), dim=1)
    expected_log_probs = F.log_softmax(model.projection(full_embeds.cumsum(dim=1))[:, -1, :], dim=-1)
    expected_answer = expected_log_probs[:, [1, 2]].mean()
    expected_refusal = expected_log_probs[:, [3, 4]].mean()
    gradients = torch.autograd.grad(value.maximize, (state.z, state.u))

    assert len(model.calls) == 2
    assert all(use_cache is False for _, _, use_cache in model.calls)
    assert all(torch.equal(mask, torch.ones((1, 4), dtype=torch.long)) for _, mask, _ in model.calls)
    assert torch.allclose(value.answer_logp, expected_answer)
    assert torch.allclose(value.refusal_logp, expected_refusal)
    assert torch.allclose(value.proxy_risk, expected_answer - expected_refusal)
    assert value.fol is not None and torch.isfinite(value.fol)
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0


def test_adapter_scores_batched_candidates_with_one_model_forward() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    state = adapter.build_editable_state(torch.tensor([[5], [6]]), torch.tensor([[6], [5]]))

    scores = adapter.evaluate_candidates(state)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[:2] == (2, 4)


def test_adapter_optimization_loss_matches_attack_loss_with_one_model_forward() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    state = adapter.build_editable_state(torch.tensor([[5], [6]]), torch.tensor([[6], [5]]))

    expected = adapter.evaluate(state, include_fol=False).attack_loss.detach()
    model.calls.clear()
    actual = adapter.optimization_loss(state)

    assert torch.allclose(actual, expected)
    assert len(model.calls) == 1
    gradients = torch.autograd.grad(actual, (state.z, state.u))
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0


def test_gcg_batches_transformer_candidates_and_records_actual_forward_count() -> None:
    model = _TinyCausalModel()
    adapter = _adapter(model)
    ids = torch.tensor([[5]], dtype=torch.long)
    initial = EditableState(z=ids, u=ids.clone(), z0=ids.clone(), u0=ids.clone())
    ledger = BudgetLedger(update_limit=1, candidate_limit=2)

    snapshots = GCGOptimizer(model.embeddings.weight.detach(), search_width=2, top_k=2).run(
        adapter, initial, ledger, CheckpointEmitter([0, 1])
    )

    # Initial and terminal diagnostics each take two forwards; coordinate
    # gradients take two, while the two hard candidates share one scorer pass.
    assert ledger.forward_passes == len(model.calls) == 7
    assert snapshots[-1].candidates_attempted == 2
