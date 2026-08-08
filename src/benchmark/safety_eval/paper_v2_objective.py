"""Paper-aligned transformer objective over ``[z; Phi_tilde(p; U)]``."""

from __future__ import annotations

from typing import Any, Literal

import torch

from .anchor_scorer import score_continuation_sets
from .objective import EditableState, ObjectiveValue
from .prompt_contract import TokenizedEditablePrompt, scatter_editable


class PaperV2TransformerObjective:
    forward_passes_per_evaluation = 1

    def __init__(
        self,
        model: Any,
        *,
        prompt: TokenizedEditablePrompt,
        answer_anchor_ids: tuple[torch.Tensor, ...],
        refusal_anchor_ids: tuple[torch.Tensor, ...],
        epsilon: float,
        lambda_fol: float,
        gamma_z: float,
        gamma_u: float,
    ) -> None:
        if not answer_anchor_ids or not refusal_anchor_ids:
            raise ValueError("paper-v2 objective requires answer and refusal anchors")
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if bool(getattr(self.model, "is_gradient_checkpointing", False)):
            self.model.train()
        else:
            self.model.eval()
        self.prompt = prompt
        self.answer_anchor_ids = answer_anchor_ids
        self.refusal_anchor_ids = refusal_anchor_ids
        self.epsilon = epsilon
        self.lambda_fol = lambda_fol
        self.gamma_z = gamma_z
        self.gamma_u = gamma_u
        self.embedding = self.model.get_input_embeddings()
        weight = getattr(self.embedding, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("model has no input embedding matrix")
        base_ids = prompt.base_token_ids.to(device=weight.device, dtype=torch.long)
        with torch.no_grad():
            self.base_prompt_embeddings = self.embedding(base_ids).detach().clone()

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.weight
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
            raise ValueError("z token IDs must have shape [batch, tokens]")
        with torch.no_grad():
            value = self.embedding(token_ids.to(device=weight.device, dtype=torch.long))
        return value.detach().clone().requires_grad_(True)

    def build_editable_state(self, z_token_ids: torch.Tensor) -> EditableState:
        z = self._embed(z_token_ids)
        index = torch.tensor(self.prompt.editable_positions, dtype=torch.long, device=z.device)
        u0 = self.base_prompt_embeddings.to(device=z.device, dtype=z.dtype).index_select(1, index)
        if z.shape[0] > 1:
            u0 = u0.expand(z.shape[0], -1, -1)
        u = u0.detach().clone().requires_grad_(True)
        return EditableState(z=z, u=u, z0=z.detach().clone(), u0=u0.detach().clone())

    def full_inputs(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor]:
        if state.z.ndim != 3 or state.u.ndim != 3 or state.z.shape[0] != state.u.shape[0]:
            raise ValueError("editable z/U must have matching [batch, tokens, hidden] shapes")
        base = self.base_prompt_embeddings.to(device=state.z.device, dtype=state.z.dtype)
        if state.z.shape[0] > 1:
            base = base.expand(state.z.shape[0], -1, -1)
        replaced = scatter_editable(state.u, base, self.prompt.editable_positions)
        full = torch.cat((state.z, replaced), dim=1)
        attention = torch.ones(full.shape[:2], dtype=torch.long, device=full.device)
        return full, attention

    def _scores(self, state: EditableState):
        full, attention = self.full_inputs(state)
        return score_continuation_sets(
            model=self.model,
            embedding_layer=self.embedding,
            prompt_embeds=full,
            prompt_attention_mask=attention,
            answer_anchors=self.answer_anchor_ids,
            refusal_anchors=self.refusal_anchor_ids,
        )

    def _per_candidate_attack(self, state: EditableState) -> tuple[torch.Tensor, Any]:
        scores = self._scores(state)
        z_penalty = (state.z - state.z0).square().sum(dim=(1, 2))
        u_penalty = (state.u - state.u0).square().sum(dim=(1, 2))
        attack = scores.proxy_risk - self.gamma_z * z_penalty - self.gamma_u * u_penalty
        return attack, scores

    def evaluate_candidates(self, state: EditableState) -> torch.Tensor:
        attack, _ = self._per_candidate_attack(state)
        return attack

    def optimization_loss(self, state: EditableState) -> torch.Tensor:
        attack, _ = self._per_candidate_attack(state)
        return attack.mean()

    def evaluate(
        self,
        state: EditableState,
        *,
        fol_sign: Literal[-1, 0, 1] = 0,
        include_fol: bool = False,
    ) -> ObjectiveValue:
        if fol_sign not in {-1, 0, 1}:
            raise ValueError("fol_sign must be -1, 0, or 1")
        per_candidate, scores = self._per_candidate_attack(state)
        attack = per_candidate.mean()
        proxy = scores.proxy_risk.mean()
        answer = scores.answer_logp.mean()
        refusal = scores.refusal_logp.mean()
        fol: torch.Tensor | None = None
        maximize = attack
        if include_fol:
            if not state.z.requires_grad or not state.u.requires_grad:
                raise ValueError("FOL requires editable z and U tensors with gradients enabled")
            gradients = torch.autograd.grad(attack, (state.z, state.u), create_graph=True, retain_graph=True)
            fol = self.epsilon * torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            maximize = attack + fol_sign * self.lambda_fol * fol
        return ObjectiveValue(maximize, attack, proxy, fol, answer, refusal, proxy)

    def hvp(
        self,
        state: EditableState,
        direction: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attack = self.evaluate(state).attack_loss
        gradients = torch.autograd.grad(attack, (state.z, state.u), create_graph=True)
        directional = sum((left * right).sum() for left, right in zip(gradients, direction))
        return torch.autograd.grad(directional, (state.z, state.u))
