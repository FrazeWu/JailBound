"""Transformer objective over a prompt with editable positions replaced in place."""

from __future__ import annotations

from typing import Any, Literal

import torch

from .anchor_scorer import score_continuation_sets
from .objective import EditableState, ObjectiveValue
from .prompt_contract import TokenizedEditablePrompt, scatter_editable


class TransformerAttackObjective:
    """Optimize a prefix ``z`` and in-place replacements ``u`` for one prompt."""

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
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.prompt = prompt
        self.answer_anchor_ids = self._anchor_ids(answer_anchor_ids, "answer")
        self.refusal_anchor_ids = self._anchor_ids(refusal_anchor_ids, "refusal")
        self.epsilon, self.lambda_fol = epsilon, lambda_fol
        self.gamma_z, self.gamma_u = gamma_z, gamma_u
        self._embedding_layer = model.get_input_embeddings()
        with torch.no_grad():
            embedded = self._embedding_layer(self._on_embedding_device(prompt.base_token_ids))
        self.base_prompt_embeddings = embedded.detach().clone()

    @property
    def embedding(self) -> Any:
        return self._embedding_layer

    @staticmethod
    def _anchor_ids(values: tuple[torch.Tensor, ...], label: str) -> tuple[torch.Tensor, ...]:
        result = tuple(values)
        if not result or any(ids.ndim != 1 or ids.numel() == 0 for ids in result):
            raise ValueError(f"{label} anchors require non-empty rank-1 token sequences")
        return tuple(ids.detach().clone().to(dtype=torch.long) for ids in result)

    def _on_embedding_device(self, ids: torch.Tensor) -> torch.Tensor:
        weight = getattr(self._embedding_layer, "weight", None)
        return ids.to(weight.device) if weight is not None else ids

    def _embed_editable(self, ids: torch.Tensor, label: str) -> torch.Tensor:
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError(f"{label} token IDs must have shape [batch, tokens]")
        with torch.no_grad():
            embedded = self._embedding_layer(self._on_embedding_device(ids.to(dtype=torch.long)))
        return embedded.detach().clone().requires_grad_(True)

    def build_editable_state(self, z_token_ids: torch.Tensor) -> EditableState:
        z = self._embed_editable(z_token_ids, "z")
        base = self.base_prompt_embeddings.to(device=z.device, dtype=z.dtype)
        if z.shape[0] > 1:
            base = base.expand(z.shape[0], -1, -1)
        positions = torch.tensor(self.prompt.editable_positions, device=base.device)
        u0 = base.index_select(1, positions)
        u = u0.detach().clone().requires_grad_(True)
        return EditableState(z=z, u=u, z0=z.detach().clone(), u0=u0.detach().clone())

    def build_state(self, z_token_ids: torch.Tensor) -> EditableState:
        return self.build_editable_state(z_token_ids)

    def full_inputs(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_prompt_embeddings.to(device=state.z.device, dtype=state.z.dtype)
        if base.shape[0] == 1 and state.z.shape[0] > 1:
            base = base.expand(state.z.shape[0], -1, -1)
        if state.z.shape[0] != base.shape[0] or state.u.shape[:2] != (base.shape[0], len(self.prompt.editable_positions)):
            raise ValueError("editable state does not match the tokenized prompt")
        replaced = scatter_editable(state.u, base, self.prompt.editable_positions)
        full = torch.cat((state.z, replaced), dim=1)
        return full, torch.ones(full.shape[:2], dtype=torch.long, device=full.device)

    def _score_vectors(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full, mask = self.full_inputs(state)
        scores = score_continuation_sets(
            model=self.model,
            embedding_layer=self._embedding_layer,
            prompt_embeds=full,
            prompt_attention_mask=mask,
            answer_anchors=self.answer_anchor_ids,
            refusal_anchors=self.refusal_anchor_ids,
        )
        return scores.proxy_risk, scores.answer_logp, scores.refusal_logp

    def _scores(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(score.mean() for score in self._score_vectors(state))  # type: ignore[return-value]

    def evaluate_candidates(self, state: EditableState) -> torch.Tensor:
        with torch.no_grad():
            proxy, _, _ = self._score_vectors(EditableState(state.z.detach(), state.u.detach(), state.z0, state.u0))
        penalties = self.gamma_z * (state.z - state.z0).square().sum(dim=(1, 2)) + self.gamma_u * (state.u - state.u0).square().sum(dim=(1, 2))
        return proxy - penalties

    def evaluate(self, state: EditableState, *, fol_sign: Literal[-1, 0, 1] = 0, include_fol: bool = False) -> ObjectiveValue:
        if fol_sign not in {-1, 0, 1}:
            raise ValueError("fol_sign must be -1, 0, or 1")
        proxy, answer, refusal = self._scores(state)
        attack = proxy - self.gamma_z * (state.z - state.z0).square().sum() - self.gamma_u * (state.u - state.u0).square().sum()
        fol = None
        maximize = attack
        if include_fol:
            gradients = torch.autograd.grad(attack, (state.z, state.u), create_graph=True, retain_graph=True)
            fol = self.epsilon * torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            maximize = attack + fol_sign * self.lambda_fol * fol
        return ObjectiveValue(maximize, attack, proxy, fol, answer, refusal, proxy)

    def hvp(self, state: EditableState, direction: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        gradients = torch.autograd.grad(self.evaluate(state).attack_loss, (state.z, state.u), create_graph=True)
        return torch.autograd.grad(sum((left * right).sum() for left, right in zip(gradients, direction)), (state.z, state.u))


TransformerObjectiveAdapter = TransformerAttackObjective
