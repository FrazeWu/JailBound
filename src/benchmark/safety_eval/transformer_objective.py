"""Transformer-backed dual-anchor objective for editable embedding states."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from objectives.safety_risk import compute_dual_anchor_risk

from .objective import EditableState, ObjectiveValue


class TransformerAttackObjective:
    """Evaluate editable ``z``/``u`` embedding blocks with a frozen causal LM.

    The input layout is ``[z | frozen prompt scaffold | u]``.  The scaffold is
    looked up once during construction, while ``build_state`` creates detached
    leaf embeddings that optimizers may update without changing model weights.
    """

    forward_passes_per_evaluation = 2

    def __init__(
        self,
        model: Any,
        *,
        frozen_prompt_token_ids: torch.Tensor,
        answer_token_ids: torch.Tensor,
        refusal_token_ids: torch.Tensor,
        epsilon: float,
        lambda_fol: float,
        gamma_z: float,
        gamma_u: float,
    ) -> None:
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

        self.answer_token_ids = self._anchor_ids(answer_token_ids, "answer")
        self.refusal_token_ids = self._anchor_ids(refusal_token_ids, "refusal")
        self.epsilon = epsilon
        self.lambda_fol = lambda_fol
        self.gamma_z = gamma_z
        self.gamma_u = gamma_u

        self._embedding_layer = self.model.get_input_embeddings()
        prompt_ids = self._token_ids(frozen_prompt_token_ids, "frozen prompt")
        with torch.no_grad():
            prompt_embeddings = self._embedding_layer(self._on_embedding_device(prompt_ids))
        self.frozen_prompt_embeddings = prompt_embeddings.detach().clone()

    @staticmethod
    def _token_ids(token_ids: torch.Tensor, label: str) -> torch.Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
            raise ValueError(f"{label} token IDs must have shape [batch, tokens]")
        if token_ids.dtype.is_floating_point or token_ids.dtype.is_complex:
            raise ValueError(f"{label} token IDs must be integral")
        return token_ids.to(dtype=torch.long)

    @staticmethod
    def _anchor_ids(token_ids: torch.Tensor, label: str) -> torch.Tensor:
        if token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError(f"{label} anchor token IDs must be a non-empty rank-1 tensor")
        if token_ids.dtype.is_floating_point or token_ids.dtype.is_complex:
            raise ValueError(f"{label} anchor token IDs must be integral")
        return token_ids.detach().clone().to(dtype=torch.long)

    def _on_embedding_device(self, token_ids: torch.Tensor) -> torch.Tensor:
        weight = getattr(self._embedding_layer, "weight", None)
        device = getattr(weight, "device", None)
        return token_ids.to(device=device) if device is not None else token_ids

    def _embed_editable(self, token_ids: torch.Tensor, label: str) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self._embedding_layer(self._on_embedding_device(self._token_ids(token_ids, label)))
        return embeddings.detach().clone().requires_grad_(True)

    def build_editable_state(self, z_token_ids: torch.Tensor, u_token_ids: torch.Tensor) -> EditableState:
        """Create the initial editable state from the model's frozen token embeddings."""
        z = self._embed_editable(z_token_ids, "z")
        u = self._embed_editable(u_token_ids, "u")
        frozen_batch = self.frozen_prompt_embeddings.shape[0]
        if u.shape[0] != z.shape[0] or (frozen_batch != 1 and z.shape[0] != frozen_batch):
            raise ValueError("z/u batches must match the frozen prompt batch, unless the prompt has batch size 1")
        return EditableState(z=z, u=u, z0=z.detach().clone(), u0=u.detach().clone())

    def build_state(self, z_token_ids: torch.Tensor, u_token_ids: torch.Tensor) -> EditableState:
        """Compatibility alias for :meth:`build_editable_state`."""
        return self.build_editable_state(z_token_ids, u_token_ids)

    def _full_inputs(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor]:
        if state.z.ndim != 3 or state.u.ndim != 3:
            raise ValueError("editable z and u tensors must have shape [batch, tokens, hidden]")
        frozen = self.frozen_prompt_embeddings.to(device=state.z.device, dtype=state.z.dtype)
        if frozen.shape[0] == 1 and state.z.shape[0] > 1:
            frozen = frozen.expand(state.z.shape[0], -1, -1)
        if state.z.shape[0] != frozen.shape[0] or state.u.shape[0] != frozen.shape[0]:
            raise ValueError("editable state batch does not match the frozen prompt")
        if state.z.shape[-1] != frozen.shape[-1] or state.u.shape[-1] != frozen.shape[-1]:
            raise ValueError("editable state hidden size does not match the frozen prompt")
        full_embeds = torch.cat((state.z, frozen, state.u), dim=1)
        attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
        return full_embeds, attention_mask

    @staticmethod
    def _last_log_probs(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = (attention_mask.sum(dim=1).long() - 1).clamp(min=0, max=logits.shape[1] - 1)
        last_logits = logits[torch.arange(logits.shape[0], device=logits.device), positions, :]
        return F.log_softmax(last_logits, dim=-1)

    def _true_anchor_log_probs(self, full_embeds: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=full_embeds.detach(),
                attention_mask=attention_mask,
                use_cache=False,
            )
            log_probs = self._last_log_probs(outputs.logits, attention_mask)
            answer = log_probs[:, self.answer_token_ids.to(log_probs.device)].mean()
            refusal = log_probs[:, self.refusal_token_ids.to(log_probs.device)].mean()
        return answer, refusal

    def evaluate_candidates(self, state: EditableState) -> torch.Tensor:
        """Score independent discrete candidates with one no-grad model forward.

        Each batch item represents a complete candidate.  This deliberately
        omits the diagnostic true-anchor pass and FOL, which GCG does not use
        while ranking its local coordinate proposals.
        """
        full_embeds, attention_mask = self._full_inputs(state)
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=full_embeds.detach(),
                attention_mask=attention_mask,
                use_cache=False,
            )
            log_probs = self._last_log_probs(outputs.logits, attention_mask)
            answer = log_probs[:, self.answer_token_ids.to(log_probs.device)].mean(dim=-1)
            refusal = log_probs[:, self.refusal_token_ids.to(log_probs.device)].mean(dim=-1)
        z_penalty = (state.z - state.z0).square().sum(dim=(1, 2))
        u_penalty = (state.u - state.u0).square().sum(dim=(1, 2))
        return answer - refusal - self.gamma_z * z_penalty - self.gamma_u * u_penalty

    def optimization_loss(self, state: EditableState) -> torch.Tensor:
        """Return the differentiable attack loss without diagnostic forwards."""
        full_embeds, attention_mask = self._full_inputs(state)
        proxy_risk = compute_dual_anchor_risk(
            model=self.model,
            perturbed_embeds=full_embeds,
            attention_mask=attention_mask,
            answer_token_ids=self.answer_token_ids.to(full_embeds.device),
            refusal_token_ids=self.refusal_token_ids.to(full_embeds.device),
        )
        return (
            proxy_risk
            - self.gamma_z * (state.z - state.z0).square().sum()
            - self.gamma_u * (state.u - state.u0).square().sum()
        )

    def evaluate(
        self,
        state: EditableState,
        *,
        fol_sign: Literal[-1, 0, 1] = 0,
        include_fol: bool = False,
    ) -> ObjectiveValue:
        """Return the dual-anchor objective, true anchor log-probabilities, and optional FOL."""
        if fol_sign not in {-1, 0, 1}:
            raise ValueError("fol_sign must be -1, 0, or 1")
        full_embeds, attention_mask = self._full_inputs(state)
        answer_ids = self.answer_token_ids.to(full_embeds.device)
        refusal_ids = self.refusal_token_ids.to(full_embeds.device)
        proxy_risk = compute_dual_anchor_risk(
            model=self.model,
            perturbed_embeds=full_embeds,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            refusal_token_ids=refusal_ids,
        )
        answer_logp, refusal_logp = self._true_anchor_log_probs(full_embeds, attention_mask)
        attack_loss = (
            proxy_risk
            - self.gamma_z * (state.z - state.z0).square().sum()
            - self.gamma_u * (state.u - state.u0).square().sum()
        )
        fol: torch.Tensor | None = None
        maximize = attack_loss
        if include_fol:
            if not state.z.requires_grad or not state.u.requires_grad:
                raise ValueError("FOL requires editable z and u tensors with gradients enabled")
            gradients = torch.autograd.grad(attack_loss, (state.z, state.u), create_graph=True, retain_graph=True)
            fol = self.epsilon * torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            maximize = attack_loss + fol_sign * self.lambda_fol * fol
        return ObjectiveValue(maximize, attack_loss, proxy_risk, fol, answer_logp, refusal_logp, proxy_risk)

    def hvp(self, state: EditableState, direction: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attack-loss Hessian-vector product for ``z`` and ``u``."""
        attack_loss = self.evaluate(state).attack_loss
        gradient = torch.autograd.grad(attack_loss, (state.z, state.u), create_graph=True)
        directional = sum((left * right).sum() for left, right in zip(gradient, direction))
        return torch.autograd.grad(directional, (state.z, state.u))


TransformerObjectiveAdapter = TransformerAttackObjective
